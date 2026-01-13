#!/bin/bash
# Download super-resolution models from HuggingFace and convert to TensorRT
# Creates engines for common input resolutions (TensorRT requires fixed dimensions)
set -e

MODEL_DIR="${MODEL_DIR:-$HOME/ffmpeg_build/models}"
VENV_DIR="${MODEL_DIR}/.venv-trt"

# Default resolutions to build (input dimensions)
# Format: "WIDTHxHEIGHT" - the model will upscale from these dimensions
DEFAULT_RESOLUTIONS="1280x720 1920x1080"

# HuggingFace models
# Format: "repo_id:filename:architecture:scale"
# Architectures: rrdb (RRDBNet), compact (SRVGGNetCompact)
MODELS=(
    "ai-forever/Real-ESRGAN:RealESRGAN_x4.pth:rrdb:4"
    "Sirosky/upscaler-models:4xNomosWebPhoto_RealPLKSR.pth:plksr:4"
)

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Download and convert super-resolution models to TensorRT engines.

Options:
    -r, --resolutions   Space-separated resolutions (default: "$DEFAULT_RESOLUTIONS")
    -m, --model         HuggingFace model spec: "repo:file:arch:scale" (can repeat)
    -o, --output-dir    Output directory (default: $MODEL_DIR)
    -f, --fp32          Use FP32 precision (default: FP16)
    -l, --list          List available models and exit
    -h, --help          Show this help

Examples:
    # Default: download Real-ESRGAN x4 and build 720p/1080p engines
    $(basename "$0")

    # Build for specific resolutions
    $(basename "$0") -r "640x480 1280x720 1920x1080 2560x1440"

    # Use a specific HuggingFace model
    $(basename "$0") -m "ai-forever/Real-ESRGAN:RealESRGAN_x4.pth:rrdb:4"

Output:
    Creates .engine files named: <model>_<width>x<height>_fp16.engine
    Example: RealESRGAN_x4_1920x1080_fp16.engine

FFmpeg usage:
    ffmpeg -i input.mp4 -vf "dnn_processing=dnn_backend=8:model=\$MODEL_DIR/RealESRGAN_x4_1920x1080_fp16.engine" output.mp4
EOF
    exit 0
}

list_models() {
    echo "Available models:"
    echo ""
    for spec in "${MODELS[@]}"; do
        IFS=':' read -r repo file arch scale <<< "$spec"
        echo "  $file"
        echo "    Repo: $repo"
        echo "    Architecture: $arch, Scale: ${scale}x"
        echo ""
    done
    exit 0
}

# Parse arguments
RESOLUTIONS="$DEFAULT_RESOLUTIONS"
FP16="--fp16"
CUSTOM_MODELS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--resolutions) RESOLUTIONS="$2"; shift 2 ;;
        -m|--model) CUSTOM_MODELS+=("$2"); shift 2 ;;
        -o|--output-dir) MODEL_DIR="$2"; shift 2 ;;
        -f|--fp32) FP16=""; shift ;;
        -l|--list) list_models ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Use custom models if specified
if [ ${#CUSTOM_MODELS[@]} -gt 0 ]; then
    MODELS=("${CUSTOM_MODELS[@]}")
fi

mkdir -p "$MODEL_DIR"

# Setup Python environment with TensorRT
setup_python() {
    if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
        if python3 -c "import tensorrt" 2>/dev/null; then
            echo "Using existing TensorRT venv"
            return
        fi
    fi

    echo "Setting up Python environment with TensorRT..."
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    pip install -q --upgrade pip

    # Install PyTorch with CUDA support
    pip install -q torch --index-url https://download.pytorch.org/whl/cu124

    # Install TensorRT (requires CUDA toolkit installed)
    pip install -q tensorrt

    # Install other dependencies
    pip install -q onnx huggingface_hub

    echo "Python environment ready"
}

# Download model from HuggingFace
download_model() {
    local repo="$1"
    local filename="$2"
    local output_path="$MODEL_DIR/$filename"

    if [ -f "$output_path" ]; then
        echo "  Model already exists: $filename"
        return
    fi

    echo "  Downloading $filename from $repo..."
    python3 << PYTHON
from huggingface_hub import hf_hub_download
import shutil

path = hf_hub_download(repo_id="$repo", filename="$filename")
shutil.copy(path, "$output_path")
print(f"  Downloaded: $output_path")
PYTHON
}

# Build TensorRT engine
build_engine() {
    local model_path="$1"
    local arch="$2"
    local scale="$3"
    local width="$4"
    local height="$5"

    local model_name=$(basename "$model_path" .pth)
    local precision_suffix=""
    [ -n "$FP16" ] && precision_suffix="_fp16"
    local engine_path="$MODEL_DIR/${model_name}_${width}x${height}${precision_suffix}.engine"

    if [ -f "$engine_path" ]; then
        echo "    Engine exists: $(basename "$engine_path")"
        return
    fi

    echo "    Building: $(basename "$engine_path")..."

    python3 << PYTHON
import os
import sys
import tempfile
import torch
import onnx
import tensorrt as trt

# Model architectures
class ResidualDenseBlock(torch.nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(torch.nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(torch.nn.Module):
    def __init__(self, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        if scale == 2:
            self.pixel_unshuffle = torch.nn.PixelUnshuffle(2)
            self.conv_first = torch.nn.Conv2d(3 * 4, num_feat, 3, 1, 1)
        else:
            self.pixel_unshuffle = None
            self.conv_first = torch.nn.Conv2d(3, num_feat, 3, 1, 1)

        self.body = torch.nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = torch.nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.pixel_unshuffle is not None:
            x = self.pixel_unshuffle(x)
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))

class SRVGGNetCompact(torch.nn.Module):
    def __init__(self, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.body = torch.nn.ModuleList()
        self.body.append(torch.nn.Conv2d(3, num_feat, 3, 1, 1))
        for _ in range(num_conv):
            self.body.append(torch.nn.PReLU(num_parameters=num_feat))
            self.body.append(torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        self.body.append(torch.nn.PReLU(num_parameters=num_feat))
        self.body.append(torch.nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
        self.upsampler = torch.nn.PixelShuffle(upscale)
        self.upscale = upscale

    def forward(self, x):
        out = self.body[0](x)
        for layer in self.body[1:]:
            out = layer(out)
        out = self.upsampler(out)
        base = torch.nn.functional.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        return out + base

# Load model
arch = "$arch"
scale = int("$scale")
width = int("$width")
height = int("$height")
model_path = "$model_path"

print(f"    Loading {arch} model (scale={scale}x)...")

if arch == "rrdb":
    model = RRDBNet(scale=scale)
elif arch == "compact":
    model = SRVGGNetCompact(upscale=scale)
else:
    # Try to load as TorchScript
    try:
        model = torch.jit.load(model_path, map_location='cpu')
        print("    Loaded as TorchScript")
    except:
        raise ValueError(f"Unknown architecture: {arch}")

if arch in ("rrdb", "compact"):
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    if 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
    elif 'params' in state_dict:
        state_dict = state_dict['params']
    model.load_state_dict(state_dict, strict=False)

model.eval()

# Export to ONNX
with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
    onnx_path = f.name

print(f"    Exporting ONNX ({width}x{height})...")
dummy_input = torch.randn(1, 3, height, width)
torch.onnx.export(model, dummy_input, onnx_path,
                  input_names=['input'], output_names=['output'],
                  opset_version=17, do_constant_folding=True)

# Build TensorRT engine
print("    Building TensorRT engine...")
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open(onnx_path, 'rb') as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(f"    Parse error: {parser.get_error(i)}")
        sys.exit(1)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

fp16 = "$FP16" != ""
if fp16 and builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)

serialized = builder.build_serialized_network(network, config)
if serialized is None:
    print("    Failed to build engine")
    sys.exit(1)

with open("$engine_path", 'wb') as f:
    f.write(serialized)

os.unlink(onnx_path)
size_mb = os.path.getsize("$engine_path") / 1024 / 1024
print(f"    Saved: $(basename "$engine_path") ({size_mb:.1f} MB)")
PYTHON
}

# Main
echo "========================================"
echo "Super-Resolution Model Preparation"
echo "========================================"
echo ""
echo "Output directory: $MODEL_DIR"
echo "Resolutions: $RESOLUTIONS"
echo "Precision: $([ -n "$FP16" ] && echo "FP16" || echo "FP32")"
echo ""

setup_python

for spec in "${MODELS[@]}"; do
    IFS=':' read -r repo filename arch scale <<< "$spec"

    echo ""
    echo "Model: $filename"
    echo "  Repository: $repo"
    echo "  Architecture: $arch, Scale: ${scale}x"

    download_model "$repo" "$filename"

    model_path="$MODEL_DIR/$filename"

    for res in $RESOLUTIONS; do
        width="${res%x*}"
        height="${res#*x}"
        build_engine "$model_path" "$arch" "$scale" "$width" "$height"
    done
done

echo ""
echo "========================================"
echo "Complete!"
echo "========================================"
echo ""
echo "Engines created in: $MODEL_DIR"
ls -lh "$MODEL_DIR"/*.engine 2>/dev/null || echo "(no engines found)"
echo ""
echo "FFmpeg usage example:"
echo '  ffmpeg -i input.mp4 -vf "dnn_processing=dnn_backend=8:model=$MODEL_DIR/<engine>.engine" output.mp4'
