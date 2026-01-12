#!/bin/bash
# Compile Real-ESRGAN model with TensorRT for faster inference
# Requires: SM 7.0+ GPU (Volta, Turing, Ampere, Ada, Blackwell)
set -e

MODEL_DIR="${MODEL_DIR:-$HOME/ffmpeg_build/models}"
VENV_DIR="${VENV_DIR:-$MODEL_DIR/.venv}"
MODEL="${MODEL:-realesr-general-x4v3.pt}"
LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.9.0}"
LIBTORCH_VARIANT="${LIBTORCH_VARIANT:-cu130}"
INPUT_PATH="$MODEL_DIR/$MODEL"
OUTPUT_PATH="$MODEL_DIR/${MODEL%.pt}-trt.pt"

echo "=== TensorRT Model Compiler ==="

# Check GPU
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU_NAME"

# Check SM version (need 7.0+)
SM_VERSION=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
if [ "$SM_VERSION" -lt 70 ]; then
    echo "ERROR: TensorRT 10.x requires SM 7.0+ (Volta or newer)"
    echo "Your GPU has SM $SM_VERSION"
    exit 1
fi

# Check model
if [ ! -f "$INPUT_PATH" ]; then
    echo "ERROR: Model not found: $INPUT_PATH"
    echo "Run ./tools/download-sr-models.sh first"
    exit 1
fi

# Check/create venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install dependencies if needed
TORCH_INSTALLED=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
if [ "$TORCH_INSTALLED" != "${LIBTORCH_VERSION}+${LIBTORCH_VARIANT}" ] || ! python3 -c "import torch_tensorrt" 2>/dev/null; then
    echo "Installing torch ${LIBTORCH_VERSION} and torch-tensorrt..."
    TMPDIR="${TMPDIR:-/tmp}"
    [ -d "$HOME/tmp" ] && TMPDIR="$HOME/tmp"
    mkdir -p "$TMPDIR"
    TMPDIR="$TMPDIR" pip install -q "torch==${LIBTORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${LIBTORCH_VARIANT}"
    TMPDIR="$TMPDIR" pip install -q torch-tensorrt
fi

echo "Input: $INPUT_PATH"
echo "Output: $OUTPUT_PATH"
echo ""

# Compile
python3 << EOF
import torch
import torch_tensorrt
import time

print(f"PyTorch: {torch.__version__}")
print(f"torch-tensorrt: {torch_tensorrt.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# Load model
model = torch.jit.load("$INPUT_PATH").cuda().eval()
print("Model loaded")

# Compile with TensorRT
print("Compiling with TensorRT (FP16)... this may take a few minutes")
start = time.time()

trt_model = torch_tensorrt.compile(
    model,
    ir="torchscript",
    inputs=[torch_tensorrt.Input(shape=(1, 3, 720, 1280), dtype=torch.float32)],
    enabled_precisions={torch.float16, torch.float32},
    truncate_long_and_double=True,
)

elapsed = time.time() - start
print(f"Compilation took {elapsed:.1f}s")

# Save using torch_tensorrt's method
torch_tensorrt.save(trt_model, "$OUTPUT_PATH", output_format="torchscript")
print(f"Saved: $OUTPUT_PATH")

# Benchmark
print()
print("Benchmarking...")
x = torch.rand(1, 3, 720, 1280).cuda()

# Warmup
for _ in range(3):
    with torch.no_grad():
        _ = trt_model(x)
torch.cuda.synchronize()

# Timed runs
runs = 10
start = time.time()
for _ in range(runs):
    with torch.no_grad():
        _ = trt_model(x)
torch.cuda.synchronize()
elapsed = time.time() - start

fps = runs / elapsed
print(f"TensorRT FP16: {fps:.1f} fps ({1000*elapsed/runs:.1f} ms/frame)")
print(f"Real-time 30fps: {'YES' if fps >= 30 else 'NO'} ({fps/30:.2f}x)")
EOF

echo ""
echo "=== Done ==="
echo ""
echo "To use with ffmpeg, you need libtorchtrt_runtime.so"
echo "Download from: https://github.com/pytorch/TensorRT/releases"
