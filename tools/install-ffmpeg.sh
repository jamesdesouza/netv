#!/bin/bash
# Build ffmpeg from source with NVIDIA + AMD hardware acceleration
# Supports: NVENC, CUVID, AMF, VAAPI, and LibTorch (CUDA/ROCm) for SR
# https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu
set -e

# Build options
BUILD_LIBAOM=${BUILD_LIBAOM:-0}
BUILD_NV_HEADERS=${BUILD_NV_HEADERS:-1}
BUILD_AMF_HEADERS=${BUILD_AMF_HEADERS:-1}

# Enable LibTorch for DNN super-resolution
ENABLE_LIBTORCH=${ENABLE_LIBTORCH:-0}
# LibTorch backend: "cuda", "rocm", or "auto" (use what's in LIBTORCH_DIR)
LIBTORCH_BACKEND="${LIBTORCH_BACKEND:-auto}"

# Build paths
SRC_DIR="${SRC_DIR:-$HOME/ffmpeg_sources}"
BUILD_DIR="${BUILD_DIR:-$HOME/ffmpeg_build}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
LIBTORCH_DIR="${LIBTORCH_DIR:-}"

NPROC=$(nproc)

# Detect GPUs
HAS_NVIDIA=$(lspci | grep -qi nvidia && echo 1 || echo 0)
HAS_AMD=$(lspci | grep -qiE "radeon|instinct|1002:" && echo 1 || echo 0)
echo "Detected: NVIDIA=$HAS_NVIDIA AMD=$HAS_AMD"

# ============================================================================
# NVIDIA CUDA Repository
# ============================================================================
echo "Setting up NVIDIA CUDA..."
if ! dpkg -l cuda-keyring 2>/dev/null | grep -q ^ii; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  rm cuda-keyring_1.1-1_all.deb
  sudo apt-get update
fi
CUDA_VERSION=${CUDA_VERSION:-$(apt-cache search '^cuda-nvcc-[0-9]' | sed 's/cuda-nvcc-//' | cut -d' ' -f1 | sort -V | tail -1)}
echo "CUDA version: $CUDA_VERSION"

# ============================================================================
# AMD ROCm Repository
# ============================================================================
echo "Setting up AMD ROCm..."
if [ ! -f /etc/apt/sources.list.d/rocm.list ]; then
  sudo mkdir -p /etc/apt/keyrings
  curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
  UBUNTU_CODENAME=$(lsb_release -cs)
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.1.1 ${UBUNTU_CODENAME} main" | sudo tee /etc/apt/sources.list.d/rocm.list
  # Pin AMD repo higher to avoid conflicts with Ubuntu's older ROCm packages
  echo 'Package: *
Pin: origin repo.radeon.com
Pin-Priority: 600' | sudo tee /etc/apt/preferences.d/rocm-pin-600
  sudo apt-get update
fi
ROCM_VERSION="7.1.1"
echo "ROCm version: $ROCM_VERSION"

sudo apt install -y \
  autoconf \
  automake \
  build-essential \
  cmake \
  git-core \
  meson \
  nasm \
  ninja-build \
  pkg-config \
  texinfo \
  wget \
  yasm \
  libaom-dev \
  libass-dev \
  libdav1d-dev \
  libfdk-aac-dev \
  libffmpeg-nvenc-dev \
  libfontconfig1-dev \
  libfreetype6-dev \
  libsoxr-dev \
  libsrt-openssl-dev \
  libssl-dev \
  libwebp-dev \
  libzimg-dev \
  liblzma-dev \
  liblzo2-dev \
  libmp3lame-dev \
  libnuma-dev \
  libopus-dev \
  libsdl2-dev \
  libtool \
  libunistring-dev \
  libva-dev \
  libvdpau-dev \
  libvpl-dev \
  libvorbis-dev \
  libvpx-dev \
  libx264-dev \
  libx265-dev \
  libxcb-shm0-dev \
  libxcb-xfixes0-dev \
  libxcb1-dev \
  zlib1g-dev \
  cuda-nvcc-$CUDA_VERSION \
  cuda-cudart-dev-$CUDA_VERSION \
  rocm-hip-runtime \
  rocm-dev \
  libdrm-dev \
  libdrm-amdgpu1 \
  opencl-headers \
  ocl-icd-opencl-dev

# Add user to video/render groups for AMD GPU access
if ! groups | grep -q render; then
  sudo usermod -aG video,render $USER
  echo "Added $USER to video,render groups (reboot required for AMD GPU access)"
fi

mkdir -p "$SRC_DIR"

# libaom (optional - system package is usually sufficient)
if [ "$BUILD_LIBAOM" = "1" ]; then
  cd "$SRC_DIR" &&
  git -C aom pull 2> /dev/null || git clone --depth 1 https://aomedia.googlesource.com/aom &&
  mkdir -p aom_build &&
  cd aom_build &&
  PATH="$BIN_DIR:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" -DENABLE_TESTS=OFF -DENABLE_NASM=on ../aom &&
  PATH="$BIN_DIR:$PATH" make -j $NPROC &&
  make install
fi

# libsvtav1
cd "$SRC_DIR" && \
git -C SVT-AV1 pull 2> /dev/null || git clone --depth 1 https://gitlab.com/AOMediaCodec/SVT-AV1.git && \
mkdir -p SVT-AV1/build && \
cd SVT-AV1/build && \
PATH="$BIN_DIR:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DBUILD_DEC=OFF -DBUILD_SHARED_LIBS=OFF .. && \
PATH="$BIN_DIR:$PATH" make -j $NPROC && \
make install

# libvmaf
cd "$SRC_DIR" &&
git -C vmaf-master pull 2> /dev/null || git clone --depth 1 'https://github.com/Netflix/vmaf' 'vmaf-master' &&
mkdir -p 'vmaf-master/libvmaf/build' &&
cd 'vmaf-master/libvmaf/build' &&
if [ -f build.ninja ]; then
  meson setup --reconfigure -Denable_tests=false -Denable_docs=false --buildtype=release --default-library=static '../' --prefix "$BUILD_DIR" --bindir="$BIN_DIR" --libdir="$BUILD_DIR/lib"
else
  meson setup -Denable_tests=false -Denable_docs=false --buildtype=release --default-library=static '../' --prefix "$BUILD_DIR" --bindir="$BIN_DIR" --libdir="$BUILD_DIR/lib"
fi &&
ninja &&
ninja install


# nv-codec-headers (NVIDIA NVENC/NVDEC)
if [ "$BUILD_NV_HEADERS" = "1" ]; then
  cd "$SRC_DIR" &&
  git -C nv-codec-headers pull 2> /dev/null || git clone --depth 1 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git &&
  cd nv-codec-headers &&
  make &&
  make PREFIX="$BUILD_DIR" install
fi

# AMF headers (AMD Advanced Media Framework for h264_amf/hevc_amf encoders)
if [ "$BUILD_AMF_HEADERS" = "1" ]; then
  cd "$SRC_DIR" &&
  git -C AMF pull 2> /dev/null || git clone --depth 1 https://github.com/GPUOpen-LibrariesAndSDKs/AMF.git &&
  mkdir -p "$BUILD_DIR/include/AMF" &&
  cp -r AMF/amf/public/include/* "$BUILD_DIR/include/AMF/"
  echo "AMF headers installed to $BUILD_DIR/include/AMF"
fi

# LibTorch for DNN super-resolution
if [ "$ENABLE_LIBTORCH" = "1" ]; then
  if [ -n "$LIBTORCH_DIR" ]; then
    # Use provided path
    if [ ! -f "$LIBTORCH_DIR/lib/libtorch.so" ]; then
      echo "Error: libtorch.so not found at $LIBTORCH_DIR/lib/"
      exit 1
    fi
    echo "Using LibTorch at: $LIBTORCH_DIR"
  else
    # Download standalone LibTorch - pick backend based on available GPU or preference
    LIBTORCH_DIR="$BUILD_DIR/libtorch"
    LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.5.1}"

    # Auto-detect backend if not specified
    if [ "$LIBTORCH_BACKEND" = "auto" ]; then
      if [ "$HAS_AMD" = "1" ] && [ "$HAS_NVIDIA" = "0" ]; then
        LIBTORCH_BACKEND="rocm"
      else
        LIBTORCH_BACKEND="cuda"
      fi
    fi

    if [ ! -f "$LIBTORCH_DIR/lib/libtorch.so" ]; then
      cd "$SRC_DIR"
      if [ "$LIBTORCH_BACKEND" = "rocm" ]; then
        # ROCm LibTorch
        LIBTORCH_ROCM="${LIBTORCH_ROCM:-rocm6.2}"
        echo "Downloading LibTorch ${LIBTORCH_VERSION} (${LIBTORCH_ROCM})..."
        LIBTORCH_URL="https://download.pytorch.org/libtorch/${LIBTORCH_ROCM}/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${LIBTORCH_ROCM}.zip"
      else
        # CUDA LibTorch
        LIBTORCH_CUDA="${LIBTORCH_CUDA:-cu124}"
        echo "Downloading LibTorch ${LIBTORCH_VERSION} (${LIBTORCH_CUDA})..."
        LIBTORCH_URL="https://download.pytorch.org/libtorch/${LIBTORCH_CUDA}/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${LIBTORCH_CUDA}.zip"
      fi
      wget -q --show-progress -O "libtorch.zip" "$LIBTORCH_URL"
      echo "Extracting..."
      unzip -q -o "libtorch.zip" -d "$BUILD_DIR"
      rm "libtorch.zip"
    fi
    echo "Using LibTorch at: $LIBTORCH_DIR (backend: $LIBTORCH_BACKEND)"
  fi
fi

# ============================================================================
# GPU Encoder Flags
# ============================================================================
# NVIDIA: NVENC/NVDEC
CUDA_FLAGS="--enable-cuda-nvcc --enable-nvenc --enable-cuvid"
NVCC_GENCODE=""
if command -v nvidia-smi &> /dev/null; then
  COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
  if [ -n "$COMPUTE_CAP" ]; then
    COMPUTE_CAP_NUM=$(echo $COMPUTE_CAP | tr -d '.')
    NVCC_GENCODE="-gencode arch=compute_${COMPUTE_CAP_NUM},code=sm_${COMPUTE_CAP_NUM}"
    echo "Detected NVIDIA GPU: compute ${COMPUTE_CAP} (sm_${COMPUTE_CAP_NUM})"
  fi
fi

# AMD: AMF (always enable if headers are installed)
AMF_FLAGS=""
if [ -d "$BUILD_DIR/include/AMF" ]; then
  AMF_FLAGS="--enable-amf"
  echo "AMD AMF headers found - enabling h264_amf/hevc_amf encoders"
fi

# ffmpeg
cd "$SRC_DIR"
if [ ! -d "ffmpeg" ]; then
  wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
  tar xjvf ffmpeg-snapshot.tar.bz2
fi
cd ffmpeg

# Patch for PyTorch 2.9+ API compatibility (initXPU -> init)
if [ "$ENABLE_LIBTORCH" = "1" ] && grep -q "initXPU()" libavfilter/dnn/dnn_backend_torch.cpp 2>/dev/null; then
  # Check PyTorch version from version.py
  TORCH_VERSION=$(grep -oP "__version__ = '\K[0-9]+\.[0-9]+" "$LIBTORCH_DIR/version.py" 2>/dev/null || echo "0.0")
  TORCH_MAJOR=$(echo "$TORCH_VERSION" | cut -d. -f1)
  TORCH_MINOR=$(echo "$TORCH_VERSION" | cut -d. -f2)
  if [ "$TORCH_MAJOR" -ge 2 ] && [ "$TORCH_MINOR" -ge 9 ] 2>/dev/null; then
    echo "Detected PyTorch $TORCH_VERSION - patching dnn_backend_torch.cpp for API compatibility..."
    sed -i 's/initXPU()/init()/' libavfilter/dnn/dnn_backend_torch.cpp
  fi
fi
# Build configure flags
EXTRA_CFLAGS="-I$BUILD_DIR/include -I/usr/local/cuda/include -O3 -march=native -mtune=native"
EXTRA_LDFLAGS="-L$BUILD_DIR/lib -L/usr/local/cuda/lib64 -s"
EXTRA_LIBS="-lpthread -lm"

# Add LibTorch paths if enabled
LIBTORCH_FLAGS=""
EXTRA_CXXFLAGS=""
if [ "$ENABLE_LIBTORCH" = "1" ] && [ -d "$LIBTORCH_DIR" ]; then
  echo "Configuring ffmpeg with LibTorch support..."
  LIBTORCH_INCLUDES="-I$LIBTORCH_DIR/include -I$LIBTORCH_DIR/include/torch/csrc/api/include"
  EXTRA_CFLAGS="$EXTRA_CFLAGS $LIBTORCH_INCLUDES"
  EXTRA_CXXFLAGS="$LIBTORCH_INCLUDES"
  EXTRA_LDFLAGS="$EXTRA_LDFLAGS -L$LIBTORCH_DIR/lib"
  LIBTORCH_FLAGS="--enable-libtorch"
  # LibTorch needs these at runtime
  export LD_LIBRARY_PATH="$LIBTORCH_DIR/lib:$LD_LIBRARY_PATH"
fi

CONFIGURE_CMD=(
  ./configure
  --prefix="$BUILD_DIR"
  --pkg-config-flags="--static"
  --extra-cflags="$EXTRA_CFLAGS"
  --extra-cxxflags="$EXTRA_CXXFLAGS"
  --extra-ldflags="$EXTRA_LDFLAGS"
  --extra-libs="$EXTRA_LIBS"
  --ld="g++"
  --bindir="$BIN_DIR"
  --enable-gpl
  --enable-version3
  --enable-openssl
  --enable-libaom
  --enable-libass
  --enable-libfdk-aac
  --enable-libfontconfig
  --enable-libfreetype
  --enable-libmp3lame
  --enable-libopus
  --enable-libsvtav1
  --enable-libdav1d
  --enable-libvmaf
  --enable-libvorbis
  --enable-libvpx
  --enable-libwebp
  --enable-libx264
  --enable-libx265
  --enable-libzimg
  --enable-libsoxr
  --enable-libsrt
  --enable-vaapi
  --enable-libvpl
  --enable-nonfree
  --enable-opencl
  $CUDA_FLAGS
  $AMF_FLAGS
  $LIBTORCH_FLAGS
)

if [ -n "$NVCC_GENCODE" ]; then
  CONFIGURE_CMD+=(--nvccflags="$NVCC_GENCODE")
fi

PATH="$BIN_DIR:$PATH" PKG_CONFIG_PATH="$BUILD_DIR/lib/pkgconfig" "${CONFIGURE_CMD[@]}" && \
PATH="$BIN_DIR:$PATH" make -j $NPROC && \
make install && \
hash -r

grep -q "$BUILD_DIR/share/man" "$HOME/.manpath" 2>/dev/null || echo "MANPATH_MAP $BIN_DIR $BUILD_DIR/share/man" >> "$HOME/.manpath"

# If LibTorch was enabled, set up environment and print instructions
if [ "$ENABLE_LIBTORCH" = "1" ] && [ -d "$LIBTORCH_DIR" ]; then
  # Add to shell profile if not already present
  LIBTORCH_ENV="export LD_LIBRARY_PATH=\"$LIBTORCH_DIR/lib:\$LD_LIBRARY_PATH\""
  if ! grep -q "libtorch" "$HOME/.bashrc" 2>/dev/null; then
    echo "" >> "$HOME/.bashrc"
    echo "# LibTorch for ffmpeg DNN super-resolution" >> "$HOME/.bashrc"
    echo "$LIBTORCH_ENV" >> "$HOME/.bashrc"
  fi

  echo ""
  echo "============================================"
  echo "LibTorch enabled! To use super-resolution:"
  echo "============================================"
  echo ""
  echo "1. Reload your shell or run:"
  echo "   $LIBTORCH_ENV"
  echo ""
  echo "2. Download/convert an SR model (e.g., Real-ESRGAN):"
  echo "   See: tools/download-sr-models.sh"
  echo ""
  echo "3. Use with ffmpeg:"
  echo "   ffmpeg -i input.mp4 -vf \"dnn_processing=dnn_backend=torch:model=realesrgan_x2.pt\" output.mp4"
  echo ""
fi

# Cleanup notes:
# rm -rf ~/ffmpeg_build ~/.local/bin/{ffmpeg,ffprobe,ffplay,x264,x265}
# sed -i '/ffmpeg_build/d' ~/.manpath
# sed -i '/libtorch/d' ~/.bashrc
# hash -r
# Debug: cat ~/ffmpeg_sources/ffmpeg/ffbuild/config.log
