FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG FASTSAM3D_PATH_ARG=/workspace
ARG MAX_JOBS=4
ARG INSTALL_FLASH_ATTN=0
ENV FASTSAM3D_PATH=${FASTSAM3D_PATH_ARG}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    VIRTUAL_ENV=/opt/venv \
    PATH=/root/.local/bin:$PATH \
    XDG_RUNTIME_DIR=/tmp/runtime-root \
    OMNI_KIT_ALLOW_ROOT=1 \
    TORCH_HOME=/workspace/checkpoints/torch-cache \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121" \
    PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf autoconf-archive automake bison flex gperf m4 meson ninja-build \
    build-essential pkg-config \
    git curl unzip zip tar \
    python3 python3-dev python3-pip python3-venv \
    cmake ffmpeg \
    xauth x11-utils \
    libdbus-1-3 libglib2.0-0 libsm6 libfontconfig1 libfreetype6 \
    libx11-6 libx11-dev libxext6 libxext-dev libxrender1 libxrender-dev \
    libxcb1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render0 \
    libxcb-render-util0 libxcb-shape0 libxcb-shm0 libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 \
    libxkbcommon0 libxkbcommon-x11-0 libxkbfile1 libxmu6 libxaw7 libxxf86dga1 \
    libgl1 libgl1-mesa-dev libopengl0 libglu1-mesa-dev freeglut3-dev \
    libegl1 libglew-dev \
    vulkan-tools zenity \
    && rm -rf /var/lib/apt/lists/*

# Python toolchain
RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && python3 -m venv ${VIRTUAL_ENV}

ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir hatchling hatch-requirements-txt

WORKDIR ${FASTSAM3D_PATH}

# Optional training/quantization/desktop dependencies; not required for Fast-SAM3D inference in this image.
RUN sed -i '/^auto_gptq==/d' requirements.txt
RUN sed -i '/^bpy==/d' requirements.txt
RUN sed -i '/^bitsandbytes==/d' requirements.txt
RUN sed -i '/^cuda-python==/d' requirements.txt
RUN sed -i '/^nvidia-cuda-nvcc-cu12==/d' requirements.txt
RUN sed -i '/^dataclasses==/d' requirements.txt
RUN sed -i 's/^open3d==0.18.0/open3d==0.19.0/' requirements.txt
RUN sed -i '/^nvidia-pyindex==/d' requirements.txt
RUN sed -i '/^point-cloud-utils==/d' requirements.txt
RUN sed -i '/^pycocotools==/d' requirements.txt
RUN sed -i '/^sagemaker==/d' requirements.txt
RUN sed -i '/^mosaicml-streaming==/d' requirements.txt

# Install uv (optional utility used in other setups)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install PyTorch first (required before CUDA extensions)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install package dependencies for inference (dev extras are intentionally skipped in Docker)
RUN pip install --no-cache-dir --no-build-isolation '.[inference]'
RUN pip install --no-cache-dir "numpy>=2,<2.3.0"

# Install dependencies that are intentionally commented out in requirements files
RUN pip install --no-cache-dir \
    "git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900" \
    "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"

RUN TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS=${MAX_JOBS} pip install --no-cache-dir \
    "git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7" \
    --no-build-isolation

RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS=${MAX_JOBS} \
    pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47" \
    --no-build-isolation

RUN if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then \
      MAX_JOBS=${MAX_JOBS} pip install --no-cache-dir flash_attn==2.8.3 --no-build-isolation; \
    else \
      echo "Skipping flash_attn installation (INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN})."; \
    fi

# Patch hydra compatibility (required by upstream README)
RUN python patching/hydra

RUN mkdir -p /workspace/checkpoints/torch-cache /workspace/outputs /tmp/runtime-root

CMD ["/bin/bash"]
