#!/usr/bin/env bash
set -euo pipefail

# Build FlashAttention-2 wheel
# Expected env: TORCH_CUDA_ARCH_LIST, MAX_JOBS (set in Dockerfile)

export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE

# Embed CUDA version in wheel filename (e.g. cu128, cu130)
# CUDA_VERSION is set by the nvidia/cuda base image (e.g. 12.8.1)
CUDA_TAG="cu$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr -d '.')"

git clone --branch v2.8.3 --depth 1 https://github.com/Dao-AILab/flash-attention.git
cd flash-attention

uv build --wheel --no-build-isolation --out-dir /output

# Rename wheel to include CUDA version tag (e.g. flash_attn-2.8.3+cu128-...)
for whl in /output/flash_attn-*.whl; do
    new=$(echo "$whl" | sed "s/flash_attn-\([^-]*\)-/flash_attn-\1+${CUDA_TAG}-/")
    mv "$whl" "$new"
done

echo "FA2 wheel built successfully:"
ls -lh /output/*.whl
