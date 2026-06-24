#!/usr/bin/env bash
set -euo pipefail

# Build FlashAttention-3 wheel (lives in hopper/ subdirectory)
# Expected env: TORCH_CUDA_ARCH_LIST, MAX_JOBS (set in Dockerfile)

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout fbe15683a881743c2625b931cc4abcc107b43154

# Embed CUDA version in wheel filename (e.g. cu128, cu130)
# CUDA_VERSION is set by the nvidia/cuda base image (e.g. 12.8.1)
CUDA_TAG="cu$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr -d '.')"

cd hopper/
uv build --wheel --no-build-isolation --out-dir /output

# Rename wheel to include CUDA version tag (e.g. flash_attn_3-3.0.0+cu130-...)
for whl in /output/flash_attn_3-*.whl; do
    new=$(echo "$whl" | sed "s/flash_attn_3-\([^-]*\)-/flash_attn_3-\1+${CUDA_TAG}-/")
    mv "$whl" "$new"
done

echo "FA3 wheel built successfully:"
ls -lh /output/*.whl
