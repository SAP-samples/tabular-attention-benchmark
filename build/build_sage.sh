#!/usr/bin/env bash
set -euo pipefail

# Build SageAttention wheel
# Expected env: TORCH_CUDA_ARCH_LIST, MAX_JOBS (set in Dockerfile)

git clone --branch v2.2.0 --depth 1 https://github.com/thu-ml/SageAttention.git
cd SageAttention

# Embed CUDA version in wheel filename (e.g. cu128, cu130)
# CUDA_VERSION is set by the nvidia/cuda base image (e.g. 12.8.1)
CUDA_TAG="cu$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr -d '.')"

# SM90+ only: SageAttention uses wgmma instructions not available on SM80
export TORCH_CUDA_ARCH_LIST="9.0;10.0"

uv build --wheel --no-build-isolation --out-dir /output

# Rename wheel to include CUDA version tag (e.g. sageattention-2.2.0+cu130-...)
for whl in /output/sageattention-*.whl; do
    new=$(echo "$whl" | sed "s/sageattention-\([^-]*\)-/sageattention-\1+${CUDA_TAG}-/")
    mv "$whl" "$new"
done

echo "SageAttention wheel built successfully:"
ls -lh /output/*.whl
