# Tabular Attention Benchmark
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/tabular-attention-benchmark)](https://api.reuse.software/info/github.com/SAP-samples/tabular-attention-benchmark)


Benchmarking different attention implementations for tabular data.

Tabular tensors have shape `(batch, rows, cols, nheads, headdim)`. We benchmark two attention patterns:
- **Column attention**: attend across columns (`seq_len=cols`, `batch_eff=batch*rows`) — memory layout is already contiguous
- **Row attention**: attend across rows (`seq_len=rows`, `batch_eff=batch*cols`) — requires transpose; some backends (FA3, FA4) can operate on strided tensors directly.

If you use our work, please cite our corresponding publication:
```
@article{Schambach:2026:BenchmarkingAttention,
  title = {Benchmarking Attention for Tabular Foundation Models},
  year = {2026},
  author = {Schambach, Maximilian and Biehl, Clemens and Thelin, Sam},
  journal = {2nd ICML Workshop on Foundation Models for Structured Data},
}
```

## Backends

| Backend          | Module      | Dependency group | Description                                                   |
|------------------|-------------|------------------|---------------------------------------------------------------|
| `sdpa_efficient` | `base.py`   | `base`           | PyTorch SDPA with xformers efficient attention                |
| `sdpa_cudnn`     | `base.py`   | `base`           | PyTorch SDPA with cuDNN attention                             |
| `fa2`            | `fa2.py`    | `fa2`            | FlashAttention-2 (includes KV-packed variants)                |
| `fa3`            | `fa3.py`    | `fa3`            | FlashAttention-3                                              |
| `fa4`            | `fa4.py`    | `fa4`            | FlashAttention-4 (requires Hopper/Blackwell, SM 9.0+)         |

Each backend has its own dependency group in `pyproject.toml` with a pinned torch version. The groups are mutually exclusive (declared as conflicts in `[tool.uv]`) since they require different torch/CUDA builds.

## Project Structure

```
tabular_attn/
├── __init__.py        # Re-exports all backends
├── base.py            # SDPA backends (math, efficient, cuDNN)
├── fa2.py             # FlashAttention-2
├── fa3.py             # FlashAttention-3
├── fa4.py             # FlashAttention-4
run_benchmark.py       # Benchmark runner
run_benchmark_plots.py # Plot generation from benchmark results
build/                 # Docker images and pre-built wheel scripts
ablation/layout/       # Layout ablation (b,R,C,h,d vs b,C,R,h,d) investigation
```

## Installation
Depending on your system, it might be necessary to update your Nvidia drivers to support evaluation of all CUDA 13-linked compiled kernels.
For Ubuntu systems, please follow the below instructions, for other Linux distributions, please consult the corresponding official guidelines.

### NVIDIA Driver (Ubuntu 22.04)

```bash
# Add NVIDIA repo
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Remove any existing driver to avoid conflicts
sudo apt remove --purge 'nvidia-*-535' 'libnvidia-*-535'
sudo apt autoremove --purge

# Install driver
sudo apt install nvidia-driver-580 cuda-toolkit-13-0
sudo reboot
```

### NVIDIA Driver (Ubuntu 24.04)

```bash
# Add NVIDIA repo
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Remove any existing driver to avoid conflicts
sudo apt remove --purge 'nvidia-*' 'libnvidia-*'
sudo apt autoremove --purge

# Install driver
sudo apt install nvidia-driver-580
sudo reboot
```

Verify driver version after reboot:
```bash
nvidia-smi
```

### Pre-built Wheels

FA3 and SageAttention are distributed as pre-built wheels placed under `build/wheels/`. 
Prebuilt wheels are attached to the releases on GitHub.
However, you can also build them from scratch yourself:
They are built inside Docker containers to match the exact CUDA/torch ABI. To rebuild them:

```bash
cd build/
make build-fa3     # FA3 wheel  (CUDA 13.0 image)
```

The FA2 wheel is pulled from the official release pages, however you may optionally also compile it from scratch:
```bash
cd build/
make build-fa2     # FA2 wheel  (CUDA 12.8 image)
```

This requires Docker with GPU support (`nvidia-container-toolkit`). See `build/README.md` for Docker setup instructions.

### Python / uv

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Dependencies are installed automatically by each `make test-*` and `make bench-*` target via `uv sync --reinstall` to accomodate different CUDA requirements per backend. 


## Running Tests

Each backend is tested independently with its own dependency group. Tests use pytest markers and a `--backend` filter to select the right backend per run.

```bash
make test-base       # SDPA efficient (torch 2.10, CUDA 13.0)
make test-cudnn      # SDPA cuDNN (torch 2.10, CUDA 13.0)
make test-fa2        # FlashAttention-2 (torch 2.8, CUDA 12.8)
make test-fa3        # FlashAttention-3 (torch 2.10, CUDA 13.0)
make test-fa4        # FlashAttention-4 (torch 2.10, CUDA 13.0, Hopper+ only)
make test-all        # Run all test suites sequentially
```

Each target runs `uv sync --reinstall` before testing to ensure the correct torch/CUDA installs and socket files.

## Running Benchmarks

Benchmarks run one process per `(backend, headdim, nheads, direction)` combination. Results are flushed to disk after every shape so a mid-run crash only loses the shape in-flight.

Depending on your GPU, run one of the following:
```bash
make benchmark           # All backends
make benchmark-h100      # H100 (all backends)
```

Results are saved to `results/{GPU}/{dtype}/{backend}/`.

## Plotting
To reproduce the plots used in the paper and create a custom plot for every evaluated GPU and shape combination, run

```bash
make plots
```

This generates all plots under `plots/`:

1. **Per-GPU combined plots** — 2×2 grids (col/row × fwd/bwd) for each GPU and config
2. **Per-GPU speedup plots** — relative to SDPA (efficient) baseline
3. **GPU comparison** (`--gpu-comparison`) — separate PDFs per GPU (A100, H100, B200) showing forward-pass col/row attention for H=12, D=64
4. **Inference-only** (`--inference-only`) — H100 forward-pass including vLLM and SageAttention
5. **Head dimension ablation** (`--headdim-ablation`) — FA3 vs cuDNN across D={16, 64, 128} on H100
6. **Roofline analysis** (`--roofline`) — memory-bound ceiling for column attention and copy-overhead decomposition for row attention (uses measured copy bandwidth from `results/copy_bandwidth.json`)


## Trouble-Shooting
### Failed to generate package metadata
If you encounter an error such as
```bash
error: Failed to generate package metadata for `flash-attn-3==3.0.0 @ path+build/wheels/flash_attn_3-3.0.0+cu130-cp39-abi3-linux_x86_64.whl`
  Caused by: Failed to read from the distribution cache
  Caused by: failed to query metadata of file `<...>/tabular-attention-benchmark/build/wheels/flash_attn_3-3.0.0+cu130-cp39-abi3-linux_x86_64.whl`: No such file or directory (os error 2)
```
then the necessary wheel (in this case, flash-attn-3) is not available at the required location (`./build/wheels`).
To resolve thism either download the pre-built wheel from our GitHub release page, or build the required wheel from scratch.
For this, refer to the README in the `./build` folder.

### Hash missmatch
If you encounter a hash mismatch such as
```bash
× Failed to read `flash-attn-3 @
  │ file:///<...>/tabular-attention-benchmark/build/wheels/flash_attn_3-3.0.0+cu130-cp39-abi3-linux_x86_64.whl`
  ╰─▶ Hash mismatch for `flash-attn-3 @
      file:///<...>/tabular-attention-benchmark/build/wheels/flash_attn_3-3.0.0+cu130-cp39-abi3-linux_x86_64.whl`

      Expected:
        sha256:60f63e7421bc72af76837996ea520cbb51dd928d05fc466de4ab62f9f4b7c7bb

      Computed:
        sha256:<...>
```
this may be because you built the wheel from scratch. The existing lock file includes the hash of the pre-built wheel available in the GitHub release.
To resolve this error, you may simply call
```bash
uv lock --upgrade-package flash-attn-3
```
to update only the effected dependency and keep the other as specified and used during the benchmark.

## How to obtain support
[Create an issue](https://github.com/SAP-samples/<repository-name>/issues) in this repository if you find a bug or have questions about the content.
 
For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## Contributing
If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License
Copyright 2026 SAP SE or an SAP affiliate company and tabular-attention-benchmark contributors. Please see our [LICENSE](LICENSE) for copyright and license information. Detailed information including third-party components and their licensing/copyright information is available [via the REUSE tool](https://api.reuse.software/info/github.com/SAP-samples/tabular-attention-benchmark).
