.PHONY: test-base test-cudnn test-fa2 test-fa3 test-fa4 test-sage test-vllm test-all \
        bench-e2e bench-e2e-fa2 bench-e2e-fa3 bench-e2e-sage \
        benchmark benchmark-a100 benchmark-h100 benchmark-gb200 \
        bench-gb200-sdpa-efficient bench-gb200-sdpa-cudnn bench-gb200-fa2 bench-gb200-fa4 bench-gb200-fa4-optim bench-gb200-vllm \
        bench-fa4-optim bench-batch plots plots-tabarena

test-base:
	uv sync --group dev --group base --reinstall
	uv run --group dev --group base pytest --backend base --verbose

test-cudnn:
	uv sync --group dev --group base --reinstall
	uv run --group dev --group base pytest --backend cudnn --verbose

test-fa2:
	uv sync --group dev --group fa2 --reinstall
	uv run --group dev --group fa2 pytest --backend fa2 --verbose

test-fa3:
	uv sync --group dev --group fa3 --reinstall
	uv run --group dev --group fa3 pytest --backend fa3 --verbose

test-fa4:
	uv sync --group dev --group fa4 --reinstall
	uv run --group dev --group fa4 pytest --backend fa4 --verbose

test-sage:
	uv sync --group dev --group sage --reinstall
	uv run --group dev --group sage pytest --backend sage --verbose

test-vllm:
	uv sync --group dev --group vllm --reinstall
	uv run --group dev --group vllm pytest --backend vllm --verbose

test-all: test-base test-cudnn test-fa2 test-fa3 test-fa4 test-sage test-vllm
# One process per (backend, headdim, nheads, direction) so a CUDA crash in one
# run doesn't kill others. Results are flushed to disk after every shape, so a
# mid-run crash only loses the shape in-flight.
# As results are increasing in size always, this will catch OOM and still save previous results
#
# Backend → dependency group mapping:
#   sdpa_efficient, sdpa_cudnn  → base
#   fa2                         → fa2
#   fa3, optim                  → fa3
#   fa4                         → fa4
#   fa4_optim                   → fa4_optim
#   sage                        → sage

# Head geometries and their sweeps live in sweeps.py (single source of truth),
# queried at run time via `python -m sweeps ...`. Per-GPU exclusions (e.g. GB200
# skips headdim=16) and the total step count are computed there too.
BACKENDS := sdpa_efficient sdpa_cudnn fa2 fa3 fa4 fa4_optim sage vllm

# Optional overrides (default: use sweeps.py schedule / skip-existing):
#   make bench-fa2 FORCE=1          -> recompute all shapes
#   make bench-fa2 REP_OVERRIDE=10  -> fixed reps instead of the per-shape schedule
FORCE ?=
REP_OVERRIDE ?=

# run_bench GROUP BACKEND [GPU_KEY]
#   Runs all (shape × configured-direction) combos for a single backend, one
#   process each so a CUDA crash is isolated. Geometries, directions, ranges,
#   reps, and the step total all come from sweeps.py (filtered by GPU_KEY, e.g.
#   gb200). Runs are resumable (skip-existing) by default.
define run_bench
	total=$$(uv run python -m sweeps --count $(if $(3),--gpu $(3),)); \
	step=0; \
	for shape in $$(uv run python -m sweeps --list-shapes $(if $(3),--gpu $(3),)); do \
		hd=$${shape%%:*}; nh=$${shape##*:}; \
		for dir in $$(uv run python -m sweeps --dirs $$hd $$nh); do \
			step=$$((step + 1)); \
			flag="--$${dir}-only"; \
			printf "\n[%3d/%s] %-18s hd=%-3s n=%-3s %s (%s)\n" \
				$$step "$$total" "$(2)" "$$hd" "$$nh" "$$dir" "$$(date '+%H:%M:%S')"; \
			uv run --group $(1) python run_benchmark.py \
				--backends $(2) --headdim $$hd --nheads $$nh \
				$(if $(REP_OVERRIDE),--rep $(REP_OVERRIDE),) $(if $(FORCE),--force,) \
				$$flag || true; \
		done; \
	done
endef

# Run all benchmarks
benchmark: bench-sdpa-efficient bench-sdpa-cudnn bench-fa2 bench-fa3 bench-fa4 bench-fa4-optim bench-sage bench-vllm bench-optim

# A100: no fa4 (requires Hopper+)
benchmark-a100: bench-sdpa-efficient bench-sdpa-cudnn bench-fa2 bench-fa3 bench-vllm bench-optim

# H100: all backends
benchmark-h100: bench-sdpa-efficient bench-sdpa-cudnn bench-fa2 bench-fa3 bench-fa4 bench-sage bench-vllm

# GB200 (Blackwell): no fa3/sage/optim (no sm_100 kernels), no headdim=16 (FA4 bwd bug)
benchmark-gb200: bench-gb200-sdpa-efficient bench-gb200-sdpa-cudnn bench-gb200-fa2 bench-gb200-fa4 bench-gb200-fa4-optim bench-gb200-vllm

bench-sdpa-efficient:
	uv sync --group base --reinstall
	$(call run_bench,base,sdpa_efficient)

bench-sdpa-cudnn:
	uv sync --group base --reinstall
	$(call run_bench,base,sdpa_cudnn)

bench-fa2:
	uv sync --group fa2 --reinstall
	$(call run_bench,fa2,fa2)

bench-fa3:
	uv sync --group fa3 --reinstall
	$(call run_bench,fa3,fa3)

bench-fa4:
	uv sync --group fa4 --reinstall
	$(call run_bench,fa4,fa4)

bench-fa4-optim:
	uv sync --group fa4_optim --reinstall
	$(call run_bench,fa4_optim,fa4_optim)

bench-sage:
	uv sync --group sage --reinstall
	$(call run_bench,sage,sage)

bench-vllm:
	uv sync --group vllm --reinstall
	$(call run_bench,vllm,vllm)

# ── GB200 (Blackwell) bench targets — sweeps.py excludes headdim=16 for gb200 ──

bench-gb200-sdpa-efficient:
	uv sync --group base --reinstall
	$(call run_bench,base,sdpa_efficient,gb200)

bench-gb200-sdpa-cudnn:
	uv sync --group base --reinstall
	$(call run_bench,base,sdpa_cudnn,gb200)

bench-gb200-fa2:
	uv sync --group fa2 --reinstall
	$(call run_bench,fa2,fa2,gb200)

bench-gb200-fa4:
	uv sync --group fa4 --reinstall
	$(call run_bench,fa4,fa4,gb200)

bench-gb200-fa4-optim:
	uv sync --group fa4_optim --reinstall
	$(call run_bench,fa4_optim,fa4_optim,gb200)

bench-gb200-vllm:
	uv sync --group vllm --reinstall
	$(call run_bench,vllm,vllm,gb200)

# ── Batch-size equivalence sweep ──────────────────────────────────────────────
# Sweeps B over the fixed geometry/ranges defined in sweeps.py BATCH_SWEEP, for
# both directions, one process per (batch, direction). Validates that batch
# affects throughput only via batch_eff (results overlay the B=1 curve).
# Backend group defaults to base/sdpa_cudnn; override: make bench-batch BATCH_GROUP=fa2 BATCH_BACKEND=fa2
BATCH_GROUP ?= base
BATCH_BACKEND ?= sdpa_cudnn

bench-batch:
	uv sync --group $(BATCH_GROUP) --reinstall
	hd=$$(uv run python -m sweeps --batch-sweep-field headdim); \
	nh=$$(uv run python -m sweeps --batch-sweep-field nheads); \
	col_rows=$$(uv run python -m sweeps --batch-sweep-field col_attn_rows); \
	row_cols=$$(uv run python -m sweeps --batch-sweep-field row_attn_cols); \
	col_cols=$$(uv run python -m sweeps --batch-sweep-field col_attn_cols); \
	row_rows=$$(uv run python -m sweeps --batch-sweep-field row_attn_rows); \
	for b in $$(uv run python -m sweeps --batch-sweep-field batches); do \
		printf "\n=== batch=%s hd=%s nh=%s %s ===\n" "$$b" "$$hd" "$$nh" "$(BATCH_BACKEND)"; \
		uv run --group $(BATCH_GROUP) python run_benchmark.py \
			--backends $(BATCH_BACKEND) --headdim $$hd --nheads $$nh --batch $$b \
			--col-only --col-attn-rows $$col_rows --col-attn-cols $$col_cols || true; \
		uv run --group $(BATCH_GROUP) python run_benchmark.py \
			--backends $(BATCH_BACKEND) --headdim $$hd --nheads $$nh --batch $$b \
			--row-only --row-attn-cols $$row_cols --row-attn-rows $$row_rows || true; \
	done

# ── E2E: TabPFNv2.6 inference benchmark across attention backends ────────────
# Sweeps full predict_proba latency across the same shape ranges as the
# attention microbenchmark. One target per dependency group / backend:
#   bench-e2e       SDPA modes (default dispatch, efficient, cudnn)  [cu128]
#   bench-e2e-fa2   "our" FA2 wrapper (native (B,S,H,D), no permute) [cu128]
#   bench-e2e-fa3   "our" FA3 wrapper                                [cu130]
#   bench-e2e-sage  SageAttention                                    [cu130]
# Each target runs both --query single (1 test row, prediction latency) and
# --query batched (1024 test rows, throughput); results land under
# results/e2e/{single,batched}-query/. The uv sync runs once since both
# queries share the group.
bench-e2e:
	uv sync --group e2e --reinstall
	PYTHONPATH=e2e uv run --group e2e python e2e/run_e2e_benchmark.py --query single
	PYTHONPATH=e2e uv run --group e2e python e2e/run_e2e_benchmark.py --query batched

bench-e2e-fa2:
	uv sync --group e2e_fa2 --reinstall
	PYTHONPATH=e2e uv run --group e2e_fa2 python e2e/run_e2e_benchmark.py --modes fa --query single
	PYTHONPATH=e2e uv run --group e2e_fa2 python e2e/run_e2e_benchmark.py --modes fa --query batched

bench-e2e-fa3:
	uv sync --group e2e_fa3 --reinstall
	PYTHONPATH=e2e uv run --group e2e_fa3 python e2e/run_e2e_benchmark.py --modes fa --query single
	PYTHONPATH=e2e uv run --group e2e_fa3 python e2e/run_e2e_benchmark.py --modes fa --query batched

bench-e2e-sage:
	uv sync --group e2e_sage --reinstall
	PYTHONPATH=e2e uv run --group e2e_sage python e2e/run_e2e_benchmark.py --modes sage --query single
	PYTHONPATH=e2e uv run --group e2e_sage python e2e/run_e2e_benchmark.py --modes sage --query batched

# ── Plots ────────────────────────────────────────────────────────────────────
# Plotting only needs matplotlib/seaborn/numpy/pandas — run outside the project
# env (--no-project) so it doesn't trigger a backend torch/flash_attn resolution.
PLOT_CMD := uv run --no-project --with matplotlib --with seaborn --with numpy --with pandas python run_benchmark_plots.py

plots:
	$(PLOT_CMD)
	$(PLOT_CMD) --agent-optimized
	$(PLOT_CMD) --gpu-comparison --nheads 12 --headdim 64
	$(PLOT_CMD) --inference-only --nheads 12 --headdim 64
	$(PLOT_CMD) --inference-only --nheads 6 --headdim 32
	$(PLOT_CMD) --headdim-ablation --nheads 8
	$(PLOT_CMD) --roofline --gpu NVIDIA_H100_NVL --nheads 12 --headdim 64
	$(PLOT_CMD) --batch-sweep --gpu NVIDIA_H100_NVL --backend sdpa_cudnn --nheads 12 --headdim 64
	$(PLOT_CMD) --batch-sweep --gpu NVIDIA_H100_NVL --backend sdpa_cudnn --nheads 6 --headdim 32
	uv run --no-project --with matplotlib --with seaborn --with numpy --with pandas python e2e/run_e2e_plots.py

plots-tabarena:
	uv run --no-project --with matplotlib --with seaborn --with numpy --with pandas python e2e/run_tabarena_plots.py
