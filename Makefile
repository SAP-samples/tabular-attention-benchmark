.PHONY: test-base test-cudnn test-fa2 test-fa3 test-fa4 test-sage test-vllm test-all \
        benchmark benchmark-h100 plots

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

test-all: test-base test-cudnn test-fa2 test-fa3 test-fa4

# ── Benchmark infrastructure ──────────────────────────────────────────────────
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

REP  := 50
REP_128 := 25

# headdim:nheads pairs to benchmark (order matters for step numbering)
SHAPES := 32:6 16:8 32:8 64:8 64:12 64:16 64:32 128:8 128:12 128:16
# FA4 backward does not support headdim=16 on Blackwell (sm_100)
SHAPES_NO_HD16 := 32:6 64:12 64:16 64:32 128:8 128:12 128:16

# Number of steps per backend: len(SHAPES) * 2 (col + row)
STEPS_PER_BACKEND := 14
BACKENDS := sdpa_efficient sdpa_cudnn fa2 fa3 fa4

# Attention sweep parameters
COL_ATTN_ROWS := 1024
COL_ATTN_COLS := 16 32 64 128 256 512 1024 2048
ROW_ATTN_ROWS := 32 64 128 256 512 1024 2048 4096 8192 16384
ROW_ATTN_COLS := 64

# run_bench GROUP BACKEND [SHAPES_VAR]
#   Runs all (shape × direction) combos for a single backend.
#   GROUP  = uv dependency group (base, fa2, fa3, fa4)
#   BACKEND = backend name passed to --backends
#   Reps are chosen per headdim: REP_64 for hd=64, REP_128 for hd=128.
define run_bench
	for shape in $(if $(3),$(3),$(SHAPES)); do \
		hd=$${shape%%:*}; nh=$${shape##*:}; \
		if [ "$$hd" = "128" ]; then rep=$(REP_128); else rep=$(REP); fi; \
		for dir in col row; do \
			step=$$((step + 1)); \
			flag="--$${dir}-only"; \
			printf "\n[%3d/$(STEPS_PER_BACKEND)] %-18s hd=%-3s n=%-3s %s rep=%-3s (%s)\n" \
				$$step "$(2)" "$$hd" "$$nh" "$$dir" "$$rep" "$$(date '+%H:%M:%S')"; \
			uv run --group $(1) python run_benchmark.py --rep $$rep \
				--backends $(2) --headdim $$hd --nheads $$nh \
				--col-attn-rows $(COL_ATTN_ROWS) --col-attn-cols $(COL_ATTN_COLS) \
				--row-attn-rows $(ROW_ATTN_ROWS) --row-attn-cols $(ROW_ATTN_COLS) \
				$$flag || true; \
		done; \
	done
endef

# Run all benchmarks
benchmark: bench-sdpa-efficient bench-sdpa-cudnn bench-fa2 bench-fa3 bench-fa4

# H100: all backends
benchmark-h100: bench-sdpa-efficient bench-sdpa-cudnn bench-fa2 bench-fa3 bench-fa4


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

# ── Plots ────────────────────────────────────────────────────────────────────

PLOT_CMD := uv run python run_benchmark_plots.py

plots:
	$(PLOT_CMD)
	$(PLOT_CMD) --gpu-comparison --nheads 12 --headdim 64
	$(PLOT_CMD) --inference-only --nheads 12 --headdim 64
	$(PLOT_CMD) --headdim-ablation --nheads 8
	$(PLOT_CMD) --roofline --gpu NVIDIA_H100_NVL --nheads 12 --headdim 64
