"""Sweep configuration for the tabular attention microbenchmark.

This module owns *what* is benchmarked and *how many reps* each shape gets. The
runner (run_benchmark.py) and the Makefile consume it so that ranges and rep
counts live in one place instead of being threaded through CLI args.

Pure data + helpers — no torch import, so it stays importable everywhere cheaply.

Two attention patterns (see CLAUDE.md):
  - column attention: seq_len = cols   (fixed rows, varying cols)
  - row attention:    seq_len = rows   (fixed cols, varying rows)
"""

# ── Default sweep ranges ──────────────────────────────────────────────────────
COL_ATTN_ROWS = 1024                                     # fixed rows for col attn
COL_ATTN_COLS = [16, 32, 64, 128, 256, 512, 1024, 2048]  # swept cols (= col seqlen)
ROW_ATTN_COLS = 64                                       # fixed cols for row attn
ROW_ATTN_ROWS_DEFAULT = [                                # swept rows (= row seqlen)
    32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
    16384, 32768, 65536, 98304, 131072,
]

# ── Per-shape config, keyed (headdim, nheads) ─────────────────────────────────
# Each entry may set:
#   "directions"     : subset of ["col", "row"] to run (default both)
#   "row_max_rows"    : cap the row ladder at this seqlen (heavy shapes OOM sooner)
#   "row_attn_rows"   : fully override the row ladder
#   "col_attn_cols"   : override swept cols
#   "col_attn_rows"   : override fixed row count for col attn
#   "row_attn_cols"   : override fixed col count for row attn
#   "exclude_gpus"    : list of GPU keys to skip this shape on (e.g. ["gb200"])
# Row caps come from the empirically-measured OOM boundary on an ~95 GB H100:
# heavier (headdim, nheads) exhaust memory at lower row counts.
# Order matters — it defines the launch/step order.
SHAPES = {
    (32, 6):   {"directions": ["col", "row"]},
    # FA4 backward does not support headdim=16 on Blackwell (sm_100) → skip on GB200.
    (16, 8):   {"directions": ["col", "row"], "exclude_gpus": ["gb200"]},
    (32, 8):   {"directions": ["col", "row"]},
    (64, 8):   {"directions": ["col", "row"]},
    (64, 12):  {"directions": ["col", "row"]},
    (64, 16):  {"directions": ["col", "row"], "row_max_rows": 65536},
    (64, 32):  {"directions": ["col", "row"], "row_max_rows": 32768},
    (128, 8):  {"directions": ["col", "row"], "row_max_rows": 32768},
    (128, 12): {"directions": ["col", "row"], "row_max_rows": 16384},
    (128, 16): {"directions": ["col", "row"], "row_max_rows": 16384},
}

# ── Repetition schedule ───────────────────────────────────────────────────────
# (seqlen_threshold, reps): the first threshold >= seqlen wins. Fast/small shapes
# get many reps for stable timing; huge/slow shapes get few so the sweep finishes.
# Applies to BOTH col (seqlen=cols) and row (seqlen=rows).
REP_SCHEDULE = [
    (1024, 50),
    (8192, 20),
    (32768, 5),
]
REP_DEFAULT = 3   # seqlen larger than the last threshold
WARMUP = 3

# ── Batch-size equivalence sweep ──────────────────────────────────────────────
# One fixed geometry + fixed rows/cols, swept over batch. Validates that batch
# affects throughput only through batch_eff = B*rows (col) / B*cols (row) — i.e.
# B>1 lands on the same throughput-vs-batch_eff curve as the B=1 sequence sweep,
# so fixing B=1 is without loss of generality. Row attention is the one to watch
# (its .contiguous() transpose could in principle break the invariance).
BATCH_SWEEP = {
    "headdim": 64,
    "nheads": 12,
    "batches": [2, 4, 8],
    "col_attn_rows": 512,   # batch_eff = B*512 -> 1024, 2048, 4096 (on the B=1 curve)
    "row_attn_cols": 64,    # batch_eff = B*64  -> 128, 256, 512
    "col_attn_cols": [16, 32, 64, 128, 256, 512, 1024, 2048],
    "row_attn_rows": [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
}


def reps_for_seqlen(seqlen):
    """Repetition count for a shape, from its sequence length."""
    for threshold, reps in REP_SCHEDULE:
        if seqlen <= threshold:
            return reps
    return REP_DEFAULT


def _cfg(headdim, nheads):
    return SHAPES.get((headdim, nheads), {})


def directions_for(headdim, nheads):
    """Which attention directions to run for this shape (default both)."""
    return _cfg(headdim, nheads).get("directions", ["col", "row"])


def row_ladder_for(headdim, nheads):
    """Row-count ladder for row attention, filtered by any row_max_rows cap."""
    cfg = _cfg(headdim, nheads)
    ladder = cfg.get("row_attn_rows", ROW_ATTN_ROWS_DEFAULT)
    cap = cfg.get("row_max_rows")
    return [r for r in ladder if cap is None or r <= cap]


def col_cols_for(headdim, nheads):
    return _cfg(headdim, nheads).get("col_attn_cols", COL_ATTN_COLS)


def col_rows_for(headdim, nheads):
    return _cfg(headdim, nheads).get("col_attn_rows", COL_ATTN_ROWS)


def row_cols_for(headdim, nheads):
    return _cfg(headdim, nheads).get("row_attn_cols", ROW_ATTN_COLS)


def shape_list(gpu=None):
    """Head geometries to launch, as (headdim, nheads) pairs, in launch order.

    `gpu` (e.g. "gb200") filters out shapes whose "exclude_gpus" names it, so a
    single source of truth here drives per-GPU shape selection in the Makefile.
    """
    out = []
    for (hd, nh), cfg in SHAPES.items():
        if gpu is not None and gpu in cfg.get("exclude_gpus", []):
            continue
        out.append((hd, nh))
    return out


def task_count(gpu=None):
    """Total (shape × configured-direction) tasks — the launch-step denominator."""
    return sum(len(directions_for(hd, nh)) for hd, nh in shape_list(gpu))


if __name__ == "__main__":
    # Tiny CLI so the Makefile can introspect the config without duplicating it:
    #   python -m sweeps --list-shapes             -> "32:6 16:8 32:8 ..."
    #   python -m sweeps --list-shapes --gpu gb200 -> excludes GB200-unsupported
    #   python -m sweeps --dirs 64 12              -> "col row"  (for the loop)
    #   python -m sweeps --count [--gpu gb200]     -> total task count (denominator)
    import argparse

    p = argparse.ArgumentParser(description="Sweep config introspection")
    p.add_argument("--list-shapes", action="store_true",
                   help="Print headdim:nheads geometries (space-separated)")
    p.add_argument("--dirs", nargs=2, type=int, metavar=("HEADDIM", "NHEADS"),
                   help="Print configured directions for a shape (space-separated)")
    p.add_argument("--count", action="store_true",
                   help="Print total (shape x configured-direction) task count")
    p.add_argument("--batch-sweep-field", default=None,
                   help="Print a BATCH_SWEEP field (headdim/nheads/batches/...)")
    p.add_argument("--gpu", default=None,
                   help="GPU key for per-GPU shape exclusions (e.g. gb200)")
    a = p.parse_args()

    if a.list_shapes:
        print(" ".join(f"{hd}:{nh}" for hd, nh in shape_list(a.gpu)))
    elif a.dirs is not None:
        print(" ".join(directions_for(a.dirs[0], a.dirs[1])))
    elif a.count:
        print(task_count(a.gpu))
    elif a.batch_sweep_field is not None:
        val = BATCH_SWEEP[a.batch_sweep_field]
        print(" ".join(str(x) for x in val) if isinstance(val, list) else val)
