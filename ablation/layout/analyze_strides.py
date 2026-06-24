"""Visualize stride patterns for the two layouts to understand memory access.

No GPU needed — just prints the stride analysis.
"""

def analyze_strides(shape, name):
    """Manually compute strides for a C-contiguous tensor of given shape."""
    ndim = len(shape)
    strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]

    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"  Shape:   {shape}")
    print(f"  Strides: {tuple(strides)} (in elements)")

    # After transpose(1, 2):
    new_shape = list(shape)
    new_shape[1], new_shape[2] = new_shape[2], new_shape[1]
    new_strides = list(strides)
    new_strides[1], new_strides[2] = new_strides[2], new_strides[1]

    print(f"\n  After .transpose(1, 2):")
    print(f"  Shape:   {tuple(new_shape)}")
    print(f"  Strides: {tuple(new_strides)} (in elements)")

    # The contiguous copy needs to read in the transposed order
    # and write sequentially. The key question: what's the inner loop pattern?
    #
    # The innermost contiguous chunk is determined by how many trailing
    # dimensions have the same strides as a contiguous tensor would.

    # Compute what contiguous strides would look like for the transposed shape:
    target_strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        target_strides[i] = target_strides[i + 1] * new_shape[i + 1]

    print(f"  Target contiguous strides: {tuple(target_strides)}")

    # Find how many trailing dims already match
    matching = 0
    for i in range(ndim - 1, -1, -1):
        if new_strides[i] == target_strides[i]:
            matching += 1
        else:
            break

    contiguous_inner = 1
    for i in range(ndim - 1, ndim - 1 - matching, -1):
        contiguous_inner *= new_shape[i]

    print(f"\n  Trailing contiguous dims: {matching}")
    print(f"  Contiguous inner block: {contiguous_inner} elements = {contiguous_inner * 2} bytes (fp16)")

    # The non-contiguous dimension's stride tells us the "jump" between blocks
    if matching < ndim:
        nc_dim = ndim - 1 - matching  # first non-contiguous dim from the right
        jump = new_strides[nc_dim]
        expected = target_strides[nc_dim]
        print(f"\n  First non-contiguous dim: {nc_dim} (size={new_shape[nc_dim]})")
        print(f"    Actual stride:   {jump} elements = {jump * 2} bytes")
        print(f"    Expected stride: {expected} elements = {expected * 2} bytes")
        print(f"    Stride ratio (actual/expected): {jump / expected:.2f}")

    return new_strides, tuple(target_strides), contiguous_inner


def main():
    batch = 2
    nheads = 8
    hdim = 64

    configs = [
        (1024, 20),
        (2048, 50),
        (4096, 50),
        (8192, 100),
    ]

    for rows, cols in configs:
        print(f"\n{'#'*70}")
        print(f"# rows={rows}, cols={cols}, batch={batch}, nheads={nheads}, hdim={hdim}")
        print(f"# Total elements: {batch * rows * cols * nheads * hdim:,}")
        print(f"{'#'*70}")

        # Layout A: (batch, rows, cols, nheads, hdim)
        # Transpose for row attention -> swap rows and cols
        a_strides, a_target, a_inner = analyze_strides(
            (batch, rows, cols, nheads, hdim),
            "Layout A: (batch, rows, cols, nheads, hdim) -> transpose(1,2) for ROW attn"
        )

        # Layout B: (batch, cols, rows, nheads, hdim)
        # Transpose for col attention -> swap cols and rows
        b_strides, b_target, b_inner = analyze_strides(
            (batch, cols, rows, nheads, hdim),
            "Layout B: (batch, cols, rows, nheads, hdim) -> transpose(1,2) for COL attn"
        )

        print(f"\n  ** Both have inner block = {a_inner} = {b_inner} elements **")
        print(f"  ** Same total elements, same inner contiguity **")

        # Now the interesting part: what does the READ pattern look like?
        # When copying to contiguous, we iterate in target order.
        # For each "row" of the inner block, we read from src at some stride.
        print(f"\n  Read pattern for Layout A:")
        print(f"    To fill contiguous output, we read blocks of {a_inner} elements")
        print(f"    Stride between consecutive dim-2 blocks: {a_strides[2]} elements ({a_strides[2]*2} bytes)")
        print(f"    dim-2 has size {rows} (original rows), so {rows} jumps of stride {a_strides[2]}")

        print(f"\n  Read pattern for Layout B:")
        print(f"    To fill contiguous output, we read blocks of {b_inner} elements")
        print(f"    Stride between consecutive dim-2 blocks: {b_strides[2]} elements ({b_strides[2]*2} bytes)")
        print(f"    dim-2 has size {cols} (original cols), so {cols} jumps of stride {b_strides[2]}")


if __name__ == "__main__":
    main()
