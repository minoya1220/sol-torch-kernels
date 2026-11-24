from collections.abc import Callable

import cutlass.cute as cute


def warp_reduction_partial(value: cute.typing.Numeric, op: Callable, width: int) -> cute.typing.Numeric:
    """Reduce `value` across groups of `width` adjacent lanes with butterfly shuffles."""
    assert 1 <= width <= 32, f"width must be in [1,32], got {width}"
    assert (width & (width - 1)) == 0, f"width must be a power of 2, got {width}"
    result = value
    for i in range(width.bit_length() - 1):
        result = op(result, cute.arch.shuffle_sync_bfly(result, 1 << i))
    return result


def pointer_alignment(*tensors) -> int:
    """Largest power of two, up to 16 bytes, that divides the address of every tensor."""
    align = 16
    for t in tensors:
        while t.data_ptr() % align:
            align //= 2
    return align
