import triton
import triton.language as tl
import torch
import benchmark


@triton.jit
def get_1d_offset(size, n_prev_chunks):
    """
    Calculate 1D memory offsets for a given chunk size and position.

    Args:
        size: Size of the current chunk
        n_prev_chunks: Number of previous chunks (used for position)

    Returns:
        Array of offsets for the current chunk
    """
    return n_prev_chunks * size + tl.arange(0, size)

@triton.jit
def get_2d_offset(offs_0, offs_1, stride_0, stride_1=1):
    """
    Calculate 2D memory offsets for matrix operations.

    Args:
        offs_0, offs_1: Offsets in first and second dimensions
        stride_0, stride_1: Stride values for memory layout

    Returns:
        2D array of memory offsets
    """
    return tl.expand_dims(offs_0, 1)*stride_0 + tl.expand_dims(offs_1, 0)*stride_1

@triton.jit
def get_1d_mask(offs, max):
    """
    Create a mask for boundary checking in 1D.

    Args:
        offs: Current offsets
        max: Maximum valid offset

    Returns:
        Boolean mask indicating valid positions
    """
    return offs < max

@triton.jit
def get_2d_mask(offs_0, offs_1, max_0, max_1):
    """
    Create a mask for boundary checking in 2D.

    Args:
        offs_0, offs_1: Current offsets in both dimensions
        max_0, max_1: Maximum valid offsets

    Returns:
        Boolean mask indicating valid positions in 2D
    """
    return (tl.expand_dims(offs_0, 1) < max_0) & (tl.expand_dims(offs_1, 0) < max_1)


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,  # Pointers to input/output matrices
    m, n, k,              # Matrix dimensions: A(m×k), B(k×n), C(m×n)
    stride_am, stride_ak, # Memory strides for matrix A
    stride_bk, stride_bn, # Memory strides for matrix B
    stride_cm, stride_cn, # Memory strides for output matrix C
    block_size: tl.constexpr,     # Block size for M, N, K dimensions
):
    """
    Compute matrix multiplication C = A × B using block-wise operations.

    This kernel implements a basic matrix multiplication by:
    1. Breaking the computation into blocks
    2. Loading blocks into shared memory
    3. Computing partial results
    4. Storing the results back to global memory

    Args:
        a_ptr, b_ptr: Input matrix pointers
        c_ptr: Output matrix pointer
        m, n, k: Matrix dimensions
        stride_*: Memory strides for each matrix
        block_size:
    """
    # Get program ID for the current thread block
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)

    # Calculate offsets for the current block
    rm = get_1d_offset(size=block_size, n_prev_chunks=pid_m)  # Offset in M dimension
    rn = get_1d_offset(size=block_size, n_prev_chunks=pid_n)  # Offset in N dimension
    rk = get_1d_offset(size=block_size, n_prev_chunks=0)      # Initial offset in K dimension

    # Calculate memory offsets for input matrices
    offs_a = a_ptr + get_2d_offset(rm, rk, stride_am, stride_ak)
    offs_b = b_ptr + get_2d_offset(rk, rn, stride_bk, stride_bn)

    # Initialize accumulator for partial results
    # Note: allow_tf32 must be set to False for older GPUs
    acc = tl.zeros((block_size, block_size), dtype=tl.float32)

    # Main computation loop - iterate over K dimension
    for _ in range(0, k, block_size):
        # Load blocks from input matrices
        a = tl.load(offs_a)  # Load block from matrix A
        b = tl.load(offs_b)  # Load block from matrix B

        # Compute partial matrix multiplication for current block
        acc += tl.dot(a, b, allow_tf32=False)

        # Update offsets for next iteration
        offs_a += block_size * stride_ak
        offs_b += block_size * stride_bk

    # Calculate output memory location and mask for boundary conditions
    c = c_ptr + get_2d_offset(rm, rn, stride_cm, stride_cn)
    mask = get_2d_mask(rm, rn, m, n)

    # Store the result
    tl.store(c, acc, mask=mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are valid
    assert a.device.type == 'cuda' and b.device.type == 'cuda', "Tensors must be on CUDA"
    assert a.dtype == torch.float32 and b.dtype == torch.float32, "Tensors must be float32"
    assert a.size(1) == b.size(0), "Incompatible dimensions for matrix multiplication"

    # Define matrix dimensions
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Define block size
    BLOCK_SIZE = 64
    # Launch kernel
    grid = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        block_size=BLOCK_SIZE,
        num_stages=2,
        num_warps=4,
    )
    return c


if __name__ == '__main__':
    benchmark.time_it(matmul, "triton")