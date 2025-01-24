import triton
import triton.language as tl
import torch


FLT_MIN: tl.constexpr = torch.finfo(torch.float32).min


@triton.jit
def acc_max_and_norm(d_acc, max_acc, d, max_cur):
    #pdb.set_trace()
    m_max = max(max_acc, max_cur)
    d_next = d_acc * tl.exp(max_acc - m_max) + d * tl.exp(max_cur - m_max)
    return d_next, m_max


@triton.jit
def kernel_fwd(in_ptr, out_ptr, ds_ptr, max_ptr, ds_ptr_stride, lock_ptr, lock_stride, M, N, stride_xm, stride_xn, N_BLOCKS: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid_m = tl.program_id(1)
    n = tl.program_id(0)
    
    if n >= N or pid_m >= M:
        return
    row_offset = in_ptr + pid_m * stride_xm
    block_range = tl.arange(0, BLOCK_SIZE_N) * stride_xn
    column_offset = n * stride_xn * BLOCK_SIZE_N + block_range
    mask = column_offset < N * stride_xn
    block_idx = row_offset + column_offset
    block = tl.load(block_idx, mask=mask, other=FLT_MIN)
    d_next, max_chunk = tl.reduce((tl.where(mask, 1.0, 0), block), axis=0, combine_fn=acc_max_and_norm)
    tl.atomic_add(ds_ptr + pid_m * ds_ptr_stride + n * stride_xn, d_next)
    tl.atomic_max(max_ptr + pid_m * ds_ptr_stride + n * stride_xn, max_chunk)

    # spinlock
    this_lock_ptr = lock_ptr + lock_stride * pid_m
    tl.atomic_add(this_lock_ptr, 1)
    while tl.atomic_cas(this_lock_ptr, N_BLOCKS, N_BLOCKS) < N_BLOCKS:
        pass

    ds = tl.load(ds_ptr + pid_m * ds_ptr_stride + tl.arange(0,N_BLOCKS))
    ms = tl.load(max_ptr + pid_m * ds_ptr_stride + tl.arange(0,N_BLOCKS))
    d_tot, m_tot = tl.reduce((ds, ms), axis=0, combine_fn=acc_max_and_norm)
    column_offset = n * stride_xn * BLOCK_SIZE_N + block_range
    block_idx = row_offset + column_offset 
    mask = column_offset < N * stride_xn
    y = tl.load(block_idx, mask=mask, other=FLT_MIN)
    y = tl.exp(y - m_tot) / d_tot
    tl.store(out_ptr + pid_m * stride_xm + column_offset, y, mask=mask)


def softmax(X, BLOCK_SIZE=8192):
    M = X.size(0)
    N = X.size(1)
    if BLOCK_SIZE > N:
        BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = min(BLOCK_SIZE // 256, 32)
    num_warps = max(num_warps, 1)
    N_BLOCKS = triton.cdiv(N, BLOCK_SIZE)
    N_BLOCKS = triton.next_power_of_2(N_BLOCKS)
    ds_ptr = torch.zeros((M, N_BLOCKS), device="cuda:0", dtype=torch.float32)
    ms_ptr = torch.zeros((M, N_BLOCKS), device="cuda:0", dtype=torch.float32) + float("-inf")
    lock_ptr = torch.zeros((M,), dtype=torch.int32, device="cuda:0")
    Y = torch.zeros_like(X)
    grid = (N_BLOCKS, triton.next_power_of_2(M))
    kernel_fwd[grid](
        X,
        Y,
        ds_ptr,
        ms_ptr,
        ds_ptr.stride(0),
        lock_ptr,
        lock_ptr.stride(0),
        M,
        N,
        X.stride(0),
        X.stride(1),
        N_BLOCKS,
        BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=2,
    )
    return Y


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1823, 34231, device="cuda")
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=-1)
    print(y_triton.sum(dim=-1))
    assert torch.allclose(y_triton, y_torch)