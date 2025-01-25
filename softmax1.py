import triton
import torch
import triton.language as tl


@triton.jit
def kernel_fwd(x_ptr, y_ptr, M, N, stride_xm, stride_xn, num_blocks: tl.constexpr, block_size: tl.constexpr):
    m = tl.program_id(0)
    if m >= M:
        return
    row_offset = x_ptr + m * stride_xm
    block_range = tl.arange(0, block_size)
    denom_acc = 0.0
    max_acc = -float("inf")
    for n in tl.static_range(0, num_blocks):
        column_offset = n * stride_xn * block_size + block_range
        mask = column_offset < N * stride_xn
        row_block = row_offset + column_offset
        x = tl.load(row_block, mask=mask, other=-float("inf"))
        max_next = tl.max(x)
        new_max_acc = tl.maximum(max_next, max_acc)
        denom_next = tl.sum(tl.exp(x - new_max_acc))
        denom_acc = denom_acc * tl.exp(max_acc - new_max_acc) + denom_next
        max_acc = new_max_acc
    for n_ in tl.static_range(0, num_blocks):
        n = num_blocks - n_ - 1
        if n_ != 0:
            column_offset = n * stride_xn * block_size + block_range
            row_block = row_offset + column_offset 
            mask = column_offset < N * stride_xn
            x = tl.load(row_block, mask=mask, other=float("-inf"))
        x = x - max_acc
        x = tl.exp(x) / denom_acc
        tl.store(y_ptr + m * stride_xm + column_offset, x, mask=mask)


def softmax(x, block_size=8192, num_warps=32):
    M = x.size(0)
    N = x.size(1)
    if block_size > N:
        block_size = triton.next_power_of_2(N)
    num_blocks = triton.cdiv(N, block_size)
    y = torch.empty_like(x)
    grid = (triton.next_power_of_2(M),)
    k = kernel_fwd[grid](
        x,
        y,
        M,
        N,
        x.stride(0),
        x.stride(1),
        num_blocks,
        block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    # ptx_file = __file__.replace(".py", ".ptx")
    # with open(ptx_file, "w") as f:
    #     f.write(k.asm["ptx"])
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1823, 32768, device="cuda")
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=-1)
    print(y_triton.sum(dim=-1))
    assert torch.allclose(y_triton, y_torch)