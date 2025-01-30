import triton
import triton.language as tl
import torch


@triton.jit
def expf_kernel(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.exp(x)
    tl.store(y_ptr + offsets, y, mask=mask)


def expf(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    N = x.numel()
    y = torch.empty_like(x)
    BLOCK_SIZE = 1024
    expf_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
        x, y, N, BLOCK_SIZE=BLOCK_SIZE
    )
    return y


if __name__ == '__main__':
    N = 2194
    x = torch.randn(N, dtype=torch.float32, device='cuda')
    y = expf(x)
