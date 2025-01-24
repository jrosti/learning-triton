from pathlib import Path
from torch.utils.cpp_extension import load_inline
import benchmark
import os


def compile_extension():
    cuda_source = Path("matmul.cu").read_text()
    cpp_source = """
torch::Tensor matmul_naive(torch::Tensor a, torch::Tensor b);
"""
    # prints warning if not explicitly specified
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

    # Load the CUDA kernel as a PyTorch extension
    matmul_extension = load_inline(
        name="matmul",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["matmul_naive"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )
    return matmul_extension


if __name__ == '__main__':
    matmul_ext = compile_extension()
    benchmark.time_it(matmul_ext.matmul_naive, "cuda_matmul")


