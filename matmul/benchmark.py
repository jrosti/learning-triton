import torch
import time

M = 4096
K = 16384
N = 1024


def time_it(matmul_fn, op, device="cuda:0"):
    A = torch.rand(M, K, dtype=torch.float32, device=device)
    B = torch.rand(K, N, dtype=torch.float32, device=device)
    torch_result = A@B
    _ = matmul_fn(A, B)
    torch.cuda.synchronize()

    start = time.time()
    C = matmul_fn(A, B)
    torch.cuda.synchronize()
    print(f"{op}: {1000*(time.time()-start)} ms")
    torch.testing.assert_close(C, torch_result, rtol=1e-4, atol=1e-3)
