# Learning Triton: softmax for wide rows

A January 2025 learning project on an NVIDIA RTX 3090 (Ampere). I wanted to understand how Triton code maps to GPU execution, and whether I could make row-wise softmax faster than `torch.softmax` for large rows.

The path:

1. **Start with fusion.** The [Triton softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) keeps a row on-chip to avoid intermediate memory traffic. Increasing the row width runs into on-chip resource limits.
2. **Process rows in chunks.** [softmax1.py](softmax1.py) assigns one program per row, with 8,192-element chunks by default. It combines each chunk's maximum and exponential sum using an [online normalizer](https://arxiv.org/abs/1805.02867), rescaling the running sum when the maximum changes. A second pass writes the normalized output. [softmax2.py](softmax2.py) records an earlier experiment with multiple programs per row, atomics and a spinlock.
3. **Inspect what the compiler actually generated.** Profiling alone left me guessing. Reading [Triton IR, LLVM IR and PTX](softmax1_compiled/) connected the source to vectorized loads, reductions and loop unrolling. The key finding was register pressure: [`tl.static_range`](https://triton-lang.org/main/python-api/generated/triton.language.static_range.html) requests aggressive unrolling, which can keep more values live across chunks. Registers are finite per SM; higher usage can limit resident warps and latency hiding, or cause spills. [CUDA register-pressure guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#register-pressure).
4. **Make the loop explicit, then measure.** In [softmax1-1.py](softmax1-1.py), I moved the final chunk's output outside the loop and used `range(0, num_blocks - 1)` for the remaining chunks. The first pass still uses `tl.static_range`. The saved [PTX](softmax1-1.ptx) confirms a loop back-edge (`@%p80 bra $L__BB0_3`); Nsight Compute reports fewer allocated registers. PTX register declarations themselves are virtual registers, so I used the profiler for the hardware count.

The saved profiles also show why lower register usage alone does not establish a speedup:

| Variant | Registers/thread | Theoretical occupancy | Kernel time |
| --- | ---: | ---: | ---: |
| [softmax1](softmax1.ncu) | 44 | 66.67% | 776.99 µs |
| [softmax1-1](softmax1-1.ncu) | 38 | 66.67% | 783.84 µs |

The overall win is the chunked kernel versus PyTorch. At 65,536 columns, the saved plot shows approximately **580 versus 430 GB/s (~1.35×)**. These are historical FP32 measurements with 16,384 rows. The benchmark reports effective bandwidth as `2 * input_bytes / time`; the chunked algorithm rereads input, so this is not measured DRAM traffic.

![Saved softmax1 versus PyTorch benchmark on RTX 3090](softmax1_bm.png)

To run the forward correctness checks and benchmark with CUDA-enabled PyTorch, Triton, pandas and Matplotlib installed:

```bash
python softmax1.py
python softmax1-1.py
python softmax_benchmark.py
```

These are learning experiments. The results above concern the forward pass; the repo also contains exploratory backward-pass and matrix-multiplication code.
