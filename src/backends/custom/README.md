# Custom Continuous Batching Engine

## 1. What This Is and Why It Exists

This directory contains a custom-built Continuous Batching inference server, built from first principles using standard PyTorch and HuggingFace Transfomers mechanics. 

The primary goal of this component is educational and demonstrative. Real-world systems like vLLM and TGI are highly optimized C++ and CUDA codebases. By implementing continuous batching purely in Python, we demonstrate a deep systemic understanding of the underlying memory bottlenecks (KV-cache fragmentation) and scheduling paradigms (iteration-level scheduling) that make modern LLM serving fast.

## 2. Request Lifecycle Diagram

```mermaid
flowchart TD
    A[Client Request] --> B[FastAPI Gateway]
    B --> C[Tokenization]
    C --> D[Add to Waiting Queue]
    D --> E[Scheduler Loop]
    E --> F{KV Cache Space?}
    F -- Yes --> G[Allocate Blocks & Move to Running]
    F -- No --> J[Wait Next Iteration]
    G --> H[Prefill Phase <br/> (Left-Padded Batching)]
    H --> I[Decode Phase <br/> (Left-Padded Cache Extension)]
    I --> K{EOS or Max Tokens?}
    K -- No --> E
    K -- Yes --> L[Free KV Blocks]
    L --> M[Resolve Future & Return]
```

## 3. What We Implement

1. **Continuous Batching (Orca-style)**: Instead of waiting for all sequences in a batch to finish (static batching), the `scheduler.py` evaluates the `running` and `waiting` queues at every token generation step. As soon as a request finishes, its response is resolved, and a new request is admitted to the batch if memory permits.
2. **KV-Cache Pooling**: `kv_cache.py` pre-allocates a fixed block of GPU memory for KV-tensors to prevent random allocator fragmentation and OOM errors during long generations.
3. **Prefill/Decode Separation**: The generation process logically splits the heavy, compute-bound prefill step from the memory-bound token-by-token decode step.

## 4. What We Deliberately Skip

To keep the logic comprehensible and maintainable in pure Python, we omit several production optimizations:
- **PagedAttention**: We don't implement custom CUDA kernels to map virtual contiguous cache block pointers to dispersed physical blocks. As a result, our engine enforces left-padding under the hood.
- **Chunked Prefill**: We process the entire prompt in one prefill pass rather than splitting it.
- **Speculative Decoding**: No draft model validation tokens.
- **Tensor Parallelism**: Runs on a single GPU constraint.
- **CUDA Graphs**: No capture of static operations, which introduces Python overhead loop-by-loop.
- **Prefix Caching**: Repeating the same system prompt recalculates its KV cache every time.

## 5. Performance Gap Analysis vs vLLM

This backend is intentionally slower than vLLM. Here is a factor-by-factor breakdown explaining why:

| Factor | vLLM Implementation | Our Implementation | Impact on Performance |
|--------|---------------------|--------------------|-----------------------|
| **Attention Kernel** | `PagedAttention` reading dispersed memory | Standard PyTorch FlashAttention via left-padded contiguous memory | High (We lose significant memory capacity to padding fragmentation) |
| **Memory Allocation**| Virtual-to-physical block mapping via Page Tables | Preallocated standard tensor blocks | High (Requires moving tensors around vs pointer referencing) |
| **Scheduler** | Optimized C++ execution | Python async loops | Medium (GIL and interpreter loop overhead) |
| **GPU Operations** | CUDA Graphs for decode | Re-launched PyTorch ops per-token | Medium (High latency per iteration) |
| **Prefill Stage** | Mixes prefixes with decode tokens | Sequestered strict phase | Minor (Temporary pipeline stalls) |

## 6. References
- *Orca: A Distributed Serving System for Transformer-Based Generative Models* (Yu et al., 2022) - Introduced iteration-level continuous batching.
- *Efficient Memory Management for Large Language Model Serving with PagedAttention* (Kwon et al., 2023) - The mechanics driving vLLM.
