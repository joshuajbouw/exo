# Durable MLX prefix reuse

This benchmark measures Exo restoring a completed MLX prefix computation from
the embedded computation store after the in-memory prefix cache has been
destroyed.

It runs directly through `StoreKVPrefixPersistence` and the optional PyO3
adapter. The current adapter uses Astrid's verified content engine internally,
but that provider is outside Exo's persistence contract. There is no daemon,
capsule, MCP, CLI, or network hop in the measured path.

## Result

Host: Apple M2 Ultra with 192 GB unified memory.
Model: `mlx-community/gemma-4-31b-it-4bit`.
Workloads: 4,096 to 16,384 retained prefix tokens plus a continuation.

| Prefix | New input | Cold TTFT | Store restore | Suffix compute | Restored TTFT | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 256 | 26.212 s | 2.994 s | 2.094 s | 5.088 s | **5.16x** |
| 8,192 | 512 | 53.051 s | 3.174 s | 3.416 s | 6.590 s | **8.05x** |
| 16,384 | 512 | 105.001 s | 4.988 s | 3.591 s | 8.578 s | **12.24x** |

Every restored run produced the same first token as its cold control. The
checkpoint publication costs were 5.023, 7.199, and 9.579 seconds respectively
on the background path; ordinary generation does not wait for that work.

A 128-prefix/32-continuation control varied around break-even because verified
store startup and retrieval dominate the tiny amount of avoided computation.
The matrix shows the expected scaling: reuse becomes more valuable as avoided
prefill grows while restore cost rises much more slowly.

Run the benchmark with:

```bash
uv pip install ./integrations/computation_store
uv run bench/computation_prefix_reuse.py /path/to/mlx-model \
  --prefix-tokens 4096 \
  --append-tokens 256 \
  --runtime-profile exact-model-closure-id
```

## Claim boundary

The result demonstrates durable partial computation reuse. It does not make
novel autoregressive decoding faster, and it does not yet memoize complete
generation results. The current physical representation is an MLX safetensors
checkpoint. The current backend converges identical and overlapping bytes, but
tensor-aware block or delta representations are needed to minimize incremental
storage for long growing sessions.

Checkpoint retention and physical reclamation remain operator policy. A
production fleet must route those through its computation-sharing domain,
resource accounting, and compaction scheduler rather than growing an unbounded
private cache.
