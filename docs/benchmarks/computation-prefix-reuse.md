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
Workload: 4,096 retained prefix tokens plus a 256-token continuation

| Path | Time to first token |
|---|---:|
| Cold 4,352-token input | 26.252 s |
| Store restore | 2.994 s |
| Compute 258-token suffix | 2.094 s |
| Restored total | 5.088 s |

The end-to-end restart speedup was **5.16x**. The first token matched the cold
control. Publishing the checkpoint took 4.936 seconds on the background path;
ordinary generation does not wait for that work.

A 128-prefix/32-continuation control measured 1.38x after storage reopen. This
small case is useful because it exposes the fixed retrieval cost: reuse becomes
more valuable as avoided prefill grows.

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
