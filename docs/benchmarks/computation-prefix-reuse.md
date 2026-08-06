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
| 4,096 | 256 | 25.931 s | 0.084 s | 1.862 s | 1.946 s | **13.33x** |
| 8,192 | 512 | 52.963 s | 0.099 s | 3.275 s | 3.375 s | **15.70x** |
| 16,384 | 512 | 105.805 s | 0.141 s | 3.465 s | 3.606 s | **29.34x** |

Every restored run produced the same first token as its cold control. The
checkpoint publication costs were 5.426, 7.228, and 11.651 seconds respectively
on the background path; ordinary generation does not wait for that work. In
the 16K run, verifying the complete 2.18 GB contiguous representation took
0.107 seconds and MLX opened it in 0.0009 seconds. The remaining 3.465 seconds
evaluated the 512 previously unseen tokens.

The restored path uses a 512-token prefill step while the cold path retains a
4,096-token step. A measured sweep found 512 to be the fastest exact shape for
the 514-token restored suffix; every tested shape produced the same first
token.

Publication phase accounting shows that verified store admission, rather than
MLX serialization, dominates the background cost. At 4K, serialization took
0.260 seconds, admission 5.078 seconds, and metadata publication 0.050 seconds.
The contiguous projection is adopted with an atomic rename after admission, so
projection adoption adds no further copy. The current adapter still writes the
verified chunk representation into the arena during admission; adopting the
contiguous file as a native store representation is the future one-write path.

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
checkpoint. A contiguous projection is a disposable accelerator: its BLAKE3
digest is bound into authenticated checkpoint metadata, every process reopen
verifies it, and a missing or changed projection is reconstructed from the
authoritative store. The current backend converges identical and overlapping
bytes, but tensor-aware block or delta representations are needed to minimize
incremental storage for long growing sessions.

The benchmark continuation is token-prefix-stable by construction. Some
stateless chat templates are not: they insert generation-only control tokens
or remove private reasoning when rendering the next request. Reusing raw
decode state across such a rewrite would change model semantics. Conserving a
completed response across turns therefore requires a session/context assembler
whose private token frontier is append-only; storage alone must not guess that
equivalence.

Checkpoint retention and physical reclamation remain operator policy. A
production fleet must route those through its computation-sharing domain,
resource accounting, and compaction scheduler rather than growing an unbounded
private cache.
