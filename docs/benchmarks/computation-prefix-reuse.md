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

### Completed conversational turn

The representative two-turn test retains the cache *after* the assistant
finishes, destroys the in-memory cache, reopens the verified frontier, and
appends the next user turn. For a 4,093-token Gemma prompt, the model answered
`OK` in two tokens; both response tokens became part of the 4,095-token private
frontier. The next user turn added 19 tokens.

| Path | TTFT | Versus cold |
|---|---:|---:|
| Full cold replay | 24.005 s | 1x |
| Standard synchronous resumed generation | 2.077 s | 11.6x |
| Fresh-process Exo batch continuation | 0.545 s | **44.1x** |
| Warm durable one-call median | 0.336 s | **71.5x** |
| Live Exo batch continuation | **0.296 s** | **81.2x** |
| Direct in-memory compute control | 0.275 s | 87.3x |

Every path produced the same first token as the cold replay. A separate
three-token run produced the same complete token sequence through the optimized
batch and standard synchronous paths. The live batch result is within 21 ms of
the direct model control; its scheduler returned the already-selected first
token in 0.39 ms. At that point the live serving path has reached the model's
short-continuation compute floor rather than an Exo cache-management floor.

The fresh-process result includes authenticated continuation lookup, a full
BLAKE3 check of the contiguous projection, MLX cache reconstruction, page-in of
the checkpoint tensors, the 19-token append, and first-token selection. Warm
durable samples avoid the initial tensor page-in but still reconstruct a new
MLX cache view. The live path instead forks the immutable response frontier
copy-on-write: existing KV tensors remain shared until the multi-token append
allocates its new state.

Exo's batch path now performs the short append and first-token selection in one
model call. It returns that primed token without speculatively computing token
two; the completed cache may trail the returned frontier by one identified
token, which the next continuation folds into its append. This removes work at
a terminal boundary without changing later generation. Splitting the
vocabulary head from the transformer was also measured and was slower, so that
route was rejected.

### Verified remembered-output speculation

A completed greedy response is also retained as a proposal keyed by the exact
prompt-token identity and runtime profile. On a later match, Exo forks the KV
frontier copy-on-write, evaluates the remembered tokens as one target-model
sequence, and returns them only if Gemma's logits verify the complete block.
The draft is never an answer cache or authority. Corrupt metadata, unsupported
cache layouts, an invalid token id, or a token mismatch becomes an ordinary
generation fallback.

The following matrix compares the same warm prefix path with and without the
remembered output. Times include submit, verification or generation, token
delivery through Exo's batch loop, and completion handling.

| Output | Verification block | Ordinary | Remembered | Speedup | Accepted |
|---:|---:|---:|---:|---:|---:|
| 32 tokens | 8 | 22.4 tok/s | 38.4 tok/s | **1.72x** | 31/31 |
| 32 tokens | 16 | 22.2 tok/s | 48.1 tok/s | **2.17x** | 31/31 |
| 32 tokens | 32 | 23.0 tok/s | 72.3 tok/s | **3.14x** | 31/31 |
| 64 tokens | 64 | 23.7 tok/s | 107.9 tok/s | **4.56x** | 63/63 |
| 128 tokens | 128 | 26.2 tok/s | 137.2 tok/s | **5.23x** | 127/127 |
| 256 tokens | 128 | 26.4 tok/s | 147.2 tok/s | **5.58x** | 255/255 |

Every optimized run emitted the exact same token sequence as ordinary greedy
generation. A 256-token verification block crossed a Gemma/MLX numerical
divergence boundary in this setup and correctly fell back, so production uses
128-token blocks rather than assuming that larger is always safe or faster.

The durable control destroyed the writer cache, flushed the KV frontier and
draft, released the embedded store, and reopened a new persistence instance.
For 64 output tokens it accepted 63/63 proposals and produced the identical
sequence at 122.0 tok/s versus 25.7 tok/s ordinary. The small apparent lead
over the live 111.9 tok/s sample is run-to-run noise and cache warmth, not a
claim that process restart improves inference.

The draft payload is canonical little-endian token ids: four bytes per token,
so a 256-token proposal is 1 KiB plus metadata. It does not duplicate model
weights or the KV checkpoint; those remain governed by the existing prefix
cache and computation-store policies.

Run the output benchmark with:

```bash
uv run bench/speculative_continuation.py /path/to/mlx-model \
  --tokens 128 \
  --block-size 128 \
  --durable
```

This first implementation indexes one most-recent branch per exact prompt. It
does not yet retrieve approximate branches, learn a branch policy, or apply
rejection-correct speculative sampling at nonzero temperature. Those are
separate policy and sampling problems; target-model verification remains the
authority in every case. Draft memory is bounded by the retained KV frontier
count, durable publications coalesce by prompt identity, and the configured
computation-store path must be scoped to the intended sharing domain.

Run the benchmark with:

```bash
uv pip install ./integrations/computation_store
uv run bench/computation_prefix_reuse.py /path/to/mlx-model \
  --prefix-tokens 4096 \
  --append-tokens 256 \
  --runtime-profile exact-model-closure-id
```

Run the conversational benchmark with:

```bash
uv run bench/conversation_frontier_reuse.py /path/to/mlx-model \
  --prefix-tokens 4096 \
  --samples 5
```

## Claim boundary

The result demonstrates durable partial computation reuse, including the
completed response frontier. It does not make novel autoregressive decoding
free: the next user tokens and the next answer still require model evaluation.
The current physical representation is an MLX safetensors checkpoint. A
contiguous projection is a disposable accelerator: its BLAKE3 digest is bound
into authenticated checkpoint metadata, every process reopen verifies it, and
a missing or changed projection is reconstructed from the authoritative store.
The current backend converges identical and overlapping bytes, but tensor-aware
block or delta representations are needed to minimize incremental storage for
long growing sessions.

The benchmark continuation is token-prefix-stable by construction. Some
stateless chat templates are not: they insert generation-only control tokens
or remove private reasoning when rendering the next request. Reusing raw
decode state across such a rewrite would change model semantics. The Responses
API therefore carries `previous_response_id`, which resolves the exact private
frontier rather than reconstructing it from visible messages. The current fast
path deliberately accepts only Gemma's known plain-user append grammar; other
models or request shapes fail closed rather than guessing equivalence. A
general session/context assembler still needs its own append contract.

Checkpoint retention and physical reclamation remain operator policy. A
production fleet must route those through its computation-sharing domain,
resource accounting, and compaction scheduler rather than growing an unbounded
private cache.
