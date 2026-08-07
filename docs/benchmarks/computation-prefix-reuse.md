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

### Growing conversation storage

The growing-session harness closes the computation store after every turn and
opens a new persistence instance for the next one. It therefore measures a
durable conversation rather than an in-memory cache accidentally surviving the
turn boundary. A three-turn smoke run at a 1,021-token initial prompt produced:

| Turn | Prompt | Reused | TTFT | New frontier | Projection growth | Authoritative growth |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,021 | 0 | 6.85 s | 1,023 tokens | 1.84 GB | 1.12 GB |
| 1 | 1,055 | 1,023 | 0.48 s | 34 tokens | 925 MB | 242 MB |
| 2 | 1,089 | 1,056 | 0.75 s | 35 tokens | 928 MB | 241 MB |

Continuation reused 97% of each later prompt and cut TTFT by roughly an order
of magnitude. The physical-growth result is unacceptable as a production
format: each 34–35-token turn added about 1.17 GB, or 32–34 MB per new token.
The generic content store removed roughly 74% of each new full checkpoint, but
the disposable safetensors projection still duplicated about 925 MB and the
rotating tensor layout manufactured about 241 MB of authoritative novelty.
This establishes two separate requirements: projections need an
operator-governed eviction budget, and checkpoints need a tensor-aware
block/delta representation rather than repeated whole-cache files. Retention
alone cannot solve the authoritative amplification.

The tensor-overlap probe then compared the two completed frontier files using
each cache entry's logical token interval rather than its physical ring-buffer
position. All 60 layer overlaps were byte-exact. Of the 925,368,320 successor
tensor bytes, 895,631,360 bytes (96.79%) already existed in the prior frontier;
only 29,736,960 bytes (3.21%) were new tensor payload. The model had 50 rotating
and 10 append-only KV layers. A range-addressed representation can therefore
reduce this turn's authoritative payload from 242 MB toward 30 MB without
approximation, compression, or numerical reconstruction.

Run the growth harness with:

```bash
uv run bench/conversation_growth.py /path/to/mlx-model \
  --initial-tokens 4096 --turns 4 --output-tokens 8

uv run bench/checkpoint_delta_probe.py \
  /path/to/base.safetensors /path/to/successor.safetensors
```

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
sequence, and returns only the longest prefix Gemma's logits verify. If a
proposal diverges in the middle of a block, Exo trims the unverified suffix
from the speculative KV fork before exposing the verified prefix, then resumes
ordinary generation or tries the next compatible ranked branch.
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

Every exact-remembered-branch run in this matrix emitted the same token sequence
as ordinary greedy generation. A 256-token verification block crossed a
Gemma/MLX numerical divergence boundary in this setup and correctly fell back,
so the exact-branch path uses 128-token blocks rather than assuming that larger
is always safe or faster. The dynamic-selector results below show why this is a
measured profile property, not a universal guarantee.

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
also exposes a provider-neutral selector contract for richer branch sources.
A selector receives the device-resident prompt tokens and the exact remembered
branch, and may return ranked content-addressed candidates. This is the seam
for a GPU relation engine: it proposes and ranks; Gemma still verifies. The
selector receives structured feedback containing accepted length, first
mismatch, selection cost, verification cost, and verification-pass count. Its
feedback method must enqueue rather than perform training on the token path.

Candidate IDs must be unique within a selection. Missing identities, duplicate
identities, invalid token IDs, selector failures, feedback failures, and unsafe
cache rollback all fail to ordinary generation. Exo also includes an opt-in GPU
Tensor Logic selector at that seam. Corpus rows are typed
context-to-continuation facts. Their context entities remain in an MLX matrix,
and T=0 selection is one batched contraction followed by a bounded host result.
The live verified token stream is a second, ephemeral relation source: a GPU
longest-suffix query recovers repeated structure without publishing it as
durable fleet memory. Corpus identity binds the privacy domain, runtime profile,
selector profile, source identities, contexts, and candidate tokens.

Dynamic re-selection after ordinary progress is deliberately experimental and
disabled by default. Its first real-trace gate exposed a concrete rollback bug:
once Gemma's sliding window was full, MLX's `RotatingKVCache.trim()` advanced
its cursors but left rejected tail tensors visible. The short synthetic fixture
never crossed the window and therefore missed it. Exo now removes the physical
tail and has a full-window regression. Re-running the same held-out Astrid tool
window at a 16-token block produced the same 64 tokens as serial greedy,
accepted 24 speculative tokens, and ran at 18.87 tok/s versus 21.27 tok/s
(0.89x). The output gate recovered; this selector/corpus pair is not yet an
optimization.

An independent serial-versus-batched diagnostic also established a separate
MLX constraint. Fully copied and copy-on-write caches produce the same batched
result, ruling out Exo's fork, but MLX's multi-token causal kernel is not
bit-identical to repeated one-token execution. On the 1,423-token held-out
window, accepted full blocks retained the same argmax while maximum logit
differences grew from 2.57 at four tokens to 14.20 at 32. Before the rollback
fix, the two affected partial prefixes flipped the next argmax; after it, both
matched serial again. Dynamic refill remains disabled until a broader
equivalence matrix and a useful hit-rate/performance gate pass. Memo presence
must not silently choose weaker generation semantics.

Reproduce that gate against an extracted trace with:

```bash
uv run bench/speculative_continuation.py /path/to/mlx-model \
  --tokens 128 --block-size 32 --tensor-logic \
  --trace-file /path/to/held-out-session.txt --trace-cut 507334
```

Diagnose the MLX kernel and partial-cache boundary directly with:

```bash
uv run bench/mlx_kv_equivalence.py /path/to/mlx-model \
  --trace-file /path/to/held-out-session.txt --trace-cut 507334
```

To exercise partial-prefix salvage and ranked fallback against the real model:

```bash
uv run bench/speculative_continuation.py /path/to/mlx-model \
  --tokens 128 --block-size 128 --mismatch-at 64

uv run bench/speculative_continuation.py /path/to/mlx-model \
  --tokens 128 --block-size 128 --mismatch-at 64 --ranked-fallback
```

On the same Gemma 4 31B / M2 Ultra rig, corrupting token 64 of a 128-token
remembered branch still produced byte-identical output. Prefix salvage reused
64 correct proposal tokens and reached 37.4 tok/s versus 25.6 ordinary
(**1.46x**). Putting the exact branch second let Exo salvage the first 64,
switch at the proven mismatch, and verify the remainder from the compatible
branch at 88.6 tok/s (**3.47x**). The benchmark selector itself took about
40 microseconds; the two target-model verification passes took 823 ms and
500 ms. These are deliberately adversarial branch-selection measurements,
not a claim about the not-yet-integrated Tensor Logic selector's hit rate or
GPU contention.

The exact-branch control on the same code reached 135.7 tok/s versus 25.6
ordinary (**5.30x**), accepted all 127 speculative tokens, and remained
byte-identical. The selector call took 32 microseconds, confirming that the new
ranked-candidate seam did not regress the earlier exact-reuse result.

Approximate branch retrieval, a learned selection policy, and
rejection-correct speculative sampling at nonzero temperature remain separate
policy and sampling problems; target-model verification remains the authority
in every case. Draft memory is bounded by the retained KV frontier count,
durable publications coalesce by prompt identity, and the configured
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
