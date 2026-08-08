# Durable model memory contract

Status: normative for the memory experiments in this repository. If an
experiment, benchmark name, comment, or implementation conflicts with this
document, this document wins until it is deliberately amended with evidence.

Amendments are prospective. A contract change must land as a separate reviewed
change before dependent implementation or measurement begins. An experiment
may confirm or refute this contract; it may not relax the contract to make
itself pass.

## The intended behavior

A conversation may contain something worth remembering. Months and billions
of intervening tokens later, a fresh model process can use that memory without
replaying the original conversation into its prompt and without retaining the
original conversational K/V cache.

The model does not search the lifetime corpus. The operating system selects a
small authorized working set from durable memory and mounts model-readable
representations for the invocation. To the conversational caller this should
feel like remembering, but selection, authority, storage, and model execution
remain separate mechanisms.

## The four distinct states

### Authoritative memory

Events, messages, artifacts, identities, time, relations, and receipts are
content-addressed durable objects. They are the source of truth. They may be
exported, audited, corrected by later events, revoked according to policy, and
used to rebuild every derived representation.

Neither model weights, latent pages, summaries, indexes, nor K/V checkpoints
may hold the only copy of authoritative memory.

### Grounded selection

Mimir derives which memories are relevant from typed relations, temporal
facts, provenance, the invocation goal, the principal's granted namespace, and
the current epoch. Huginn assembles the selected working set. Selection is an
external deterministic operation with a canonical proof; it is not keyword
matching, prompt guessing, or a learned router inside Gemma.

The capability system authorizes access. Tensor Logic may propose and explain
the selected set, but does not grant access. A selection proof binds the exact
page identities it permits. A proof for one page cannot activate another.

### Latent memory pages

A latent page is a disposable, model- and runtime-specific `Derived`
representation of authoritative evidence. Gemma reads it through the separate
global-memory attention channel. It consumes no conversational positions and
is not concatenated to ordinary K/V.

The originating text, token ids, and conversational K/V are absent from the
strict latent-recall test. A page is useful only when a grounded selector has
chosen it. Asking Gemma to select among an undifferentiated bank is not the
architecture and is not required for memory correctness.

Pages inherit the privacy domain, retention, revocation, and accounting policy
of their evidence closure. Revocation removes the page from the next selection
proof and restores the exact unpaged model path.

### Hot continuation K/V

Native K/V is an exact accelerator for the current conversational frontier. It
supports fast turn continuation, process restart, branch sharing, and
conservation of already-computed prefill. It is large, positional,
model-specific, and bounded by the model's context semantics.

Native K/V is not long-term memory. Persisting or restoring it does not satisfy
the durable-memory claim. It may always be evicted and reconstructed from the
authoritative conversation while that conversation remains retained.

## Lifecycle

1. An ordinary model invocation consumes a user turn.
2. Astrid commits the turn and its receipts as authoritative objects.
3. A frozen writer may derive a latent page from the model's internal response
   to that turn. The page is identity-bound to its model, runtime, writer, and
   evidence closure.
4. Much later, a new invocation produces a grounded selection proof from the
   current query, goal, time, relations, and capabilities.
5. Huginn resolves the selected page identities and mounts only that bounded
   set through the memory sidecar.
6. Gemma answers using its ordinary weights, current conversational K/V, and
   the mounted latent pages. The historical source text is not replayed.
7. The pages are unmounted after the invocation. Hot K/V may be retained as a
   computation cache; authoritative memory remains independently durable.

## Proven and unproven claims

The current evidence proves:

- Gemma can read a removable global-attention memory channel.
- One frozen shared writer can compile unseen ordinary conversations into
  serialized pages that survive process death.
- With the correct externally selected page mounted, a fresh Gemma process
  recalled 32/32 held-out memories without source text, source token ids, or
  ordinary K/V.
- The grounded selector then closed the integration boundary: with 100,000
  unrelated catalog entries present, page-bound proofs selected the same eight
  memories and fresh Gemma recalled 16/16 withheld natural-language queries.
  Wrong concepts, revoked grants, domain mismatch, expired epochs, and page
  substitution all failed safely.
- The same eight pages and their canonical catalog survived writer-process
  death in Astrid's durable object store. A fresh reader with no source-artifact
  path reconstructed and identity-checked the closure before reproducing 16/16
  recall.
- Revocation restores the exact unpaged path.
- Separately, native conversational K/V can survive process death and produce
  exact continuation. That is a hot-cache result only.

The current evidence does not prove:

- deriving registered concept identities from arbitrary natural language;
- unseen relation and vocabulary transfer at arbitrary scale;
- compression or capacity beyond the measured pages;
- safe simultaneous activation of arbitrary pages; or
- that model weights or latent pages replace authoritative evidence.

The failed simultaneous-bank and reader-writer experiments do not invalidate
latent memory. They show that Gemma should not be trained or expected to become
the authority-bearing page selector.

## Implementation invariants

- A strict natural-memory invocation gives the model its ordinary current
  conversation only. Memory-specific operation descriptions, relation names,
  candidate identities, catalogs, or routing instructions may not be added to
  its prompt.
- An experiment that supplies such an interface is a tool-routing control. It
  cannot satisfy or weaken the grounded-memory claim, regardless of accuracy.
- Production-style activation accepts a typed selection proof bound to the
  exact page id; an arbitrary proof string is insufficient.
- The memory sidecar never performs retrieval or decides relevance.
- Page identity binds its bytes, model, runtime profile, and canonical layer
  order.
- No-page and post-revocation execution remain byte-identical to the base path.
- The native-K/V benchmark identifies itself as a continuation-cache baseline
  and makes no long-term-memory claim.
- Benchmarks that replay source tokens, restore source K/V, or allow the model
  to see the answer are not latent-memory evidence.

## Code map

- `LATENT_MEMORY_RESULT.md`: formal statement of the demonstrated mechanism,
  protocol, measurements, controls, and claim boundary.
- `src/exo/worker/engines/mlx/latent_memory.py`: production adoption of the
  proven page format, identity checks, Gemma 4 attention mount, and exclusive
  invocation scope. The inactive path does not alter model input.
- `MlxBuilder` / `SequentialGenerator`: injection seam for a trusted resolver
  that supplies an already-selected page by invocation identity. Enabling this
  seam deliberately disables request batching until heterogeneous pages have a
  correct per-row representation; no resolver is configured by default.
- `gemma4_memory_sidecar_mlx.py`: latent-page representation and mounting seam.
- `grounded_memory_selection.py`: deterministic catalog, capability, epoch, and
  page-bound proof derivation.
- `train_gemma4_conversation_memory.py`: frozen writer and held-out page capture.
- `evaluate_gemma4_conversation_memory.py`: fresh-process latent recall.
- `evaluate_grounded_conversation_memory.py`: end-to-end authoritative
  selection and fresh-process recall with unrelated-corpus controls.
- `durable_grounded_memory.py`: root-last publication and verified reopen over
  Astrid's computation-store adapter.
- `evaluate_durable_grounded_memory.py`: two-process durable publication and
  latent-recall gate with no source-artifact path in the reader.
- `neural_parameter_pages.py`: capability-scoped deterministic page selection.
- `native_kv_continuation_baseline.py`: exact hot-continuation control.
- `NEURAL_PARAMETER_PAGES.md`: chronological experiment registrations and
  results, subordinate to this contract.
- `PERSISTENT_WEIGHT_MEMORY.md`: LoRA experiment record, subordinate to this
  contract.

The production seam does not implement grounding, authorization, page
compilation policy, or a public request API. Those remain upstream concerns.
In particular, it cannot derive a page from prompt text and it cannot make the
failed internal multi-page routing experiments valid.
