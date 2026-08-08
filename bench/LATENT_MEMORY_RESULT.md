# Persistent latent attention memory: formal result

Status: experimentally demonstrated and reproduced through the production Exo
MLX mount. This document records the mechanism and the evidence obtained. It
does not supersede the governing claim boundaries in
[`MEMORY_CONTRACT.md`](MEMORY_CONTRACT.md).

Date: 2026-08-08  
Model: `mlx-community/gemma-4-31b-it-4bit`  
MLX: `0.32.0`  
MLX-LM: `0.31.3`  
Production integration commit: `e126c1a7`  
Production replay commit: `21c401c9`

## Result

A frozen Gemma 4 model can use a durable, removable numerical memory formed
from an earlier natural-language statement without receiving that statement,
its token ids, or its historical conversational K/V cache during recall.

The demonstrated memory is an immutable **latent attention page**. It contains
model-native key/value tensors for selected full-attention layers. During one
later invocation, the current hidden state attends to those tensors through a
separate softmax and adds the result to the ordinary attention output. The page
does not occupy prompt positions and is not appended to the conversational K/V
cache. Removing the page restores the exact unpaged execution path.

In the registered held-out experiment, a fresh model process reproduced 32/32
exact answers from sixteen persisted pages. All page identities survived the
process boundary, and deliberately mounting the wrong page recovered the
queried fact 0/8 times.

This establishes a persistent latent-memory mechanism. It does not establish
autonomous memory selection, arbitrary factual learning, unbounded capacity,
or a final compact representation.

## Mechanism

The mechanism has four operations: capture, compile, persist, and mount.

### 1. Capture a native latent representation

The source statement is processed once through Gemma's ordinary chat template.
Let the resulting embedding sequence be

```text
X = Embed(Tokenize(source_statement)).
```

Gemma 4 alternates sliding and full attention. At each of its ten
full-attention layers

```text
L = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59},
```

the capture path computes the same normalized key and value projections used
by that layer:

```text
K_native[l] = KNorm[l](KProj[l](H[l]))
V_native[l] = VNorm[l](VProj[l](H[l]))
```

with the tensors arranged as

```text
[batch = 1, key/value heads, source slots, head dimension].
```

The source slots are numerical activations. The persisted page does not contain
the source string or its token ids.

### 2. Compile the captured representation into readable memory

A shared `NativeMemoryCompiler` transforms the captured keys and values. For
each full-attention layer it contains two independent residual projections,
one for keys and one for values:

```text
C(z) = z + W_up GELU(W_down z).
```

For the demonstrated model:

- head dimension: 512;
- residual rank: 32;
- memory layers: 10;
- independent key and value transforms per layer; and
- total writer parameters: 655,360.

The count follows directly from

```text
10 layers × 2 transforms × (512×32 + 32×512) = 655,360.
```

Gemma's parameters remain frozen. Only the shared compiler is optimized.
Training presents a memory page alongside a later question and minimizes
cross-entropy over the expected answer tokens. The compiler therefore learns a
common transformation from Gemma's statement activations into keys and values
that the unchanged model can read later.

After training, the compiler is frozen. An unseen statement is converted into
a page by the same fixed operation:

```text
Page(statement) = {
    l: (C_key[l](K_native[l]), C_value[l](V_native[l]))
    for l in full_attention_layers
}.
```

No per-memory optimization occurs for the held-out pages.

### 3. Persist an immutable page

Each layer record contains:

```text
LayerMemory = {
    layer_index,
    keys,
    values,
    gate
}.
```

The page record contains:

```text
LatentMemoryPage = {
    page_id,
    model_id,
    runtime_profile,
    ordered_layers
}.
```

`page_id` is recomputed over the model identity, runtime profile, ordered layer
indices, gates, tensor dtypes, tensor shapes, and tensor bytes under the domain
`exo-gemma4-memory-page-v1`. Load rejects missing or additional tensors,
non-canonical layer order, malformed metadata, or an identity mismatch.

The page is model- and runtime-specific disposable derived state. The
authoritative event or conversation remains separate and is required to
rebuild, audit, correct, or erase the memory.

### 4. Mount the page for one invocation

Selection and authorization occur before the model mount. The runtime receives
an already-selected page plus a selection record bound to that exact `page_id`.
Before inference it verifies:

- selection-to-page identity binding;
- model identity;
- runtime profile;
- complete page identity;
- canonical layer set and order;
- finite gate values in `[0, 1]`; and
- tensor shapes against the loaded model.

For a current layer input `H[l]`, ordinary Gemma attention runs unchanged and
produces `A_base[l]`. The mounted channel derives the normal layer query:

```text
Q[l] = QNorm[l](QProj[l](H[l])).
```

It then performs a second, independent attention operation over the page:

```text
A_memory[l] = OProj[l](
    Attention(Q[l], K_page[l], V_page[l])
)

A_effective[l] = A_base[l] + gate[l] × A_memory[l].
```

The ordinary and memory attentions have separate softmax normalizations. The
page is not concatenated with the prompt's keys and values. Consequently:

- memory slots consume no context-window positions;
- memory slots do not advance ordinary K/V-cache offsets;
- the current prompt remains byte-identical; and
- the base weights are never modified.

Activation is held exclusively for the generator's lifetime and cleared in a
`finally` path. A failure, cancellation, or normal completion removes the
page. Exo currently chooses sequential generation whenever this resolver is
enabled because the loaded model holds one active page set. Requests with
different pages are not admitted to ordinary heterogeneous batching.

## Experimental protocol

### Training set

The experiment used 32 artificial private calibration facts. Each fact paired
a novel fictional instrument identifier with one of eight short uppercase
answer words. Four statement forms and four later question forms varied the
surface language.

The shared rank-32 compiler was trained for 320 updates with Adam at `1e-3`.
The frozen Gemma model supplied both the captured native representations and
the downstream answer loss, but its parameters were not optimized.

### Held-out set

Eight new instrument identifiers and held-out value assignments were excluded
from compiler training. Each was expressed in two unseen statement forms,
producing sixteen pages. Each page was tested with two unseen question forms,
producing 32 exact-recall cases.

The compiler was frozen before any held-out page was formed. Therefore success
required one shared transformation to generalize from training conversations
to unseen statements and values; it was not sixteen separately fitted
adapters.

### Fresh-process replay

After page creation and the original evaluation, the writer/model process
exited. A new process then:

1. loaded Gemma 4 from its model files;
2. loaded each serialized page using the production Exo page decoder;
3. recomputed and checked every page identity;
4. mounted exactly one page for a question;
5. generated at temperature zero; and
6. unmounted the page before the next case.

The fresh process was not given source statements, source token ids,
historical conversational K/V, a textual summary, candidate answers, a memory
operation description, or a routing hint.

## Measurements

### Behavioral result

| Measurement | Result |
|---|---:|
| Held-out page/question cases | 32 |
| Exact latent recalls | 32/32 |
| Base-model recalls without a page | 0/8 |
| Wrong-page recovery of requested value | 0/8 |
| Exact base restoration after unmount | 8/8 |
| Fresh-process exact recalls | 32/32 |
| Stable page identities after serialization | 16/16 |

The fresh-process replay used:

```text
case_digest   = 85e24c89fa50544261e653b7e190c354860a82f2d84f2b90e447ebe20d7197b1
writer_digest = 1c629ac8eeff1000f2a3c1b09f64267ecef284247dfc9c0fcbdb31d1c0d7d718
```

### Formation cost

| Measurement | Result |
|---|---:|
| Capture of 32 training statements | 13.851 s |
| Compiler training | 320.620 s |
| Held-out page capture and freeze | 7.015 s |
| Compiler parameters | 655,360 |

These are experimental training costs, not the steady-state cost of recalling
an already-created page.

### Astrid-backed durability result

A separate storage gate published eight selected pages and their canonical
privacy-domain catalog through Astrid's computation-store adapter. The writer
then exited. A fresh reader received the model path and store path, but not the
original artifact directory, and reproduced 16/16 exact recalls after verified
reconstruction.

| Measurement | Result |
|---|---:|
| Pages published | 8 |
| Exact recalls after verified reopen | 16/16 |
| Publication and root advancement | 0.423 s |
| Verified reopen and projection | 0.088 s |
| Authoritative store size | 25,275,789 bytes |

The recovered manifest identity was
`d005a297de851cef4bda1224193ad93543b9f6c4d2a18dc870efaf28158d3615`.
The production decoder adopts the same page encoding and identity construction
byte-for-byte, with a compatibility regression that loads an experiment page
through the production implementation.

### Representation size

| Measurement | Result |
|---|---:|
| Held-out pages | 16 |
| Total serialized bytes | 49,517,584 |
| Mean bytes per page | 3,094,849 |
| Smallest page | 3,033,409 bytes |
| Largest page | 3,197,249 bytes |

This representation is functional, not compact. Its size scales with the
number of captured source slots, memory layers, key/value heads, head dimension,
and tensor precision.

### Initial execution-cost probe

A separate real-model probe used the production mount with a three-token
current input and a ten-slot page:

| Measurement | Result |
|---|---:|
| Inactive warm forward | 74.7 ms |
| Active warm forward | 77.6 ms |
| Difference | +3.9% |
| Ten-slot resident page | 819,200 bytes |
| Ordinary K/V offset | 3 tokens in both paths |

This is one warm sample, not a latency distribution. It demonstrates that the
page did not enter ordinary K/V and provides an initial overhead observation;
it does not justify a general performance claim.

## Controls and failure conditions

The following controls distinguish the mechanism from easier substitutes:

- **No-page control:** the base model recovered 0/8 held-out values.
- **Wrong-page control:** another valid page recovered the requested value 0/8
  times.
- **Revocation control:** removing the page restored exact base logits and
  behavior.
- **Serialization control:** all sixteen page identities were stable in a new
  process.
- **Prompt exclusion:** source statements and memory instructions were absent
  from recall prompts.
- **K/V exclusion:** historical conversational K/V was not restored, and page
  slots did not change current K/V offsets.
- **Identity substitution:** a proof for one page cannot activate another.
- **Malformed-page rejection:** stale identity, incompatible model/runtime,
  wrong layer order, invalid gates, and incompatible shapes fail before model
  execution.
- **Lifecycle isolation:** nested replacement is rejected and exceptions clear
  active state.

An earlier typed-interface router control was declared VOID because it taught
Gemma a memory operation through its prompt. That result is not part of this
evidence. The successful latent recall described here supplies no such
interface to Gemma.

## Distinction from adjacent mechanisms

### Not retrieval-augmented prompting

RAG retrieves text or embeddings and normally places rendered information into
the model's input. This mechanism supplies no source text at recall. The
retrieved object is a model-native K/V page consumed inside attention.

### Not conversational K/V persistence

Persisted conversational K/V reuses the exact positional computation of an old
prefix and remains bounded by the context architecture. A latent page is
formed by a shared compiler, has no conversational position, and is attended
through a separate channel by a fresh query.

### Not LoRA or permanent fine-tuning

The base weight matrices are unchanged. The per-memory object is not a weight
delta. One shared trained compiler produces removable per-memory K/V pages.

### Not prefix tuning

Prefix tuning commonly presents learned vectors as virtual prefix positions in
ordinary attention. This implementation uses a separate softmax and does not
extend the ordinary prompt or cache length.

### Not autonomous memory

The model does not decide which page to load. Page selection, authorization,
time, provenance, and conflict policy remain outside Gemma. This separation is
a correctness property, not a missing trick hidden inside the result.

## Supported claim

The evidence supports the following statement:

> For the tested Gemma 4 model and artificial held-out fact task, a single
> frozen low-rank compiler can turn unseen natural-language statement
> activations into durable, identity-bound attention K/V pages. A fresh frozen
> model can later recall the held-out values exactly when the corresponding
> page is mounted through a separate attention channel, without replaying the
> source text, source token ids, or historical conversational K/V.

The evidence does not yet support these stronger statements:

- that arbitrary conversations can be compressed into useful pages;
- that pages preserve rich episodes, causality, or temporal order;
- that one page supports open-ended paraphrase and reasoning;
- that many simultaneously mounted pages compose without interference;
- that the current representation is storage-efficient;
- that pages can migrate across model or runtime changes;
- that memory formation should occur for every turn;
- that Gemma can safely select memories by itself; or
- that latent pages replace authoritative durable evidence.

## Implementation and reproduction map

- Production page format and attention mount:
  `src/exo/worker/engines/mlx/latent_memory.py`
- Exo builder and invocation lifetime:
  `src/exo/worker/engines/mlx/builder.py` and
  `src/exo/worker/runner/llm_inference/batch_generator.py`
- Shared compiler and training harness:
  `bench/gemma4_memory_sidecar_mlx.py` and
  `bench/train_gemma4_conversation_memory.py`
- Fresh-process production replay:
  `bench/evaluate_gemma4_conversation_memory.py`
- Real-model execution probe:
  `bench/probe_gemma4_memory_sidecar.py`
- Unit and compatibility tests:
  `src/exo/worker/engines/mlx/tests/test_latent_memory.py` and
  `bench/tests/test_gemma4_memory_sidecar_mlx.py`

The chronological experiment history, including failed and VOID lines, remains
in [`NEURAL_PARAMETER_PAGES.md`](NEURAL_PARAMETER_PAGES.md). Those failures are
part of the result boundary and must not be silently reinterpreted as support
for this mechanism.
