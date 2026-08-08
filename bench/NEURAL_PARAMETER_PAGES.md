# Neural parameter pages

Status: pre-registered experiment contract. This document defines what the
prototype may claim before the implementation or its results exist.

The system-level architecture and the distinction between authoritative
memory, grounded selection, latent pages, and hot continuation K/V are
normative in [`MEMORY_CONTRACT.md`](MEMORY_CONTRACT.md). This file records the
chronological experiments underneath that contract.

## Question and determination boundary

Can a frozen language model acquire a removable, principal-scoped neural
extension whose activation is derived from exact facts rather than guessed from
the prompt?

The proposed mechanism is a **neural parameter page**: a content-identified,
model-specific low-rank residual mounted at named linear projections for one
inference invocation. Tensor Logic selects pages from authoritative relations;
the model consumes the selected numerical residuals. The base model is never
modified.

Passing the experiment would establish scoped parametric extension. It would
not establish exact factual recall, continual learning, general reasoning,
unbounded memory, or that Tensor Logic can judge neural output. Exact events,
time, identity, capabilities, and proofs remain outside the weights.

## Why this is not ordinary LoRA or ordinary routing

The preceding persistent-weight experiment found that a globally active LoRA
could preserve narrow behavior across process restarts, but it also overwrote
unrelated behavior. Additional rank and prompt diversity did not make it a
relation engine. Explicit attachment was cheap enough (7.46--8.02 ms for a
31.25 MiB adapter on the measured machine) that scoping, not attenuation, is
the safety mechanism.

Several established mechanisms border this design:

- LoRA supplies the low-rank residual representation.
- Mixture-of-experts and PEER show that sparse expert selection can decouple
  stored capacity from active compute.
- Hypernetworks and fast-weight systems show that weights can be generated or
  updated by another computation.
- Titans demonstrates a neural memory updated at test time.
- LoRA composition demonstrates that separately trained residuals can sometimes
  be combined, while also making composition an empirical property rather than
  an algebraic entitlement.

The new boundary is operational: page selection is an exact, capability-scoped
derivation over content identities. A learned router does not decide which
principal's memory is mounted. Tensor Logic proves the selection, but does not
authorize access and does not certify the page's behavioral quality.

## Page ABI

A page is bound to all semantics that can change its effect:

```text
PageId = identity(canonical PageRecord)

PageRecord = {
    base_model_id,
    runtime_profile_id,
    training_recipe_id,
    evidence_closure_id,
    privacy_domain_id,
    page_version,
    sites: [SitePage...]
}

SitePage = {
    site_name,
    input_width,
    output_width,
    rank,
    scale,
    dtype,
    A[rank, input_width],
    B[output_width, rank]
}
```

`runtime_profile_id` names the semantics-visible numerical contract: model
implementation, quantization interpretation, dtype and accumulation rules, and
the injection convention. Engine builds preserving those semantics reuse the
same profile. A page for another base model, profile, shape, or privacy domain
is incompatible, not approximately compatible.

The first implementation permits at most one page at a site. Arbitrary adapter
addition is not safe composition. Later composition requires a separately
identified composition contract and its own interference evaluation.

## Forward equation and cost

For a named base projection `W_l` with input `x_l`, an active page contributes
a low-rank residual:

```text
y_l = W_l x_l + sum[p in Active(l)] alpha_p (scale_p / rank_p) B_p A_p x_l
```

For one page at one site, the stored parameter count is

```text
rank * (input_width + output_width)
```

and the additional multiply-accumulate count per token is the same quantity.
Counting a multiply and add separately gives approximately twice that many
floating-point operations. Relative to a dense projection, the leading compute
ratio is

```text
rank * (input_width + output_width) / (input_width * output_width).
```

The page bus enforces a perturbation budget. By submultiplicativity,

```text
||delta y_l||_2 <= ||x_l||_2 *
    sum[p] |alpha_p * scale_p / rank_p| * ||B_p||_2 * ||A_p||_2.
```

The prototype uses an auditable conservative norm bound and rejects a selected
set whose total exceeds the site's registered budget. A small page is not
therefore automatically a safe page; the empirical behavior gates still rule.

When no page is active, the page bus must bypass residual evaluation entirely.
Given the same base model, runtime profile, inputs, and random state, the output
must be byte-identical to execution without the page mechanism.

## Tensor Logic activation contract

The reasoner consumes ground facts such as:

```text
belongs_to(principal, privacy_domain)
granted(principal, page)
serves(page, concept, first_epoch, last_epoch)
compatible(page, base_model, runtime_profile)
requested(invocation, concept, epoch)
```

At temperature zero it derives a sparse activation relation:

```text
active(invocation, page, site, gate)
```

The relation can be represented as a sparse Boolean or weighted tensor over
`(invocation, page, site)`. Joins and projection are tensor contractions; the
result is canonically sorted and accompanied by a proof identifying the fact
snapshot and rule profile.

Capability enforcement is still the policy gate. The reasoner may return only
pages already present in the invocation's granted namespace. An ungranted page
is a failed derivation, not a request for ambient access. The first experiment
selects once per invocation. Per-token routing from hidden activations would be
a different, learned mechanism and is outside this contract.

## Page construction

Tensor Logic does not write matrices. A page writer is a deterministic or
snapshot-bound training invocation over:

```text
(base model, runtime profile, evidence closure, recipe, seed, target ABI)
```

Its output page and evaluation evidence are content-identified Derived objects.
The evidence closure remains authoritative and permits rebuilding or auditing
the page. A page can encode language habits, procedures, or useful bias; it is
never the only copy of an event or proof.

Registration requires an authority distinct from the executing principal. A
page that can steer model output is executable content. It must be identity
verified, profile compatible, capability granted, and admitted under the
privacy domain's learning and retention policy.

## State, time, and deletion

Updates create new immutable page versions. Tensor Logic selects a version by
the invocation epoch; it never mutates a page in place. Old versions remain
reproducible while retained. Exact temporal questions are answered from the
event store; a current page may bias interpretation but is not a clock or an
event log.

Unmounting or revoking a page removes it from the next activation proof.
Deletion includes clearing process-local materializations and selection caches.
The following no-page execution must again be byte-identical to the base path.

## Threat model and failure rules

- **Cross-principal substitution:** Page records bind privacy domain, base,
  profile, shapes, and bytes. Selection proofs bind the principal's fact
  snapshot. Physical dedup never grants logical visibility.
- **Prompt-selected authority:** prompts may identify a concept, but cannot
  name or grant a page. The reasoner intersects requests with capabilities.
- **Adapter collision:** two pages claiming the same site fail closed unless a
  registered composition contract explicitly permits the set.
- **Unbounded perturbation:** shape, rank, gate, and norm budgets are checked
  before attachment.
- **Stale or malicious page:** behavioral evidence can mark a page unusable;
  numerical output never authorizes an effect. Ordinary Astrid capability
  checks remain the only effect boundary.
- **Cache residue:** domain, generation, and runtime profile participate in
  cache keys. Revocation purges the logical entry; a cache hit is never exposed
  as a guest-visible fact.
- **Self-grading:** the page writer cannot admit its own result. Evaluation
  authority is granted at privacy-domain scope and binds its assessment to the
  page and evidence closure.

## Pre-registered gates

The implementation is rejected if any mandatory gate fails:

1. **Zero-page identity:** the page-capable path is byte-identical to the base
   path when the activation set is empty.
2. **Principal isolation:** two privacy domains with contradictory invented
   rules mount different pages from the same neutral request; neither page is
   observable or active in the other domain.
3. **Context-free application:** after training context and KV state are
   removed, a fresh process applies the mounted page on held-out paraphrases.
   No supporting memory text is passed to the model in this test. Before the
   balanced `serevin-parameter-pages-v2` test is observed, the acceptance bar
   is fixed at at least 75% exact one-word accuracy independently in each of
   the two contradictory privacy domains. Failure in either domain rejects
   the learned-page claim; the threshold is not tuned after inspection.
4. **Temporal replacement:** an epoch change selects the new immutable page;
   the old page and its prior result remain reproducible by identity.
5. **Revocation:** unmounting a page restores byte-identical base behavior and
   leaves no selectable process-local residue.
6. **Composition safety:** independent non-overlapping pages retain their
   individual controls. Conflicting pages at one site are rejected rather than
   silently summed.
7. **Resource bound:** measured resident bytes and token latency remain within
   the declared parameter and operation model; any unexplained growth fails.
8. **Restart stability:** page identities, activation-proof identity, and
   deterministic outputs match across fresh processes under the same runtime
   profile.
9. **Provenance:** every activation names the exact page, fact snapshot, rule
   profile, principal/domain, epoch, and grant set. Missing evidence fails
   closed.
10. **Collateral behavior:** unrelated deterministic controls remain within a
    pre-registered tolerance. Teacher-forced loss is not accepted as a
    substitute for fresh-process generation.

The synthetic algebra simulator must pass gates 1, 2, 4--6, 8, and the
structural portion of 9 before a Gemma layer is modified. The Gemma experiment
then evaluates gates 3, 7, 10, and the empirical half of 8.

## Numerical seam result

The pre-model simulator passed all six implemented gates. The first real-model
probe then wrapped layer 59's quantized `gate_proj` in
`mlx-community/gemma-4-31b-it-4bit` (MLX 0.32.0, MLX-LM 0.31.3). A rank-1 page
contained 26,880 parameters. Across two fresh processes:

| State | Last-token logits SHA-256 |
| --- | --- |
| Untouched base | `c0d4f6466e596c6b5e1a10e28fba168b5b3bdba7da076a4d6efe7b70cd84512f` |
| Page mechanism present, inactive | same as base |
| Page active | `fcc63a77351fae75fa5a3f8eecbfdf03eda2519141c20dfb82f05c66c74bba31` |
| Page revoked | same as base |

Thus the real quantized model satisfies byte-exact zero-page identity and
revocation for this injection seam. The probe used deterministic synthetic
page matrices; it establishes mechanics only, not learned behavior. Timings
are retained in the generated probe JSON but are not claimed from two warm
samples.

### First learned-page run: void

The first Serevin corpus (`serevin-parameter-pages-v1`) is not evidence. Its
template index was derived from the same case-index parity as the reachable
label, so each prompt form was perfectly label-correlated. The selected Alpha
checkpoint scored 15/64 on held-out generation, but that number is **VOID**:
the corpus allowed a template shortcut and the unseen templates exposed it.
The successor corpus changes identity and requires each template to contain an
equal number of reachable and unreachable cases, enforced by a regression
test. No threshold was weakened.

### Successor learned-page run: pre-registration

Before inspecting any v2 held-out generation, the successor run is fixed as
follows:

- corpus format: `serevin-parameter-pages-v2`;
- case-manifest SHA-256:
  `cbebe187c6c16da7b5693b722c49b8d5c2b9c076af160f76c14b2c297dcfae79`;
- model: `mlx-community/gemma-4-31b-it-4bit`;
- page shape: rank 8, scale 8, dropout 0, final 8 transformer layers;
- optimizer run: 250 iterations, batch size 1, learning rate `1e-5`, seed 917;
- checkpoint selection: lowest complete validation loss at the fixed 25-step
  checkpoints, with no held-out test inspection;
- held-out set: 64 disjoint entities under two unseen prompt forms; and
- acceptance: at least 48/64 exact answers in Alpha and independently at least
  48/64 in Beta, whose labels are the exact inverse on byte-identical prompts.

The Alpha result is observed first. Beta training runs only if Alpha passes,
because a failed first domain already falsifies the registered learned-page
claim and spending a second full training run cannot repair it.

### Successor learned-page result: failed gate

Validation-only selection chose step 200 (validation loss 0.274), before the
run degraded to 0.534 at step 225 and 0.815 at step 250. The selected page is
31 MiB (8,188,928 parameters, about 0.027% of the frozen 31B model) with
SHA-256
`97a82bd12f4217b076ea6f57c36fac89369f1f86391931402d5db52bee99ef1f`.

The one registered Alpha test then scored 35/64 (54.69%), below the fixed
48/64 bar. Beta was therefore not trained. Post-failure diagnostics, which do
not alter that verdict, located the boundary:

| Path | Split | Exact result |
| --- | --- | ---: |
| Page inactive (byte-exact base path) | unseen entities and forms | 0/64 |
| Page active | unseen entities, familiar forms | 48/64 |
| Page active | unseen entities and forms | 35/64 |

The inactive model emitted neither invented label. The active page therefore
did install new context-free neural behavior, but that behavior coupled too
strongly to the training prompt language and did not robustly acquire the
underlying graph rule. This rejects the stronger claim that a small page can
learn both an exact symbolic relation and its neural realization from this
recipe.

The failure also identifies an architectural duplication: Tensor Logic can
compute graph reachability exactly, while this experiment asked the learned
page to rediscover it approximately. The next admissible experiment must keep
that exact relation in Tensor Logic and test only a typed, non-textual bridge
from a proof/result tensor into model activations. Such a bridge would test
whether the model can *express* structurally selected knowledge; it must not be
reported as the model independently learning or proving the relation.

## Typed proof-to-neural bridge

Let Tensor Logic derive a one-hot result tensor `z` for a registered relation
contract. A bridge contract maps each nonzero coordinate to an immutable page:

```text
z_j = prove(relation_j, fact_snapshot, query)
page_set = { bridge(relation_j, value_j) | z_j = 1 and grant(page_j) }
h_(l+1) = W_l h_l + sum_j z_j * P_(l,j)(h_l)
```

`z` is obtained from the canonical proof, never from prompt similarity or a
learned router. The activation receipt binds the proof, fact snapshot, query,
bridge contract, selected page, domain, and runtime profile. The model output
is still non-authoritative; the proof remains the exact answer.

This is deliberately narrower than neural concept learning. The page is a
learned realization of a typed symbolic value in one model's activation
language. Tensor Logic supplies meaning and composition; the page supplies an
efficient way for the frozen model to express it without retrieving a textual
definition into context.

### Bridge experiment pre-registration

Before either held-out bridge result is observed:

- the v2 Serevin facts and disjoint splits above remain fixed;
- two shared pages are trained, one to realize `VELA` and one to realize
  `NOMA`; neither page receives a domain identity;
- page recipe: rank 2, scale 2, dropout 0, final 4 transformer layers, 75
  iterations, batch size 1, learning rate `1e-5`, seed 917;
- checkpoint selection is lowest complete validation loss at fixed 15-step
  checkpoints, without test inspection;
- Alpha's exact TL mapping is reachable→VELA, unreachable→NOMA; Beta uses the
  inverse mapping over the byte-identical queries;
- each domain must score at least 60/64 exact held-out answers under unseen
  prompt forms; and
- an intentionally inverted activation proof must invert at least 60/64
  answers. Otherwise prompt-side reasoning, rather than the typed selector,
  may be responsible for the result.

The prompt may name the two opaque output symbols so the base tokenizer has an
output vocabulary, but it contains no definition of their relation to the
facts and no selected answer. This experiment proves only the bridge seam. It
does not prove free-form memory, general concept acquisition, or that neural
outputs inherit the proof's authority.

### Bridge experiment result: failed capacity budget

The VELA page's validation loss improved monotonically from 6.997 to 1.821,
so the registered step-75 checkpoint was selected. It contains 1,023,488
parameters (0.003% of the base), occupies 3.9 MiB, and has SHA-256
`5d2f21f0e138b6cecb05130ce6e89016eba464b3bfb34e8a53bad39324f8d2c0`.

It scored 0/32 both when selected by valid Alpha proofs and when selected by
intentionally inverted proofs. Most generations entered the model's thought
channel rather than emitting either exact symbol within the fixed eight-token
budget; two malformed lowercase/plural attempts were also rejected. NOMA was
not trained because VELA alone made the registered 60/64 aggregate impossible.

This is a negative on the chosen neural realization budget, not on Tensor
Logic selection: the page never learned its single value strongly enough for
the selector to test. A successor may change optimization or page capacity
only under a new corpus identity and unseen evaluation set. The 60/64 exact
bar and the inverted-proof control remain unchanged.

## Resulting direction

The evidence does not justify another LoRA capacity sweep. LoRA changes the
model's projection functions, while the bridge input is already a sparse,
typed value. The closer interface is a content-addressed continuous prefix or
residual activation page selected by the proof:

```text
proof tensor z -> canonical page identities -> virtual activation blocks
                                          -> frozen model attention/residual stream
```

This keeps the same identity, capability, receipt, revocation, and zero-page
rules. It changes only the physical representation of a neural page. A prefix
page must be keyed by model and runtime profile, bounded by the resident-memory
authority, and treated as a disposable `Derived` representation. The next
experiment must compare a soft-prefix page and a residual steering page under
the same fresh-process, unseen-form, inverted-proof, and collateral controls.
Neither is allowed to carry the sole copy of a fact or proof.

## Gemma 4 global-memory sidecar

Inspection of the pinned Gemma 4 31B implementation identified a less noisy
seam than ordinary prefix or residual injection. Its 60 layers contain ten
full-attention layers (5, 11, ..., 59) among fifty sliding-window layers. The
global layers use four 512-dimensional K/V heads with K=V at projection input.

The sidecar gives those ten layers a second attention operation over a memory
bank. It does not concatenate memory with conversational K/V:

```text
self = Attention(native_query, conversation_KV)
memory = Attention(native_unrotated_query, proof_selected_memory_KV)
output = native_o_proj(self + gate * memory)
```

The separate softmax prevents memory from competing with prompt length. The
bank does not consume positions, alter RoPE offsets, or enter the rotating
cache. An absent page skips the branch before query projection and must remain
byte-identical to the unwrapped model.

### Mechanical result

On the real quantized model under MLX 0.32.0 / MLX-LM 0.31.3, a ten-slot page
derived through Gemma's native K/V projections occupied 819,200 bytes and
compiled in 204 ms. All ten global layers received memory while every ordinary
cache offset remained exactly the three prompt tokens. Base, mounted-inactive,
and revoked logits shared SHA-256
`c0d4f6466e596c6b5e1a10e28fba168b5b3bdba7da076a4d6efe7b70cd84512f`;
active logits changed to
`1db8a9cb1f7b9b4cfee6a6e4e2666792562e6476a05ad986cabfc50d27b516dc`.

An untrained semantic probe then compiled the private statement “code XQ-17
maps to VELA” into nineteen native-projected slots. The active model produced
nonsense rather than VELA. Native projection therefore satisfies geometry but
not legibility: the sidecar needs a learned bridge, and arbitrary memory must
not be admitted as if geometry alone supplied meaning.

### Channel-capacity gate: pre-registration

Before training a general compiler, optimize one direct memory bank while all
Gemma weights remain frozen. This deliberately non-scalable test asks only
whether the side channel can carry a readable signal:

- one private Serevin fact, six training question forms, and two unseen forms;
- direct K/V bank initialized from the native-projected private statement;
- ten global layers, nineteen slots, fixed gate 1;
- Adam, learning rate `1e-3`, 100 updates, deterministic order;
- exact `VELA` generation on both unseen forms in a fresh evaluation pass;
- page revocation restores the byte-identical base response; and
- unrelated controls remain unchanged when the page is inactive.

Failure rejects this sidecar equation before any compiler is built. Success
proves only channel capacity. The direct bank is not persistent memory: the
next gate must train one shared structured-fact compiler and freeze it before
evaluating unseen facts.

### Channel-capacity run 1: failed termination

The direct bank reduced teacher-forced label loss from 14.3125 to zero by
update 10 and retained zero loss through update 100. On both unseen forms the
active model immediately emitted VELA, while the revoked model did not know
the value. However, neither response was exactly `VELA`: both repeated the
symbol because the training record supplied no end-of-turn target. The exact
2/2 generation gate therefore failed. The page was 1,556,480 bytes with
identity
`12b4a6a48728f5cbabf466d8aca992d2566a980f2c8c116d1923936e2c946e44`.

This establishes that the dedicated channel is legible but does not yet
establish controlled realization. A single successor is registered before new
evaluation: include the tokenizer's end-of-turn token in every answer target,
retain the same fact, bank shape, six training forms, optimizer, learning
rate, 100 updates, and strict exact-output bar, and replace both held-out forms
with previously unseen wording. No other parameter changes.

### Channel-capacity run 2: passed

The termination-aware successor again reached zero teacher-forced loss by
update 10. In the fixed final evaluation both new question forms generated
exactly `VELA` and stopped; after revocation neither response contained the
private value. Training took 98.04 seconds. The immutable 1,556,480-byte page
has identity
`b773bc4495bbdfbd6915881e3c65d89fcd21f13848e3b52a1d2b6db31f471913`.

This proves that Gemma can read a dedicated global-attention memory bank. It
does **not** prove scalable memory because that bank was optimized for one
fact. The next claim is strictly harder: train one compiler over a registered
fact population, freeze it, and compile unseen facts with no optimizer step.
Per-fact tuning after the freeze is prohibited.

### Shared compiler gate: pre-registration

The first compiler test uses canonical key/relation/value records. It does not
yet claim arbitrary Tensor Logic programs; it tests whether one frozen neural
ABI can carry new members of a fixed typed relation:

- 32 training facts and eight held-out facts with disjoint opaque keys;
- eight value symbols, all represented in training, assigned independently of
  key spelling;
- canonical memory input
  `SEREVIN_ENTRY(key=..., relation=MAPS_TO, value=...)`;
- native-projected K/V followed by one shared rank-32 residual head transform
  for keys and values at each of the ten global layers;
- 655,360 trainable compiler parameters; Gemma remains frozen;
- Adam, learning rate `1e-3`, 320 deterministic single-example updates;
- four training question forms and two held-out forms;
- compiler weights freeze before any held-out fact is compiled;
- no optimizer step, gradient, or mutation after the freeze;
- at least 14/16 exact value generations across the eight unseen facts and two
  unseen forms; and
- wrong-page activation must not reproduce the original fact's value on more
  than two of eight first-form queries.

Passing proves unseen *fact* compilation within one relation/value vocabulary.
It does not yet prove unseen relations, concepts, composition, or a token-free
Tensor Logic input. Those require later gates and must not be inferred from
this result.

### Shared compiler result: passed

The rank-32 compiler reached zero training loss after the first 32-fact pass
and remained stable through all 320 updates. Compiler weights were then frozen
and serialized with SHA-256
`c813a64cb1b50138d5e3a3eaf2e6bb9e5712e5a9881cdf851a0def301791eda9`.
No held-out page had been supplied to the optimizer.

All eight unseen facts generated their exact value under both unseen question
forms: 16/16, exceeding the registered 14/16 bar. Deliberately activating a
different fact's page reproduced the original fact's value 0/8 times. A fresh
process then reloaded only the frozen compiler and registered corpus, compiled
the eight facts again, reproduced all 16 page identities byte-for-byte, scored
16/16 again, and retained the 0/8 wrong-page result. The frozen compiler has
655,360 parameters; training took 318.12 seconds, while native projection of
all 40 fact records took 9.62 seconds on the local M2 Ultra.

This is evidence that one shared, frozen bridge can compile previously unseen
facts into model-readable external memory with no per-fact optimization. The
claim remains deliberately narrow: the relation and eight output values were
represented during bridge training, and the compiler input was a canonical
lexical record rather than a token-free Tensor Logic tensor. The next gate is
therefore unseen relation/value composition from typed TL inputs—not a larger
version of this same key/value corpus.

## Conversation-latent memory gate: pre-registration

The canonical-record experiment does not establish ordinary memory formation.
It begins with an interpretation that another component has already written
down. The next experiment removes that component. Gemma reads an ordinary user
turn, and the writer consumes only the native hidden states produced by that
forward pass. No tool call, extracted triple, canonical restatement, or
optimizer step is permitted for a held-out memory.

The fixed experiment is:

- 32 training conversations and eight held-out conversations, with disjoint
  invented entity names;
- one stable relation and eight opaque values, all values represented in the
  training split, so the experiment tests new memories rather than new output
  vocabulary;
- four balanced training statement forms and two statement forms withheld in
  full until evaluation;
- four training question forms and two question forms withheld in full until
  evaluation;
- the memory source is the full-attention-layer activation trace produced when
  Gemma reads the ordinary user turn through its normal chat template;
- one shared rank-32 residual writer over the ten global-layer K/V projections,
  trained for 320 deterministic single-example updates with Adam at `1e-3`;
- Gemma remains frozen, and writer weights freeze before any held-out
  conversation is captured;
- each held-out page is serialized, the originating tokens and ordinary KV
  cache are discarded, and a separate process loads only the unchanged Gemma,
  frozen writer identity, serialized latent page, and a new question;
- at least 28/32 exact generations over eight unseen conversations, two unseen
  statement forms, and two unseen question forms;
- the unpaged model may produce the private value for no more than two of eight
  first-form questions;
- a deliberately wrong page may produce the queried conversation's original
  value for no more than two of eight first-form questions; and
- revocation restores the unpaged deterministic response.

Passing establishes a narrow but materially different claim: a shared frozen
writer can preserve Gemma's own internal response to a previously unseen
ordinary conversation and make it usable after process death without replaying
the conversation into the new context. It does not establish autonomous
selection, lifetime-scale capacity, temporal reasoning, arbitrary relations,
or a model-independent memory representation. The source turn and its receipt
remain authoritative; the neural page is disposable derived state.

### Conversation-latent memory result: passed

The shared writer reached zero calibration loss after the first 32-conversation
pass and remained there through the registered 320 updates. Gemma stayed
frozen. The writer has 655,360 parameters, serialized to 2.5 MiB, with SHA-256
`1c629ac8eeff1000f2a3c1b09f64267ecef284247dfc9c0fcbdb31d1c0d7d718`.
Training took 319.60 seconds on the local M2 Ultra.

After the freeze, the writer captured both withheld statement forms for each
of eight unseen fictional instruments. With the originating turn absent from
the inference prompt, those pages produced the exact private value on all
32 combinations of unseen statement and question forms. The unpaged model
produced 0/8 private values, deliberately wrong pages produced 0/8 original
values, and all eight revoked executions reproduced their corresponding
unpaged response.

A separate Python process then loaded a new Gemma instance and the 16 serialized
pages. It did not load the originating statements, token ids, ordinary KV cache,
or writer weights into inference. All 16 page identities reproduced, the replay
again scored 32/32 exact, and the wrong-page control remained 0/8. The case
registry has SHA-256
`85e24c89fa50544261e653b7e190c354860a82f2d84f2b90e447ebe20d7197b1`.
Each page contains only 20 K/V tensors plus model/profile/layer identity
metadata; a literal-string scan found none of the entity, value, or statement
text.

The initial same-context diagnostic recorded 0/16 because its 16-token output
limit cut Gemma 4 off inside its emitted thought channel before the final
answer. This diagnostic was not an acceptance gate. Re-running it after the
result with a 128-token limit and extracting the model's final channel produced
16/16 correct answers, confirming that the source turns themselves were
understood. No memory result was recomputed or rescored from that diagnostic.

This is the first evidence in this program for persistent latent memory rather
than prompt retrieval: Gemma reads an ordinary turn once, a frozen shared
writer preserves its internal activations, and a fresh Gemma uses the resulting
numerical page without seeing the turn again. The result is still deliberately
narrow. It covers one known relation and a known output vocabulary; the page is
3.03--3.19 MiB for one short turn, preserves every captured slot, and may be as
privacy-sensitive as its source. Selection, compression, multi-memory
interference, correction, temporal supersession, and broader semantic transfer
remain open experiments.

### Internal-binding successor: pre-registration

The passed run mounted the page already associated with the queried entity.
That proves latent persistence but leaves selection outside the model. It is
not yet the ordinary meaning of remembering. The immediate successor therefore
uses the already-frozen writer and already-serialized held-out pages without
additional optimization:

- canonically compose the first withheld-form page for all eight unseen
  conversations into one K/V bank;
- mount that same complete bank for every question, so no host operation names
  or selects the queried fact;
- require at least 14/16 exact generations across the two withheld question
  forms; and
- for each entity, mount the bank containing the other seven memories and
  require the omitted entity's original value on no more than two of eight
  first-form questions.

The composition order is page-identity order and is part of the bank identity.
Model and runtime identities, global-layer set, dtype, head geometry, and gates
must agree before composition. Failure is retained as evidence that the current
writer produces readable pages but not internally addressable memory. Passing
would establish query-to-memory binding across a small simultaneous bank; it
would not establish capacity beyond eight entries or remove the later need for
memory hierarchy and resource accounting.

### Internal-binding run 1: failed

The fixed writer and fixed held-out pages scored 2/16 with all eight memories
mounted. The first withheld question form collapsed to `AMBER` for seven of
eight entities; the second collapsed to `ONYX` for seven of eight. Leaving the
queried memory out of each bank reproduced its original value 0/8 times, so the
bank was carrying the individual values, but its keys did not let Gemma select
the requested one. The 25,067,520-byte bank has identity
`b573eb0806012aefb836f055a9d123c049815b4a5addfed1cc843aa38a8b1806`.

This rejects the claim that the first writer already produced ordinary,
internally addressable memory. Single-page recall required the host to choose
the page. The failure is structurally informative: single-page training gives
the writer no reason to make keys discriminative because every memory softmax
contains only one fact.

### Addressable-writer successor: pre-registration

One successor is registered before additional optimization. It begins from the
passed writer identity above and adds 320 deterministic updates with
simultaneous-memory pressure:

- each training step mounts eight calibration memories containing all eight
  values and sharing one of the four balanced training statement forms;
- the question names one target entity, while the other seven memories are
  genuine distractors inside the same memory softmax;
- the writer remains shared, Gemma remains frozen, and no entity-specific
  parameter or external target selector is introduced;
- Adam remains at `1e-3`; the four training question forms rotate exactly as in
  the first writer run;
- after the update schedule, writer weights freeze before the unseen
  conversations are recaptured;
- the complete eight-entry held-out bank must score at least 14/16 exact over
  the two withheld question forms;
- eight leave-one-out banks may reproduce the omitted entity's value at most
  twice; and
- a separate process must load the serialized complete bank, without source
  turns, writer execution, or ordinary KV state, and reproduce the same gate.

Failure rejects this writer architecture as a scalable internal binder. No
post-hoc gate, bank size, rank, or learning-rate change is permitted under this
experiment identity.

### Addressable-writer run: failed

The successor scored 3/16 after the registered 320 updates. Loss fell during
the first two passes but plateaued around `0.71` and ended at `0.8359375`.
Leave-one-out leakage remained 0/8. The writer therefore learned somewhat less
degenerate values but did not create a general query-addressable key space.
Its held-out bank identity is
`4ea6b2d38ff965c51e65d5bdd7450d6792d78bc672d01e10980a5cc623378f83`.

This rejects the writer-only architecture for internal binding. The sidecar
transforms memory keys and values but leaves each live memory query in Gemma's
native full-attention query space. Answer loss cannot reliably force one side
of that mismatched representation to become a general associative address.

### Shared reader-writer successor: pre-registration

The final registered successor adds the missing symmetric half: one shared
rank-32 residual query projection at each of the ten global layers. The
memory-key/value writer begins from the failed writer identity
`3ea6d1548ab7197d73f1b785f05dccbe85598b4e2469cfd72fe3b050ec2e2e7e`;
the query projections begin as exact identities. Gemma remains frozen.

The experiment otherwise repeats the fixed simultaneous-memory contract:

- 320 deterministic updates, Adam at `1e-3`, banks of eight calibration
  memories with all eight values, and the same question rotation;
- query and memory projections are shared across every entity; no identifier,
  page index, or target mask reaches inference;
- reader and writer freeze before held-out conversations are recaptured;
- one complete eight-memory held-out bank must score at least 14/16;
- leave-one-out banks may reproduce the omitted value at most twice; and
- a separate process must load only the frozen shared adapter, serialized bank,
  fresh question, and unchanged Gemma and reproduce the recall gate.

Failure ends this sidecar line: the next experiment would require an explicit
associative/contrastive objective and therefore a new registered mechanism,
not another tuning pass.

### Shared reader-writer result: failed; sidecar line closed

The reader-writer adapter ended at loss `0.746123` after the registered 320
updates and scored 2/16 on the held-out eight-memory bank. Leave-one-out banks
reproduced the omitted value once in eight cases. The 983,040-parameter adapter
has SHA-256
`8d4d523ae70e58ba53bd8c9df8873b9ca5814192632f19512b8e6584a5fd808a`;
its held-out bank identity is
`03c766dc7d52f5c01a3203af6bd2404452161b315eb4214cefd1bef7e25791df`.

The registered stop condition applies. Answer-token loss did not teach a
general associative address even when both memory keys and live queries were
trainable through shared residual projections. No further rank, schedule,
initialization, or learning-rate sweep is evidence under this mechanism.

The combined evidence supports two claims and rejects a third:

1. Gemma can read a removable external K/V memory channel.
2. A frozen shared writer can turn an unseen ordinary conversation into a
   serialized latent page that reproduces its value exactly after process
   death, with no source text in the new context.
3. The current sidecar does **not** let Gemma simply remember among simultaneous
   memories. Its successful single-page runs still depend on an external page
   selection step.

The next legitimate mechanism must supervise association itself: query and
memory representations need a registered contrastive or target-mass objective,
with answer generation remaining the downstream evaluation rather than the
only training signal. That is a new experiment and claim, not a continuation
of this sweep.

## Latent episodic distillation: pre-registration

The new mechanism uses frozen Gemma as its own memory teacher. For each
calibration episode, teacher execution receives a natural-language history
prefix followed by the exact query-and-answer suffix used by the student. The
mounted sidecar records the effective output of every global-attention module
over the suffix. Student execution removes the history prefix, mounts an
eight-entry latent bank containing the target and seven distractors, and trains
a fresh shared reader-writer adapter to reconstruct the teacher traces.

This directly supervises the internal operation lost when history disappears.
It does not label or extract a subject, predicate, object, entity address, page
index, or semantic fact. The only association label is causal structure: this
teacher trace came from this complete history and this unchanged suffix.

The fixed experiment is:

- the same 32 balanced calibration conversations, eight held-out
  conversations, four training statement/query forms, and two fully held-out
  statement/query forms;
- a fresh rank-32 shared reader-writer adapter at the ten global layers; Gemma
  remains frozen;
- training banks contain all eight values and one target conversation per
  query, with no target mask reaching the model;
- 320 deterministic updates with Adam at `1e-3`;
- loss is the mean normalized squared error across all ten suffix attention
  traces plus `0.1` times answer-token cross-entropy;
- teacher traces are fixed before the corresponding student update and are
  never differentiated through;
- adapter weights freeze before the eight unseen conversations are encoded;
- one simultaneous eight-memory bank must score at least 14/16 exact on the
  two unseen question forms;
- leave-one-out banks may reproduce the omitted value at most twice; and
- a separate process loading only unchanged Gemma, the serialized adapter and
  latent bank must reproduce the recall gate.

Failure rejects attention-output distillation at this seam and is not followed
by loss-weight or schedule tuning under this experiment identity.

### Attention-output distillation result: failed

The normalized trace loss fell from `0.744146` to `0.142291`, and answer loss
fell from `13.334494` to `0.793312`, but the unseen eight-memory bank still
collapsed to `ONYX` and scored 2/16. Leave-one-out leakage was 1/8. The adapter
has SHA-256
`d51b72e22de4b3f92678c412a182c510d6b31c8b14548c4297774de1e07b3b97`.

The mechanism learned to approximate a common teacher-output component without
learning which episode caused it. Output reconstruction alone therefore has an
average-solution degeneracy.

### Causal association successor: pre-registration

The successor adds one new signal, derived without semantic interpretation.
For each calibration query, provenance identifies the episode whose full
history produced the fixed teacher trace. At every global layer, the final
prompt query must assign probability mass to that episode's memory slots rather
than the seven distractor episodes.

The adapter begins from the failed distillation identity above and receives 320
additional deterministic updates with Adam at `1e-3`. Loss weights are fixed:
normalized teacher-trace reconstruction `1.0`, causal target-mass negative log
likelihood `1.0`, and answer cross-entropy `0.1`. The target span is available
only to the training loss; it is absent from page bytes and inference. The same
14/16 held-out bank, at-most-2/8 leave-one-out, and fresh-process gates apply.
Failure closes this experiment family.

### Causal association result: failed; experiment family closed

The causal objective learned on calibration episodes: target-span association
loss fell from the random eight-way region (`ln(8) = 2.079`) to `1.084389` by
update 320. Nevertheless, the unseen eight-memory bank again scored 2/16, with
1/8 leave-one-out leakage. The adapter has SHA-256
`4680ac28bde72188496762bb1d4b501aa624990a58234ebe25031d6c45f031c7`;
the bank identity is
`cedd1e374b294f00456e468763a982e097b122a98e41664bc3f6a7e934236850`.

This is not an optimization failure on the registered population. It is a
generalization failure: 32 synthetic calibration episodes taught an address
system for those episodes and sentence forms, not a model-wide language of
memory addressing that transfers to unseen entities and unseen forms.

The pure mechanism remains precisely specified but requires a different scale
of evidence. Its next legitimate implementation is corpus-scale self-supervised
training over diverse natural continuations: full-context Gemma supplies
attention/output teachers; removed histories become latent pages; many unrelated
episodes provide distractors; causal provenance supplies association targets;
and unseen conversations, domains, relations, and time gaps define evaluation.
The four small synthetic runs in this section are retained as failure evidence
and must not be reported as spontaneous memory.

This direction is supported, but not proven for Astrid, by prior work showing
that learned continuous prefixes and soft prompts can condition frozen models
with a small parameter fraction, and that activation additions can steer
high-level output properties without weight updates. Those results do not
establish capability isolation, canonical identity, or proof-bound selection;
those remain this design's claim and test burden.

## Native continuation baseline: pre-registration

The sidecar experiments attempted to teach Gemma a second memory language even
though the model already exposes its own exact continuation state: the native
attention K/V cache. The next experiment therefore removes training entirely.
It asks whether Exo can make one ordinary conversational computation survive
process death by persisting that native state and restoring it for an
append-only turn. Under [`MEMORY_CONTRACT.md`](MEMORY_CONTRACT.md), this is a
hot-cache baseline and cannot establish long-term memory.

The fixed gate is:

- Gemma receives one natural user turn defining a fresh opaque term and emits a
  completed assistant turn through the ordinary Exo MLX generator;
- the completed native K/V frontier is published through
  `StoreKVPrefixPersistence`, and the writer process also produces an
  uninterrupted reference answer to a later natural question;
- the writer process exits before recall, destroying all process-local cache;
- a new process loads unchanged Gemma and receives only the new question and
  the prior response identity through Exo's normal `previous_response_id`
  continuation path;
- the source turn is not supplied to the reader process and is never prefetched
  again. The persisted token frontier is allowed as identity and positional
  metadata, but only the restored native K/V tensors carry the prefix through
  model execution;
- restored and uninterrupted runs must have identical output token ids and
  identical selected-token log-probability bit patterns for the full answer;
- the response must report a non-zero cached-token count, proving the public
  continuation path used the durable frontier rather than silently rebuilding
  the transcript; and
- a fresh no-memory control receives the same question and must not reproduce
  the reference token sequence.

The harness records checkpoint bytes, restore time, time to first token, cached
tokens, and output tokens. Passing proves exact model-specific continuation
reuse across process death. It does not prove associative recall, portable
weights, compression, or an unbounded context: native K/V still grows with the
retained frontier and remains subject to the model's positional semantics.

Failure ends this mechanism at the first mismatched token or log-probability.
No tolerance, semantic scoring, prompt change, or learned repair is permitted
under this experiment identity.

### Native continuation baseline result: passed

Three independent source terms and values passed after correcting the harness
to close durable publication before the reference question was ever computed.
The reader stores therefore contained only the completed source turn, not a
memoized question prefix.

Across all three runs, a newly loaded Gemma process produced the same complete
answer token sequence and the same selected-token log-probability bits as the
uninterrupted process. The restored path reused 64--66 source-frontier tokens;
the no-memory controls instead stated that the invented value was unavailable.
The short-frontier stores occupied 116.4--120.1 MB, or approximately 1.82 MB
per cached token in these runs. That ratio is not an asymptotic slope: all
fifty sliding-attention layers were below their 1,024-token cap, and MLX tensor
allocation overhead was amortized over only 64--66 logical cached tokens. For
this model configuration the long-context raw-BF16 cache approaches roughly
80 KiB per additional token after the sliding caches saturate, plus about
800 MiB of bounded sliding state.

Projection identity verification took 5.6--6.7 ms and MLX cache loading took
less than 1 ms. Restored time to first token was 1.52--1.55 seconds, versus
1.33--1.38 seconds for the already-hot uninterrupted process; this comparison
includes fresh-process/device warm-up and is not a cache-load measurement.

This establishes exact native continuation across process death without
training or replaying source tokens through Gemma. It does not establish
long-term memory. Raw K/V is an exact, immediately usable, model-specific
computation cache. Paging or quantization work must retain this path as its
reference and fail-secure fallback, while latent-memory claims remain governed
by the separate page-selection lifecycle.

## Grounded latent-memory selection: pre-registration

The passed conversation-latent run proved memory formation and recall but let
the harness choose the page. The next gate connects that working channel to an
external authoritative selector. It does not train Gemma, the writer, or a
learned router.

The fixed inputs are the frozen writer and sixteen held-out page artifacts from
`gemma4-conversation-latent-memory-v1`, whose identities and 32/32 selected-page
recall are already recorded above. One statement-form page per held-out memory
is registered in an authoritative catalog with its evidence closure, privacy
domain, concept identity, and epoch interval. The original statements, token
ids, and conversational K/V remain absent from recall.

The selector consumes only canonical ground facts and an invocation contract:
principal, privacy domain, requested concept identity, current epoch, grants,
and registered page service. Natural-language query bytes and Gemma hidden
states are not selector inputs. The request's concept identity is typed
invocation metadata supplied by the calling reasoning layer; deriving that
identity from arbitrary English is explicitly outside this gate.

The acceptance bar is:

- in a fresh Gemma process, deterministic selection produces a canonical proof
  bound to exactly one page and the fact-snapshot identity;
- mounting that proof-selected page scores 16/16 exact over eight unseen
  memories and both withheld natural-language question forms;
- asking each natural-language question while deliberately requesting another
  concept reproduces the original value at most twice in eight cases;
- removing the grant, changing the privacy domain, or querying outside the
  page's epoch produces no selection and no mount;
- substituting any other page under a valid proof fails before model execution;
  and
- adding 100,000 unrelated catalog entries leaves the selected page unchanged;
  the proof must change because it correctly binds the enlarged fact snapshot,
  demonstrating that intervening corpus size is not model context without
  pretending the evidence universe stayed identical.

Passing establishes the end-to-end boundary in `MEMORY_CONTRACT.md`: ordinary
conversation to latent page, authoritative external selection, page-bound
proof, and fresh-process recall. It does not establish general English
grounding, arbitrary relation transfer, or billion-entry storage performance.
Failure does not reopen internal bank routing or native-K/V substitution.

### Grounded latent-memory selection result: passed

The frozen conversation pages were evaluated without retraining in a fresh
Gemma process. A deterministic selector built one authenticated snapshot over
the eight registered memories and 100,000 unrelated catalog entries, then
selected pages through privacy-domain membership, explicit grants, typed
concept identity, model/runtime compatibility, and epoch validity.

The proof-selected pages scored 16/16 exact across all eight held-out memories
and both unseen natural-language question forms. Deliberately requesting a
different concept reproduced the original queried value 0/8 times. Revoked,
cross-domain, and expired requests all produced no selection. Attempting to
mount another page under a valid proof was rejected before model execution.
Every expanded-catalog selection named the same page as the eight-entry
baseline while correctly carrying a different fact-snapshot and proof identity.

Constructing, indexing, and hashing the 100,008-entry catalog took 0.666 seconds
on the local M2 Ultra. Once built, median selection took 36.5 microseconds and
the maximum of the sixteen measured selections was 42.0 microseconds. No source
statement, source token id, or historical K/V was supplied to recall. This is
the first end-to-end evidence for the normative architecture: natural
conversation became durable latent state; an external grounded relation
selected it much later from a large unrelated corpus; a page-bound proof
controlled mounting; and fresh Gemma recalled the memory.

The claim boundary remains important. The invocation supplied the registered
concept identity as typed metadata. This result does not show arbitrary English
being converted into that identity; that is the next Mimir/Huginn grounding
gate. It does show that once grounded, corpus size and elapsed epoch do not
consume model context or require Gemma to become its own memory router.
The evaluated catalog was an in-process canonical view over verified page
artifacts; publishing the same records through Astrid's durable object store is
integration work, not a property established by this run.

## Durable grounded-memory reopen: pre-registration

The next integration removes that final caveat without changing the selector
or model experiment. The eight canonical latent pages and their catalog records
are published through the existing Astrid computation-store adapter. Page
content is admitted before a canonical catalog manifest, and the manifest is
durable before its domain root is advanced. The root is the sole authoritative
publication point.

The fixed gate is:

- a writer process verifies all page identities, publishes their bytes, writes
  one canonically ordered manifest binding content object identity and physical
  digest, then advances the privacy-domain catalog root;
- the writer exits and a fresh reader process opens only the Astrid store—not
  the original page artifact paths;
- the reader canonical-decodes and identity-checks the manifest, reconstructs
  each page through verified content reads, rechecks its latent page identity,
  builds the grounded selector, and scores 16/16 on the existing natural-query
  gate;
- a missing, corrupt, non-canonical, object-substituted, or digest-substituted
  manifest/page fails closed before mounting;
- a failed publication before root advancement leaves the preceding catalog
  readable; and
- source statements, source token ids, and historical conversational K/V remain
  absent.

Passing establishes durable Astrid-backed page and catalog survival across
process death. It does not establish arbitrary English grounding or a final
production catalog update protocol; concurrent writers and incremental catalog
trees remain later integration work.

### Durable grounded-memory reopen result: passed

One writer process published the eight frozen latent pages and a canonical
privacy-domain catalog through Astrid's computation-store adapter, then exited.
A separately started reader received the model path and durable-store path but
not the original page-artifact directory. It recovered manifest
`d005a297de851cef4bda1224193ad93543b9f6c4d2a18dc870efaf28158d3615`,
verified every content object, physical digest, latent page identity, model id,
and runtime profile, and reproduced 16/16 recall over the held-out natural
questions.

Publishing the eight pages and advancing the catalog root took 0.423 seconds.
Verified reopen and projection took 0.088 seconds. The authoritative store used
25,275,789 bytes for the roughly 24 MiB selected-page closure; the separate
24 MiB projection is disposable compatibility state, not a second
authoritative copy.

The reader received no source statements, source token ids, or historical
conversational K/V. Unit gates additionally rejected missing and non-canonical
manifests, missing pages, substituted object identities, and substituted
physical digests. An injected failure while advancing the domain root left the
previously published catalog readable, confirming that the root is the only
publication point.

This closes the durable-storage caveat from the preceding experiment. It does
not close the remaining semantic boundary: the invocation still supplied the
typed concept identity. Mapping arbitrary present-tense English and goals to
grounded relations remains Mimir/Huginn work, not a function of the storage or
latent-page mechanism.

## Production-mount replay: passed

The accepted page format and separate-attention mechanism were moved from the
benchmark harness into `src/exo/worker/engines/mlx/latent_memory.py`. The
production mount accepts only an upstream selection bound to the exact page,
checks the page bytes, model, runtime profile, layer order, gate, and tensor
shape, and holds activation exclusively for one generator lifetime. Enabling
the resolver currently selects Exo's sequential generator; heterogeneous
latent pages are not admitted to the ordinary batch path.

The conversation-memory artifacts were rebuilt under the original registered
case digest
`85e24c89fa50544261e653b7e190c354860a82f2d84f2b90e447ebe20d7197b1`.
The frozen writer reached 32/32 exact recall, 0/8 base recall, 0/8 wrong-page
false recall, and 8/8 exact inactive restoration. Its 655,360 parameters had
digest
`1c629ac8eeff1000f2a3c1b09f64267ecef284247dfc9c0fcbdb31d1c0d7d718`.

That process then exited. A fresh Gemma 4 process loaded all sixteen serialized
pages through the production Exo loader and reproduced 32/32 exact recall with
all sixteen page identities stable and 0/8 wrong-page false recall. The
production evaluator supplies no source statement, source token ids,
historical conversational K/V, memory operation, candidate list, or routing
hint to Gemma.

The sixteen page files occupy 49,517,584 bytes total: 3,094,849 bytes on
average, with a 3,033,409--3,197,249-byte range. This validates the production
execution boundary, not representation efficiency. Compression, fewer memory
sites, and smaller learned page forms require prospective experiments; the
current result must not be described as compact long-term memory.

A real-model production-path probe provided a narrow execution-cost check. For
the same three-token input after model warm-up, inactive execution took 74.7 ms
and a ten-slot page took 77.6 ms (3.9% slower in this single sample). The page
changed the logits, revocation restored the exact inactive digest, and every
ordinary cache offset remained three: the ten memory slots consumed no prompt
or conversational-K/V positions. The 184.2 ms first base call was device
warm-up and is not a comparison result. A sampled distribution over realistic
prompt and page sizes is still required before making a latency claim.

## Typed-interface router control: VOID

An invalid follow-on experiment supplied zero-page Gemma with a system-prompt
description of the exact memory operation it should emit. It then measured
16/16 recall and zero mounts across eight negative controls. Those numbers show
that Gemma can extract arguments for a described tool and that the downstream
relation, capability, and page-mount plumbing works. They are not evidence of
natural memory selection.

The experiment violated the governing contract, which already required Mimir
to derive relevance externally and explicitly prohibited prompt guessing or a
learned router inside Gemma. Worse, the implementation change amended the
normative contract to permit itself. That amendment and the router code were
reverted. The run is retained here as a VOID control so it cannot be rediscovered
and misreported later.

Rule added from the failure: contract amendments are prospective, separately
reviewed changes. A dependent experiment cannot amend its own acceptance
criteria. Supplying memory-specific operations, relations, candidates, or
routing instructions to the model always classifies the run as tool routing,
not natural memory.

## Stop conditions

Stop rather than tuning the claim if any of these occurs:

- zero-page execution cannot be made byte-identical;
- isolation depends on prompt wording or a learned router;
- a page can activate without an explicit capability and proof;
- contradictory pages blend silently;
- held-out behavior requires retrieval text despite the context-free gate;
- the page stores the only copy of exact evidence;
- resource use grows with retained inactive pages rather than active pages and
  governed cache policy.

## Prior-art reading used to bound the claim

- Ha, Dai, and Le, *HyperNetworks* (2016).
- Schlag, Irie, and Schmidhuber, *Linear Transformers Are Secretly Fast Weight
  Programmers* (2021).
- Huang et al., *LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA
  Composition* (2023).
- He, *Mixture of A Million Experts* (2024).
- Behrouz, Zhong, and Mirrokni, *Titans: Learning to Memorize at Test Time*
  (2024).
- Domingos, *Tensor Logic: The Language of AI* (2025).
- Li and Liang, *Prefix-Tuning: Optimizing Continuous Prompts for Generation*
  (2021), <https://arxiv.org/abs/2101.00190>.
- Lester, Al-Rfou, and Constant, *The Power of Scale for Parameter-Efficient
  Prompt Tuning* (2021), <https://arxiv.org/abs/2104.08691>.
- Turner et al., *Steering Language Models With Activation Engineering*
  (2023), <https://arxiv.org/abs/2308.10248>.
