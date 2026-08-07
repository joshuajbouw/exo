# Neural parameter pages

Status: pre-registered experiment contract. This document defines what the
prototype may claim before the implementation or its results exist.

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

This direction is supported, but not proven for Astrid, by prior work showing
that learned continuous prefixes and soft prompts can condition frozen models
with a small parameter fraction, and that activation additions can steer
high-level output properties without weight updates. Those results do not
establish capability isolation, canonical identity, or proof-bound selection;
those remain this design's claim and test burden.

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
