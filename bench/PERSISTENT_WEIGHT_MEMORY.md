# Persistent weight memory experiment

This experiment asks a narrow question: can a model learn a novel rule, lose
its teaching context and KV cache, restart in another process, and still apply
the rule without damaging unrelated behavior?

The synthetic Kelthara calculus is absent from the base model. Its terms and
rules are generated deterministically. Training, validation, test, and stress
sets separate identifiers, graph depths, and prompt forms. The strict gate is
75% correct decisions and 75% byte-exact canonical witnesses on a fresh
process. Unrelated deterministic prompts are a collateral-behavior control.

## Results

All runs use `mlx-community/gemma-4-31b-it-4bit`. The untouched model scored
0/32 because it had no Kelthara behavior and emitted no valid decision.

| Run | What changed | Adapter | Decisions | Exact witnesses | Result |
| --- | --- | ---: | ---: | ---: | --- |
| Surface-controlled pilot | 8 layers, rank 8 | 31.25 MiB | 26/32 | 11/32 | Persists behavior, but the corpus still exposed lexical shortcuts |
| Relation-only, full path | Same adapter surface | 31.25 MiB | 8/32 | 0/32 | Chance; exact graph reconstruction is not learned |
| Relation-only, compact witness | Same adapter surface | 31.25 MiB | 15/32 | 1/32 | Fails the reasoning gate |
| Wider compact witness | 16 layers, rank 16 | 124.67 MiB | 13/32 | 8/32 | More capacity does not fix free generation |
| Six training phrasings | Wider adapter; seventh/eighth forms held out | 124.67 MiB | 11/32 | 8/32 | Prompt diversity does not produce the operator |

The strongest pilot loaded in 2.12 seconds in a fresh process. Its
`adapters.safetensors` digest is
`db53482f3a7c7e23d0ec763e1d27a6038dc29c1bf37d3e1eed3c11eae11a3ace`.
Two independent reloads produced byte-identical result arrays with digest
`0eaa9e553ba4dc5803eb6340c79e3094d50bb6c25dd6955d8043e024ca858121`.
The later relation-only runs deliberately invalidate the pilot as evidence of
transitive reasoning.

Attenuating the pilot did not expose a safe global operating point:

| LoRA scale | In-domain decisions | Unrelated behavior |
| ---: | ---: | --- |
| 20 | 26/32 | 8/8 prompts overwritten by Kelthara output |
| 10 | 22/32 | Heavily contaminated |
| 5 | 14/32 | Still contaminated |

Explicit scoping is cheap enough to be the safety mechanism rather than a
numeric compromise. Across five fresh processes, attaching the 31.25 MiB
adapter to an already constructed model took 7.46-8.02 ms. MLX materializes
some work lazily, so a separate one-token probe included the first use: median
prompt-plus-token time was 0.476 seconds on the base and 0.511 seconds with the
expert active, about 7.5% overhead. These are local M2 Ultra microbenchmarks,
not full-generation throughput claims.

Teacher-forced validation loss was not a sufficient gate. One wider run
reached 0.030 validation loss and then scored only 13/32 under free generation.
Production evaluation must therefore cross a fresh-process generation
boundary. Training also became unstable after its best checkpoint in two
runs, so validation-selected early stopping is mandatory.

## Determination

Ordinary LoRA is not a general persistent-memory primitive. It can compile a
narrow behavior into a small model-specific delta, but a globally active delta
can overwrite unrelated behavior, and neither additional capacity nor prompt
diversity made it a reliable relation engine. Exact events, identities, and
proofs must never live only in probabilistic weights.

The useful architecture is narrower:

1. Base model weights remain immutable.
2. Verified events and facts remain authoritative content-addressed objects.
3. Mimir computes exact relation witnesses and decisions; it does not ask the
   model to reproduce graph paths.
4. A trained adapter is a disposable, model-specific `Derived` object that may
   provide domain language or procedural bias only.
5. Huginn selects and mounts such an expert explicitly for the relevant domain
   and assembles the exact supporting objects into the turn's bounded context.
6. The expert is absent outside that scope. This is the isolation boundary
   that the scale sweep could not obtain numerically.
7. After the turn, KV state may be discarded. The corpus, proofs, and optional
   expert persist independently and can be reconstructed for another model.

The follow-up contract and its pre-model algebra gates are in
[`NEURAL_PARAMETER_PAGES.md`](NEURAL_PARAMETER_PAGES.md). It replaces prompt-
or learned-router scoping with a capability-scoped Tensor Logic activation
proof and treats conflicting adapter composition as an error by default.

This preserves the desired context-scrubbing property without pretending that
weights are an exact database. The next inference experiment should test
explicitly gated expert activation and composition, not another globally
active LoRA sweep.

## Reproduction

`persistent_weight_memory.py prepare` freezes the primary corpus and hashes
every artifact. `prepare-stress` creates the 128-case depth-6-through-12 set.
`generate-evaluate` records model, adapter, hashes, timings, generated text,
and gates in JSON. `control-evaluate` records unrelated behavior. The LoRA
configuration used by the final capacity test is in
`persistent-weight-memory-lora.yaml`.
