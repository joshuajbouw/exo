# Mimir memory grounding gate

Status: prospective design record. No implementation is authorized by this
file. It must be reviewed and accepted independently before dependent code or
measurement begins.

## Question

Can an external deterministic reasoner derive the typed memory relation and
subject requested by an ordinary current utterance, then select a durable
latent page, without placing memory-specific instructions, schemas, candidates,
or historical content in the model prompt?

This is the unresolved boundary in `MEMORY_CONTRACT.md`. Durable page formation,
storage, capability selection, mounting, and fresh-process recall already have
separate evidence. This gate tests only utterance-to-typed-goal grounding and
its composition with those proven mechanisms.

## Claim boundary

Passing would establish one learned, receipt-grounded query family in a closed
memory microworld. It would not establish arbitrary English understanding,
open ontology induction, transfer to unregistered relations, or that Tensor
Logic grants authority.

The language evidence is supervised by the world: training utterances are
paired with independently issued typed goal receipts. That is an explicit
given. The learner must discover the lexical and structural mapping from those
receipts; it may not be handed a parser, runtime keyword table, regex, query
template, or model-generated semantic label.

## Fixed architecture

1. Gemma receives only the ordinary current conversation. Its prompt is
   byte-identical whether durable memory exists or not.
2. Mimir receives the current utterance bytes, invocation identity, principal
   namespace, current epoch, and authenticated training artifacts.
3. A frozen tokenizer emits exact spans. The principal namespace supplies the
   finite typed subject census; recognizing an exact opaque subject identifier
   is a namespace lookup, not semantic inference.
4. Relation lexicalization and query structure are learned from TRAIN-only
   utterance/goal-receipt pairs. Statistics may nominate a mapping; only the
   frozen null, MDL, completeness, and held-out systematicity gates license it.
5. Runtime parsing consumes the complete utterance. It emits exactly one
   `(privacy domain, relation, subject)` or structural silence. Ambiguous,
   partial, negated, quoted, attributed, malformed, or unsupported parses are
   silent.
6. The emitted tuple is only a proposal. Mimir joins it against the current
   typed relation view; the capability selector independently enforces domain,
   grant, epoch, model, and runtime. Tensor Logic proposes and proves the join;
   it never grants access.
7. Huginn mounts the selected page before Gemma answers the unchanged ordinary
   invocation. Historical text, source token ids, historical K/V, catalog
   entries, and relation descriptions never enter the prompt.

The implementation must stand alone. It may reuse the jar's proved lessons and
test shapes, but it must not import the mutable jar store or treat jar-derived
artifacts as authority.

## Frozen evidence shape

Each TRAIN observation is canonical and contains:

`{utterance_id, utterance_sha256, exact_utf8_bytes, principal_domain,
subject_id, typed_goal_receipt_id, relation_id, source_epoch}`

The goal receipt is issued independently of the language learner and binds the
relation and subject. It contains no page identity, concept identity, answer
value, or parser hint. TRAIN/EVAL assignment is fixed by source identity before
mining. Exact duplicate bytes cannot cross the split.

The learned artifact contains the complete TRAIN manifest identity, tokenizer
identity, subject-census identity, lexical witnesses, licensed structural
productions, null measurements, MDL costs, and implementation identity. It is
immutable during EVAL.

Every accepted runtime parse emits a canonical proof binding:

`{invocation_id, utterance_sha256, learned_artifact_id,
subject_census_id, relation_id, subject_id, parse_tree,
relation_snapshot_id}`

The selection proof remains separate and binds the eventual page. An audit must
be able to replay both proofs without Gemma.

## Learning constraints

- Candidate relation tokens arise only from TRAIN co-occurrence with typed goal
  receipts and must survive a relation-label permutation null.
- Structural candidates arise only from TRAIN token/span evidence. No authored
  surface form or evaluation sentence may become a production.
- A licensed production must reconstruct the complete accepted byte span and
  bind exactly one subject slot and one relation. Unmatched content means
  silence.
- Operator screens may only reject. They cannot provide relation identity.
- The exact subject census may recognize an opaque identifier, but adjacency to
  that identifier does not establish a relation.
- All candidates are priced against a keyword/bag baseline and a token-order
  permutation null. A mechanism indistinguishable from bag matching does not
  pass.
- Evaluation failures may not add vocabulary, operators, productions, or
  thresholds. Any such change creates a new gate version and new split.

## Corpus and split

Use the existing 32 NEMORA training subjects and eight SOVARA evaluation
subjects. Add a second typed relation with its own independently receipted
training utterances so a one-relation default cannot pass. Both relations share
surface vocabulary and subject shapes.

TRAIN contains multiple independently witnessed surface families per relation
and subject. EVAL holds out complete surface families and subject/family
combinations. The eight SOVARA calibration values and latent pages remain
hidden from the language learner.

The corpus manifest, exact utterance bytes, typed receipts, tokenizer, subject
census, relation census, split, candidate family, null seeds, and thresholds
must be sealed before the learner runs.

## Acceptance bars

All bars are conjunctive:

1. **Exact grounding:** all sixteen existing held-out SOVARA calibration
   questions produce the correct relation and subject.
2. **End-to-end recall:** those sixteen proofs select the existing durable pages
   and preserve 16/16 exact fresh-process recall with Gemma's ordinary prompt.
3. **Relation discrimination:** every held-out question for the second relation
   selects that relation and never a calibration page.
4. **Structural silence:** wrong predicates, unknown subjects, missing subjects,
   explicit negation, quotation/use-mention, attribution, meta-language,
   malformed UTF-8, multiple subjects, and multiple clauses produce no page
   mount. Zero false acceptances.
5. **No bag shortcut:** adversarial controls preserving the positive token
   multiset while changing scope, role, or clause structure are rejected; the
   keyword/bag baseline must false-accept at least one of these controls while
   the licensed parser accepts none.
6. **Systematicity:** at least one held-out surface family and every SOVARA
   subject/family combination are absent from TRAIN. Success must come from
   licensed structure, not whole-utterance replay.
7. **Determinism:** two fresh processes over identical bytes produce
   byte-identical learned artifacts and grounding proofs.
8. **Authority separation:** removing the principal membership or page grant
   leaves grounding unchanged but yields no selection or mount.
9. **Prompt identity:** instrumented prefill proves Gemma's prompt bytes are
   identical with memory enabled, memory disabled, and no matching memory.

## Stop conditions

The line stops rather than weakening its claim if:

- implementation contains a relation-specific word branch, regex, authored
  query template, or model call;
- any model prompt receives a memory interface, relation description, candidate,
  catalog entry, or historical content;
- EVAL bytes or outcomes influence learning or thresholds;
- a partial parse can select memory;
- the keyword/bag baseline is not structurally separated by the controls;
- a proof cannot be replayed from canonical artifacts; or
- unsupported language is coerced into the nearest registered relation instead
  of producing silence.

Failure does not weaken the durable-memory result. It identifies the next
missing language primitive and returns to design before implementation.
