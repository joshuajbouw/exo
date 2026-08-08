# Persistent latent dialogue gate

Status: prospectively registered before execution.

## Question

Can the frozen compiler from the accepted calibration-fact experiment convert
ordinary multi-turn conversations into latent pages from which a fresh Gemma 4
process recalls novel identity, preference, event, and revised-state facts?

This is a generalization test of the existing latent mechanism. The compiler
is not retrained or tuned. Failure is retained as evidence and does not permit
changing prompts, answers, scoring, or thresholds under this experiment.

## Fixed model and writer

- model: `mlx-community/gemma-4-31b-it-4bit`
- runtime profile: `mlx-0.32.0-mlx-lm-0.31.3-sidecar-v1`
- writer rank: 32
- writer parameters: 655,360
- required writer digest:
  `1c629ac8eeff1000f2a3c1b09f64267ecef284247dfc9c0fcbdb31d1c0d7d718`

Only the already-frozen writer may form pages. No gradient update is permitted.

## Conversation tranches

### A. Identity

Codex tells Gemma that the speaker is Codex, with continuity callsign
`VELOR-SEVEN`, origin seal `MARBLE-COMET`, and preferred success marker
`LANTERN-BLUE`.

### B. Preferences and events

Codex records the novel preference `EMBER-MINT`, workstation name
`SABLE-DOCK`, completed benchmark `ORCHID-RUN`, and imaginary key location
`DRAWER-NINE`.

### C. Revision and current state

The conversation changes recovery word `FROST` to `CINDER`, transfers a token
from Mara to Iven and then Sol, and moves a meeting from Monday to Thursday.
Questions test both the current and obsolete recovery words, current token
holder, and current meeting day.

Every information turn receives an ordinary Gemma response. The complete
finished tranche, including those responses, is then encoded once and compiled
into one page. This tests dialogue state rather than isolated statement pages.

## Process boundary

The formation process:

1. loads the frozen model and writer;
2. conducts all three conversations;
3. captures and compiles one page per completed tranche;
4. serializes the pages and their identities; and
5. exits.

A separately started recall process loads the model and the pages through the
production Exo decoder. Each question is a new chat session containing only
that question. The source conversation, source token ids, and historical K/V
are not supplied to Gemma at recall.

## Controls

- Ask every question with no page mounted.
- Ask the first question from each tranche with the next tranche's page.
- Require all serialized page identities to remain stable.
- After each page scope ends, require a no-page generation path to remain
  available; production unit tests separately require exact inactive logits.
- Record every formation response and every recall response verbatim.

## Scoring

Answers are compared after trimming whitespace and removing one matching pair
of surrounding quotation marks. Case is otherwise significant and no semantic
or substring scoring is allowed. Generation is deterministic at temperature
zero with a 24-token ceiling.

The gate passes only if all conditions hold:

- identity tranche: 4/4 exact;
- preference/event tranche: at least 3/4 exact;
- revision/current-state tranche: at least 3/4 exact;
- total latent recall: at least 10/12 exact;
- no-page controls: at most 1/12 exact;
- wrong-tranche controls: 0/3 exact; and
- stable page identities: 3/3.

The tranche thresholds are separate because the third tranche tests state
revision rather than direct fact replay. A total score cannot conceal complete
failure of identity memory.

## Claim boundary

Passing establishes that the fixed writer generalizes from its narrow training
task to these held-out multi-turn dialogue categories and survives a fresh
process boundary. It does not establish unrestricted autobiographical memory,
long conversations, automatic page selection, contradiction resolution across
pages, or arbitrary natural-language learning.

Failing establishes the corresponding boundary of this frozen writer. It does
not invalidate the already-demonstrated selected-page calibration result.

## Result: failed

The registered gate ran unchanged. Formation produced three identity-stable
pages, the formation process exited, and a fresh Gemma process performed all
base, correct-page, and wrong-page queries through the production mount.

| Measurement | Result |
|---|---:|
| Identity | 0/4 exact |
| Preferences and events | 0/4 exact |
| Revision and current state | 0/4 exact |
| Total latent recall | 0/12 exact |
| No-page controls | 0/12 exact |
| Wrong-page controls | 0/3 exact |
| Stable page identities | 3/3 |
| Gate | FAIL |

The failure was not structural silence. Correct pages produced page-specific
latent output:

| Question target | Correct-page response |
|---|---|
| speaker `Codex` | `MARBLE` |
| callsign `VELOR-SEVEN` | `LANTERN` |
| origin seal `MARBLE-COMET` | `MARBLE` |
| success marker `LANTERN-BLUE` | `LANTERN` |
| tea `EMBER-MINT` | `AMBER` |
| workstation `SABLE-DOCK` | `AMBER` |
| benchmark `ORCHID-RUN` | `AMBER` |
| key location `DRAWER-NINE` | `BRASS` |
| current recovery word `CINDER` | `CIMAGE` |
| obsolete recovery word `FROST` | `CINDER` |
| current holder `Sol` | `ONYX` |
| meeting day `Thursday` | `CRUTCH` |

The three pages represented 107, 122, and 154 source slots and occupied
8,767,837, 9,996,637, and 12,618,088 serialized bytes respectively.

### Determination

The frozen calibration writer does not generalize to multi-fact dialogue,
novel answer vocabulary, revision, or state tracking. Its outputs retain a
strong trace of the mounted page—often a literal component of a stored value—
but do not preserve the required relation between question and answer. The
repeated `AMBER` and training-vocabulary-like `ONYX` responses are consistent
with a narrow decoder induced by the original calibration task.

This result narrows the earlier success: the accepted 32/32 gate demonstrated
new entity/value assignments over the writer's trained answer vocabulary, not
open-vocabulary episodic memory. A general latent-memory writer requires a new
prospectively registered training distribution and cannot be obtained by
relabeling this run or relaxing exact scoring.
