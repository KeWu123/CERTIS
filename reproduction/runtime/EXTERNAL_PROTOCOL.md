# Locked external join study protocol

Recorded after three fully preserved exploratory DBpedia pilots and before
this study's implementation/execution. These are constructed joins on real
external text/model tapes, not native SemBench workloads or live inference.
The source and complete data checks are in `external_inputs/validation.json`.
No query is selected using its oracle outcomes. All 32 workloads below and
all declared seeds/arms must be retained, including empty answers and loss.

## Fixed targets and queries

For each of DBpedia and GoEmotions, sort the 5,000 unique released row_ids.
The first 2,500 rows form L and the last 2,500 form R. A node identity is
(predicate_id, row_id, pinned oracle-tape version); disjoint row domains
ensure the two sides never denote the same variable. Ordinary typed/tagged
inner join derivations are materialized with their actual bag multiplicity.

DBpedia pairs, selected from English predicate meanings without consulting
outcomes: (agent,person), (person,athlete), (organization,company),
(place,natural_place), (work,musical_work), (event,societal_event),
(athlete,sports_team), (company,educational_institution).

GoEmotions pairs: (sent_positive,joy), (sent_negative,anger),
(sent_ambiguous,confusion), (is_question,curiosity),
(mentions_person,admiration), (sent_negative,sadness),
(sarcastic,annoyance), (sent_positive,love).

Each pair has two joins. DBpedia single-key is l1; multi-key joins on
three type-incidence levels, with key=(level,value) for l1/l2/l3.
GoEmotions single-key is the lexicographically first supplied annotation
(a designated empty-annotation key if none); multi-key uses every supplied
annotation. These annotations are existing input metadata, not the fixed
LLM oracle's truth, and no method may treat them as oracle labels.
Use distinct incidences per identity/key. A pair sharing several keys
contributes once for every shared key. Null type values do not match.

Initial primary cache is empty. The three predeclared sample seeds are
101,202,303. Use lambda=1/10 and per-query failure budget delta=1/20.
This is a per-query probability, not a claim that all independent trial
audits succeed jointly with probability .95. Reusing one covered region
record supports all later masks/allowances for that same fixed key system.

## Direct certificate arms (D)

Draw an SRS without replacement of n=min(budget,side target size) per
side, with budgets 125 and 500, fixed before the new draw's labels. All
random audited identities are charged, including audit labels irrelevant
to the final mask. Regions are the identity sets for each nonempty side/key;
merge identical regions before allocating delta across their intervals.
Condition on the realized number of draws in a region and use exact
hypergeometric inversion for its positive total. The exact binary numeric
alpha executed must never exceed the allocated nominal rational budget.
Cache facts then refine original full-region intervals without treating
deterministic execution as additional random sampling.

Primary D arm: key envelope plus truthful feedback, batch 128. Controls:
same-source static initial certificate; deterministic point-only key
envelope with zero audit; and pure potential-degree versus helper ranking.
Overlapping-key mode is explicitly a safe relaxation; single-key mode is
Boolean-sharp under the stated interval-only evidence. Preserve each key's
negative signed maximum; clipping it to zero is an implementation error.

The global mask order is ascending r_v=s_v*sum_opposite_incident s_u,
with one contribution per shared key and stable identity tie-breaking.
This product is a ranking surrogate, not an independence assumption for
coverage. At 65 declared cardinalities floor(N*t/64), t=0,...,64, form
nested base masks, rescuing every known positive. Execute descending r_v
in batches of 128. The degree-only control sets s=1 and uses no helper.
Within a retained subgraph, unknowns without any possible positive retained
partner need no evaluation. It is permissible to use this exact semijoin
reduction, but final materialization must still include every positive
retained derivation. A frozen inconsistency path never advances its mask;
keep its status and actual output in the results.

Pre-execution clarification (no study cells executed): the static certificate
arm buys every initially retained unknown identity R0, in its fixed order,
even when subsequent zero labels would permit additional semijoin pruning.
This is the exact reference for the feedback request-set containment theorem.
Dynamic semijoin controls remain separate competitive executors; no pathwise
dominance over them is asserted. The feedback arm may use the semijoin
reduction above. Both arms share the initial audited facts and mask.

Additional matched-quality control fixed before the matrix: require every
key separately to meet the same 0.9 recall target, using its exact local
count-polygon maximum. Use the identical paid audit, ranking, mask family,
feedback and semijoin executor. The maximum of local signed supports must
be nonpositive, rather than their sum. This valid uniform per-key allocation
cannot exploit safe retained mass on other keys. It isolates the value of
one output-bag target without changing evidence or oracle units. It is a
simple certificate comparator, not a reproduction of any external system.

## Structural-cut alternatives (C)

Use the exact factorized minimum-unit-price cut on the same input metadata
and actual identity namespaces. No cost based on unseen oracle output may
influence cut selection. After paid structural acquisition, form exact K/w.
Compare zero-audit point-fact residual certification, and a fresh residual
count audit with fixed budget 125 or 500 per remaining predicate/side.
Sampling from the then-unknown population is a new random audit conditional
on the prior truthful history, with its own complete delta ledger; add
known counts back to recover full-region inequalities. Do not shuffle
deterministic cut labels and pretend they were random samples.
Use disjoint four score bins (fixed label-blind before the new audit), exact
intervals, largest certified nested mask and batch 128 feedback. Include
all labels purchased for the cut and new audit. These arms are alternative
certification records, not an uncharged best-of-methods certificate.

## Exact and simple competitive controls

Run no-audit/no-helper exact identity scan with zero-fan-out elimination;
left-first and right-first exact short-circuit execution with function cache
and semijoin reduction; minimum structural cut followed by exact residual
execution (classical COVER); and helper-guided adaptive exact greedy with
dynamic possible-partner elimination. Report each, rather than only an
oracle-selected minimum. Their retrospective per-case minimum is allowed
only as a labeled lower-envelope diagnostic, not an executable strategy.
All execute against the same pinned truth and actually materialize answers.

All D/C confidence-source ablations must share metadata, score access,
allowance, batching and initial cache whenever their contracts permit it.
Different structural/source policies are compared as complete alternatives,
not claimed to isolate a single component. Source-specific failures remain.

## Costs, uncertainty and retained evidence

The primary endpoint is distinct oracle requests for complete query
execution, including audit, cut and residual union. The harness logs every
requested identity and returned label by phase. Also report taped input
and output tokens separately. Instrument score reads or conservatively
charge all relevant helper inputs used for global ranking; exact controls
without scores pay none. Report existing-helper and purchased-helper
cost sensitivity over helper/oracle token-cost ratios 0,.01,.05,.1,1;
these are modeled prices, not measured cloud fees. No new training is run.
The upstream helper selection cost is not known and cannot be declared
zero from scratch; label this shared pretrained-signal assumption.

Measure local preprocessing, audit inversion, graph/cut, certificate
projection, execution bookkeeping and SQL answer materialization separately
on a shared host. Never sum taped per-batch latencies as live end-to-end
time. Record warmup and repeat timing on fixed representative settings
selected by predicate names/metadata, not winning outcomes. All paired
cases are dependent fixed-target measurements. Report full distributions,
per-workload/seed values and adverse cases; do not invent confidence
intervals treating dependent query cells as independent populations.

An independent full-tape DuckDB evaluation must verify returned bag subset,
exact full/returned bag cardinalities, affine decomposition when applicable,
and actual lost fraction. Counts alone are not output materialization:
create the returned derivation table or an equivalently verified relation.
Save query SQL, source hashes, parameters, regions, interval budgets, ledgers,
runtime failures and output checks. New-code changes require regression
against identity enumeration/reference arithmetic before broader execution.

Pilot gate: first run every one of the 32 workloads at seed 101 and budget
125 with all controls. This is a predeclared feasibility/correctness gate,
not an opportunity to drop adverse queries. Then complete seeds 202/303 and
budget 500, retaining the first run. If a true execution/resource limitation
occurs, record it and finish the feasible scientifically relevant range;
do not label an unexecuted arm complete.
