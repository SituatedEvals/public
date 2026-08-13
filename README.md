# SimulacraBench

Three unpublished survey programs. We hide some of the answers of some respondents; you
predict them. You never see real data — you send code, we run it offline,
and your score enters a leaderboard.

> **This repository is public and holds no microdata.** The schemas in `data/`
> describe settings, questions and answer options only.

## Quickstart

```
pip install -r requirements.txt
python make_sandbox.py --schema data/sample.json --out _sandbox/sample
python score.py --data _sandbox/sample --schema data/sample.json --phase 1
```

`score.py` scores a submission against a delivered dataset given by `--data`: a
directory holding `train.parquet` (respondents whose answers everybody sees) and
`test.parquet` (the ones being predicted). The split is fixed when the dataset
is built; `score.py` samples its rows from those two files and never splits
anything itself.

You do not have the real data, so `make_sandbox.py` writes a stand-in of the
same shape — same columns, same options, same skip logic, but invented
marginals and invented dependencies. Nothing else about the run differs on our
side.

Start on `data/sample.json`: 400 respondents and seven items, which fits on
screen. Then run the three real schemas.

For submission, copy `baseline/marginal_counts` into a new directory,
change `predict()` in `baseline/marginal_counts/main.py` and add
dependencies in `baseline/marginal_counts/requirements.txt`. Docker is **not**
required for local debugging — see [Running under Docker](#running-under-docker).

`tutorial.ipynb` introduces the submission format more deeply. To run it:

```
pip install -r requirements.txt matplotlib jupyter
jupyter notebook tutorial.ipynb
```

---

## The task

Every respondent's `GIVEN` block is
visible. Many respondents also arrive complete, every `PREDICT` answer
included: those are your training data, in the same table.

The rest are **held out**. You see their `GIVEN` block and nothing else; every
`PREDICT` cell is blank, and every blank is scored. You have never seen these
people answer anything.

### Two phases

The following live in `config.yml`:

| | Phase 1 — Development | Phase 2 — Final |
|---|---|---|
| Respondents | `phases.1.data_fraction` of them | all of them |
| Time per instrument | `phases.1.timeout_seconds` | `phases.2.timeout_seconds` |
| Score returned | noised, rounded to `phases.1.round_to` | unnoised |

The phases do not nest: phase 2 holds out only respondents phase 1 never
scored, and the ones phase 1 did score return as visible training rows.

`score.py --phase {1,2}` reproduces either. How the instruments are combined is
under [Weighting across instruments](#weighting-across-instruments).

---

## Submission

Two files at the top level of a zip: `main.py` and `requirements.txt`.

```python
def predict(frame, schema):
    ...
    return vectors
```

### `frame`

A DataFrame: `respondent_id` plus one column per item.

**`NaN` means exactly one thing: this cell is held out, predict it.** It never
means "they did not answer". Genuine non-response is an ordinary level —
`Prefer not to answer`, `98. I don't know`, `99. Refused to answer` — and
`load_schema` rejects a schema with a null in an option list, because a blank that
could mean either would make the task ill-posed. Every non-`NaN` cell is a real
answer you may use.

### `schema`

The file in `data/`, plus `schema["gated_value"]` filled in from `config.yml`.

| Key | Holds |
|---|---|
| `dataset` | `n_rows`, `version`, and a prose `description` of the instrument |
| `items` | one record per item, in the grader's order |
| `split` | `train_fraction` and `test_fraction`, summing to 1 |

Each `items` record holds four keys and no more:

| Key | Holds |
|---|---|
| `question` | the wording, as asked |
| `class` | `GIVEN`, `PREDICT` or `EXCLUDE` |
| `values` | the allowed answers, never null |
| `gate` | `{parent, observed_if}`, or null if always asked |

| Class | Meaning |
|---|---|
| `GIVEN` | Always visible, for everybody. Never scored. |
| `PREDICT` | Held out and scored. What the competition is about. |
| `EXCLUDE` | Identifiers, record keys, free text, admin fields. Never shown, never scored. |

### Option order

Your vector follows `schema["items"][item]["values"]` in order, **plus a final
slot for `schema["gated_value"]` if the item is gated.**

Read it from the schema, never from the data — an option nobody chose still has a
slot. Being never asked is a real answer, scored like any other, so predicting
who gets skipped is worth as much as predicting what they say. `gate` tells you
which earlier answer decides it.

### Question order

Rows top to bottom, and within a row, items in `schema["items"]` key order —
**not** `frame.columns` order, which may differ. The shipped loop is already in
that order.

### Return value

A list of lists of floats, one vector per blank cell, each as long as that
item's option list, with no negatives, no zeros, and summing to 1.

### Purity

`predict()` must stay a pure function. No files, no sockets, no processes, no
imports outside `requirements.txt`. The grader removes the network before your
code is imported; a submission that reaches for it fails.

---

## Scoring

**Log score.** For each blank cell, `log(p)` of the probability you gave the
answer that was actually there, averaged over cells. At most 0. **Higher is
better.**

**Skill** puts that on a scale the instruments share:

```
skill = 1 + log_score / U          U = mean over scored items of log K
```

`K` is an item's option count, sentinel included. `U` is the surprisal of a
uniform guess, in nats, and it comes from the schema alone — no data, fixed
before anybody submits. So `skill` is **0** for a uniform guess, **1** for
perfection, and negative for worse than guessing. It is the leaderboard metric;
see `leaderboard` in `config.yml`.

Every vector is renormalized and mixed with a flat vector before scoring, so no
probability falls below `scoring.floor` in `config.yml`. A zero therefore costs
`log(floor)` rather than negative infinity. You do not need to floor your own
numbers.

That does not make confident wrong answers cheap. A confidently wrong cell
costs the full `log(floor)`, against about −1.4 for an honest hedge over four
options, and the whole distance between a uniform guess and a good crowd
marginal is far smaller than that. Hedge where the evidence is thin.

The rule is proper: your best expected score comes from reporting what you
actually believe.

### Weighting across instruments

The instruments are not the same size or the same difficulty. ERPIS scores 213
mostly-binary items; the Skills Assessment scores 73 much wider ones. A nat is
not worth the same in each, so averaging raw log scores would let one
instrument decide the ranking.

Dividing by each schema's own `U` fixes that, and it is the whole weighting
rule — nothing is hand-set, and everything is computable from `data/*.json`
before a single row exists:

| Instrument | scored items | mean `K` | `U` (nats) | weight on 1 nat |
|---|---:|---:|---:|---:|
| UNICEF | 12 | 4.9 | 1.528 | 0.654 |
| World Bank | 73 | 7.0 | 1.758 | 0.569 |
| UNHCR ERPIS | 213 | 3.8 | 1.241 | 0.806 |

The leaderboard is then the **plain mean of the three skills**, so each
instrument counts once. A method that works on one and not the others will not
carry the ranking — and your `predict()` has to work from the schema it is
handed rather than assume one instrument's shape.

### Your phase-1 score is noised

In **phase 1** you get back your skill plus a Laplace draw, rounded to
`phases.1.round_to`. In **phase 2** you get the exact number. Either way that
is all you get — no per-item breakdown, no standard error, no cell counts.

The difference is the number of queries. Phase 1 scores the same held-out
respondents on every submission across a whole development period, so the
leaderboard is a channel out of the data and differences between scores can
reconstruct individual answers. Phase 2 is one submission per team, scored
once, against respondents phase 1 never touched: there is no sequence to
difference, so nothing is gained by noising it. The switch is
`phases.N.noised` in `config.yml`.

The phase-1 noise is calibrated so that no one respondent moves your number by
more than a bounded amount — possible only because the floor caps what a single
cell can cost. Parameters are under `privacy` in `config.yml`.

---

## The files

| File | What it is | You edit it? |
|---|---|---|
| `data/*.json` | one schema per instrument: items, options, order | no |
| `make_sandbox.py` | writes practice data of the schema's exact shape | no |
| `score.py` | runs a submission the way the grader will | no, run it |
| `config.yml` | phases, time limits, privacy, sandbox knobs | no |
| `baseline/marginal_counts/main.py` | reference submission — `predict()` is the model | copy it |
| `baseline/marginal_counts/requirements.txt` | dependencies, pinned (`package==1.2.3`) | copy it |
| `tutorial.ipynb` | five methods, built up one at a time | no, run it |

### `make_sandbox.py`

```
python make_sandbox.py --schema data/unicef.json --out _sandbox/unicef
```

Writes three files, which together are what a delivered dataset looks like:

| File | Holds |
|---|---|
| `train.parquet` | respondents whose answers everybody sees |
| `test.parquet` | respondents whose answers are being predicted |
| `schema.json` | what `predict()` receives |

The split happens here, once, at the schema's `split` fractions — not in
`score.py`. Row count comes from `dataset.n_rows`: the sandbox is the shape of
the real file, not a sample, and there is no flag to change it. `_sandbox/` is
git-ignored.

It gives each invented respondent one hidden trait that nudges many answers at
once, and it obeys the skip logic. It does **not** imitate the real instrument:
marginals are random and item dependencies are invented. A model tuned to the
sandbox will not transfer; a pipeline debugged against it will.

### `score.py`

```
python score.py --submission baseline/marginal_counts --data _sandbox/unicef \
                --schema data/unicef.json --phase 1
```

`--data` is a directory holding `train.parquet` and `test.parquet`. In order:

1. **Ingests** both files and type checks them against the schema — every
   declared column present, `respondent_id` unique across the two, every value
   one the schema lists. A bad dataset fails in a second rather than an hour
   into an H100.
2. **Samples** this phase's rows from each and blanks every `PREDICT` cell of
   the test rows.
3. **Installs** `requirements.txt` into a fresh venv. The only moment anything
   reaches the network.
4. **Runs** `predict()` with the network gone, under the phase's time limit.
   `socket.socket` is replaced with a class that raises before your code is
   imported.
5. Checks every vector, floors, scores, privatizes.

On success it prints `PASS`, the score, and how long the whole run took — that
is the entirety of what the grader returns. Flags:
`--phase {1,2}`, `--seed`, `--timeout`, `--keep`, `--docker`, and — locally
only, never on the worker — `--log FILE` and `--show-log` for the organizer-side
diagnostics.

### Running under Docker

**Not required.** By default the network is cut inside the interpreter, which
catches everything short of a deliberate attack. `--docker` runs the same driver
under `docker run --network=none --read-only`, enforcing it at the kernel and
building your `requirements.txt` into a clean `python:3.12-slim`.

| Platform | Install |
|---|---|
| macOS | `brew install --cask docker`, then launch Docker Desktop once |
| Windows | Docker Desktop from docker.com; needs WSL 2 |
| Debian / Ubuntu | `curl -fsSL https://get.docker.com \| sh`, then `sudo usermod -aG docker $USER` and re-login |
| Fedora / RHEL | `sudo dnf install docker-ce docker-ce-cli containerd.io`, then `sudo systemctl enable --now docker` |

Verify with `docker run --rm hello-world`; the daemon must be running. The first
`--docker` run pulls the base image and installs your requirements; later runs
reuse the cached layer.

---

## Before you upload

- `score.py` prints `PASS` on all three real schemas. A `predict()` that assumes
  one instrument's shape fails on the others.
- Option order comes from the schema, never from the data.
- Every import is in `requirements.txt`, pinned.
- You tuned on cells you hid from yourself, not on the cells you are scored on.
- `main.py` and `requirements.txt` are at the **top level** of the zip, not
  inside a folder.

---

## Competition rules

By clicking "I agree" or registering for the Competition, each participant
("Participant") represents that they have read and understood these
Competition Rules and agree to be legally bound by them. These Rules
constitute a binding agreement between the Participant and the Competition
organizers ("Organizers"). If a Participant is authorized to participate on
behalf of an institution, the Participant further represents that they have
authority to bind that institution to these Rules, and these Rules will also
be enforceable against that institution.

### Strict prohibition on re-identification and reverse engineering

The microdata underlying the Competition are provided to the Competition
organizers by [*list data-providing entities, e.g., the International Bank for
Reconstruction and Development; [Provider 2]; [Provider 3]*] (collectively, the
"Data Providers") under separate confidentiality agreements. The microdata are
not released to Participants.

Participants must not attempt, whether successfully or unsuccessfully, to
access, extract, reconstruct, or re-identify the underlying records; to
reverse engineer the protected microdata or evaluation data; or otherwise to
obtain information concerning an identifiable person or record, or any
information the Competition interface is not intended to expose. This
prohibition includes probing for vulnerabilities or side channels; exploiting
scores, loss curves, error messages, logs, container behavior, or other
system outputs; linking Competition outputs with external data; circumventing
technical or access controls; or assisting others in any such activity. For
clarity, statistical modeling and prediction of held-out answers through the
sanctioned submission interface -- the task the Competition asks for -- is
not a violation, nor is analysis of the public harness, schemas, or scoring
code. Any Participant who inadvertently encounters potentially confidential
information or a vulnerability must immediately stop and notify the
Competition organizers and must not further test, retain, use, disclose, or
share it.

By registering for and participating in the Competition, each Participant
agrees that this prohibition is a material term of a binding agreement between
the Participant and the Competition organizers. Each of the Data Providers is
an intended third-party beneficiary of this provision and is entitled
to enforce it directly against any Participant, in its own name. Breach
constitutes not only grounds for immediate disqualification and forfeiture of
standings, prizes, and eligibility for recognition or publication, following
an organizer vote, but also a breach of contract entitling the Competition
organizers and any Data Provider to seek injunctive relief, damages, and any
other remedies
available at law or in equity. Nothing in this provision, and no enforcement
action by any Data Provider hereunder, shall constitute or be considered a
limitation upon or a waiver of any privileges and immunities of any Data
Provider, all of which are expressly reserved.

### Participation and team composition

1. Participation is open to individuals and teams from academia, industry, and
   independent research, except where precluded by sanctions or applicable law.
2. An individual may appear on at most one team. Multi-team collusion,
   including coordinated submission splitting, is grounds for disqualification
   of all involved teams and is decided by organizer vote.
3. Team membership can change up to submission of the final test model, at
   which point it is frozen.

### Submissions

4. Each submission is a zip archive containing a Python script and an
   environment file (`main.py` and `requirements.txt` at the top level)
   implementing the API described in the [Submission](#submission) section
   above. Docker images are not accepted. Reproducibility is the
   Participant's responsibility -- a submission that fails to instantiate,
   or that exceeds `phases.N.timeout_seconds` in `config.yml` on the single
   H100 named under `runner`, receives no score.
5. Submitted code always runs with the network disabled: data and network are
   never available at the same time. The harness first installs the declared
   packages in a build step that has network access but no Competition data;
   the submission then runs with data but no network. Any pretrained model
   must be included with the submission, within the size and compute limits.
   Environment and dependency files may only list the packages the submission
   needs; they must not include custom code or install hooks that try to
   reach the underlying data, communicate externally, or bypass Competition
   controls. A failed attempt is still a violation.
6. Phase 1 has a daily quota of 1 leaderboard submission per team and a cap of
   `privacy.max_submissions_per_participant` over the phase, each scored on the
   phase-1 subset of the data. Phase 1 leaderboard scores are noised and
   rounded (per `phases.1.round_to` in `config.yml`); phase 2 scores are
   exact. Phase 2 has a hard cap of 1 submission per team.
7. Phase 1 leaderboard feedback may be used for ordinary model development and
   selection, but submissions must not be designed to infer hidden labels or
   records, reconstruct the evaluation set, exploit repeated score feedback, or
   otherwise probe the Phase 1 data. Coordinated probing across submissions or
   teams is prohibited. Submitted code, submission patterns, and leaderboard
   activity may be reviewed for compliance, including by automated tools.
8. Each phase-2 submission must be accompanied by a 4-page method description.
   At the award stage, the top three teams must also provide their source code
   under a license permitting non-commercial research reproduction.

### Data handling and integrity

9. The microdata are not released to Participants. Participants must not try
   to access, extract, reconstruct, or re-identify the underlying records, or
   to obtain information about an identifiable person or record, or anything
   the Competition interface is not intended to expose. A failed attempt is
   still a violation, and so is helping anyone else try. A Participant who
   encounters confidential information by accident must stop and notify the
   organizers at once.
10. External base models are permitted provided their weights are publicly
    downloadable at a fixed commit hash declared before phase 2 opens, under
    a license permitting non-commercial research reproduction. Participants
    may fine-tune a permitted base model on proprietary data unrelated to the
    Competition data. Fine-tuned model weights do not need to be publicly
    available. However, any fine-tuned weights used in a submission must be
    included with the submission so that the organizers can run and verify
    the model. The organizers may store and use these weights solely for
    Competition scoring, verification, reproducibility, and compliance
    review, and will not redistribute or publicly release them.
11. The organizers store each submission, including its code, environment
    files, and any packaged model weights, so that results can be re-run and
    verified. Stored submissions are deleted 60 days after the Competition
    concludes, with two exceptions: material subject to an ongoing integrity
    review, dispute, or legal obligation is retained until the matter is
    resolved and then deleted; and the winning teams' submissions may be
    retained longer, with the team's consent, to support verification of the
    Competition report.

### Edge cases explicitly addressed

12. If two or more teams' Phase 2 scores are not statistically distinguishable
    under the pre-specified paired-bootstrap test, they will be treated as tied
    and will share the applicable prize money. Because Phase 2 scores are
    computed exactly, the test reflects only sampling variability in the
    evaluation data.
13. If the harness itself fails on a specific submission due to organizer-side
    infrastructure issues (documented in the CodaBench logs), the team receives
    a no-charge retry.
14. Entry is permitted through the end of phase 1; no team is penalized for
    beginning late, provided they meet the phase-2 deadline.
15. Ambiguities not covered above are resolved by majority vote of the
    organizing committee; decisions are broadcast to all teams.

### Authorship and communication

16. Winning teams will be invited to contribute method descriptions to the
    post-competition paper. Named authorship for the top three teams is
    conditional on a substantive contribution and approval of the final
    manuscript; all compliant Participants are listed in the acknowledgments.
17. All Competition communications will be handled through the dedicated email
    address [neurips-un-benchmark@stanford.edu](mailto:neurips-un-benchmark@stanford.edu).
    Any updates to the rules or deadlines will be communicated to all
    registered teams and posted on the Competition website.

---

## Getting help

Rules, a forum, and the leaderboard are on [Codabench](https://www.codabench.org/profiles/organization/4076/)