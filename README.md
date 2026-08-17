# SimulacraBench

This README is the submission documentation of the 
[SimulacraBench competition](https://www.codabench.org/profiles/organization/4076/). The participation contract is contained under the Terms tab of that page. Submission of a model is conditional on consent with these terms.

> **This repository is public and holds no microdata.** The schemas in `data/`
> describe settings, questions answer options only, and declares prediction targets only.

## Quickstart

```
pip install -r requirements.txt
python make_sandbox.py --schema data/sample.json --out _sandbox/sample
python score.py --data _sandbox/sample --schema data/sample.json --phase 1
```

`score.py` scores a submission against a frame produced from data contained in the directory given in `--data`. The shape of the frame consists of some full respondents, and some masked respondents. Each missing answer is graded separately using the [proper](https://www.wikiwand.com/en/Scoring_rule) log score. It also contains a `--schema` file, which defines how a submission (described below) will be interpreted in as an answer.

You do not have the real data, so `make_sandbox.py` writes a stand-in of the same shape based on the `--schema` to `--out`. This includes skip logig, types, and missingness.

To inspect the mechanics on a small dataset, the above quickstart for `data/sample.json` produces a small, 400-respondent sample.

For submission, copy `baseline/marginal_counts` into a new directory, change
`predict()` in its `main.py`, add dependencies to its `requirements.txt`, and add files that should be run at runtime, and list in `models.txt` Huggingface model identifiers. Then compress:
```bash
cd my_submission && zip -r ../my_submission.zip .
```
and upload the resulting `.zip` file. Before uploading, check that the archive
itself is well-formed — this validates the `.zip`, not your score:

```bash
python tools/check_submission_zip.py my_submission.zip
```

If you fitted something offline, copy `baseline/bundled_artifact` instead: it
loads weights from a file bundled in the `.zip`, which is the mechanism a model
would use.

The code in this repo includes containerization for transparency, but is not required for local development. It is documented in [Running under Docker](#running-under-docker).

`tutorials/en.ipynb` runs entirely on `data/sample.json`. It walks through the
shape of the task — schema, frame, gating, canonical order, scoring — and then
works through what a good model is actually for: making a survey estimate more
precise without replacing the survey. To use it, run:

```
pip install -r requirements.txt jupyter
jupyter notebook tutorials/en.ipynb
```

The same notebook is available in each official language of the United Nations —
`ar`, `zh`, `en`, `fr`, `ru`, `es`. All six are built from one source by
`tutorials/build.py`, so the code is identical and only the prose, the comments
and the printed labels differ; to change the tutorial, edit `tutorials/build.py`
and `tutorials/translations.yml` and rebuild rather than editing a notebook.

---

## Task

Probabilistic completion of a **respondent × question** grid. Each instrument is
one wide table: `respondent_id` plus one column per question. Every respondent's
`GIVEN` block is visible. A number of respondents given in the `--schema` arrives complete. The rest are **held out**. You see their `GIVEN` block and nothing else; every
`PREDICT` cell is blank, and every blank is scored using the log scoring rule. 

For each blank you return a **probability distribution over that question's
option list** — not a guess at the answer. Every scored answer is categorical,
drawn from that question's own options. The metric is a strictly proper scoring
rule, so your best expected score comes from reporting what you actually
believe.

The three datasets are documented in the schemas contained in `/data`. The leaderboard gives per-Dataset and the mean of *skills*:

```math
\mathrm{skill} = 1 + \frac{\mathcal{L}}{U}
\qquad\qquad
\mathrm{Skill} = \tfrac{1}{3}\left(\mathrm{skill}_{\text{UNICEF}} + \mathrm{skill}_{\text{World Bank}} + \mathrm{skill}_{\text{UNHCR}}\right)
```

where the log score and the uniform reference are

```math
\mathcal{L} = \frac{1}{|C|} \sum_{(i,j)\in C} \log p_{ij} \left[ y_{ij} \right]
\qquad\qquad
U = \frac{1}{|S|} \sum_{j \in S} \log K_j
```

and the vector that is actually scored is yours, renormalized and mixed with a
flat one:

```math
p_{ij} = \varepsilon + \left(1 - K_j\,\varepsilon\right)\frac{q_{ij}}{\sum_{k} q_{ij,k}},
\qquad \varepsilon = 10^{-3}
```

$S$ is the instrument's scored items, $K_j$ the option count of item $j$, $C$ the blank cells, $q_{ij}$ the vector you returned and $y_{ij}$ the answer that was actually there. So **0 is a uniform guess and 1 is perfection**; the full account is under [Scoring](#scoring).

A `predict()` does not need to perform well on all datasets, but must produce legal outputs (a probability distribution per output) for all of them.

### Phases

There are two phases: a development and a test phase. They are distinguished by the frame given, the compute allocated (for module import, model loading, per compute call), and the returned values. These are documented in `config.yml`, which is the authority. 

| | Phase 1 — Development | Phase 2 — Final |
|---|---|---|
| Visible, answers included | `TRAIN` | `TRAIN` and `DEV` |
| `GIVEN` only, `PREDICT` masked and scored | `DEV` | `TEST` |
| Wall-clock Compute Budget | 900 s (`phases.1.timeout_seconds`) | 3600 s |
| Submissions | 1 per day | 1 |
| Score returned | Laplace-noised, rounded to `phases.1.round_to` (0.01) | exact |

Every respondent carries one of three roles — `TRAIN`, `DEV` or `TEST` — in the
`role` column of the delivered file, assigned once when the dataset is built and
never redrawn. How many respondents carry each is declared in the schema's
`split` block, as counts. Nothing is subsampled in either phase: a phase takes
every row of every role it is entitled to.

The phases therefore do not nest: `TEST` respondents are not shipped at all in
phase 1, so a phase-1 leaderboard probed all through development is not an
answer key for phase 2, and `DEV` respondents return in phase 2 as visible rows,
answers included, which is where they are worth most.

Because the scored set is fixed within a phase, every submission is scored on
the same cells and two leaderboard entries are directly comparable — the
sampling variability they share cancels in the difference between them.

The budget covers the **whole run**: module import, model loading, and all three
`predict()` calls together.

---

## Submission

All submissions are **code submissions**. You upload a `.zip`; the platform runs
your code against the phase's hidden slice and writes the predictions on your
behalf.

```text
submission.zip/
  main.py               # required, at the `.zip` root
  requirements.txt      # pinned dependencies, installed before the run
  models.txt            # Huggingface identifier, e.g., `google-bert/bert-base-cased`
  any_other_files/      # weights, lookup tables, fitted state
```

Build the archive from **inside** your submission directory, so `main.py` sits
at the archive root rather than inside a folder:

```bash
cd my_submission && zip -r ../my_submission.zip .
```

Do not upload a `.zip` that contains another `.zip` — the entry module has to be a
real file at the root. Archives with absolute or parent-escaping paths, or with
implausible file counts or compression ratios, are rejected. The `.zip` is
capped at 1 GB. `requirements.txt` and `models.txt` are optional, hosted and
under `score.py` alike: the image already carries numpy, pandas, torch,
transformers and the rest, and `requirements.txt` exists only to add what the
image does not have.

```python
def predict(frame, schema):
    ...
    return vectors      # list[list[float]]
```

`predict()` is called **once per instrument** — not once per cell — with two
positional arguments, and returns one probability vector per blank cell.

There is no runtime training hook, and training must happen **offline**.
Module-level code runs once when the container starts, before `predict()` is
called, so that is where weights, tokenizers, lookup tables and fitted state
should be loaded — not inside `predict()`, which is on the clock. An import or
setup failure there fails the submission before any predictions are made.

### `frame`

A wide DataFrame: `respondent_id` plus one column per shipped item, in schema
key order. Visible respondents come complete; held-out respondents have every
`PREDICT` cell `NaN`. Only `GIVEN` and `PREDICT` items are shipped — `EXCLUDE`
records stay in the schema and never appear as columns, so filter on `class`
rather than assuming the two line up.

**`NaN` means exactly one thing: this cell is held out, predict it.** It never
means "they did not answer". Genuine non-response is an ordinary level —
`Prefer not to answer`, `98. I don't know`, `99. Refused to answer` — and
`load_schema` rejects a schema with a null in an option list. Every non-`NaN` cell is a real
answer you need to assign a probability to.

### `schema`

The file in `data/`, plus `schema["gated_value"]` filled in from `config.yml`.

| Key | Holds |
|---|---|
| `dataset` | `n_rows`, `version`, and a prose `description` of the instrument |
| `items` | one record per item, in the grader's order |
| `split` | `n_train`, `n_dev` and `n_test`, counts of respondents summing to `dataset.n_rows` |
| `gated_value` | the level meaning "this person was never asked" (`NA_GATED`) |

Each `items` record holds four keys:

| Key | Holds |
|---|---|
| `question` | the wording, as asked |
| `class` | `GIVEN`, `PREDICT` or `EXCLUDE` |
| `values` | the allowed answers, never null |
| `gate` | `{parent, observed_if}`; null or absent when the item is always asked |

| Class | Meaning |
|---|---|
| `GIVEN` | Always visible, for everybody. Never scored. |
| `PREDICT` | Held out and scored. What the competition is about. |
| `EXCLUDE` | Identifiers, record keys, free text, admin fields. Never shown, never scored. |

### Option order

Your vector follows `schema["items"][item]["values"]` in order, **plus a final
slot for `schema["gated_value"]` for the probability this question being gated.**

Read it from the schema, never from the data — an option nobody chose still has a
slot. Skip-logic gating is not missingness: being never asked is a real answer,
scored like any other, so predicting who gets skipped is worth as much as
predicting what they say. `gate` tells you which earlier answer decides it, and
a gated item's answer is determined whenever its parent is visible.

### Question order

Rows top to bottom, and within a row, items in `schema["items"]` key order —
**not** `frame.columns` order, which may differ.

### Return value

A list of lists of floats, one vector per blank cell, each as long as
that item's option list with the gate sentinel included. Ingestion validates the
return tup before anything is written, and applies exactly these rules:

| Rule | Failure |
|---|---|
| The return value is a `list` or `tuple` | a wrong type fails the submission |
| One vector per blank cell | a wrong count fails the submission |
| Each vector is numeric and finite and as wide as that item's option list, sentinel included | a wrong width fails the submission, in canonical order |
| Every entry is finite and non-negative | `NaN`, infinity, or negative numbers fail the submission |
| Each vector sums to more than zero | an all-zero vector fails the submission |

You do not need to floor or normalize: the grader renormalizes every vector and
mixes it with a flat vector before scoring. An exception raised inside
`predict()` fails the submission, as does a failure while importing your module.

### What `predict()` may and may not do

Read whatever you bundled — weights, lookup tables, fitted state — from inside
your own directory, and import whatever the image provides or your
`requirements.txt` and `models.txt` declare. No other downloads. The grader
removes the network before your code is imported, and will raise an exception, failing the submission.

---

## What you can build

The task is open-ended. Fit statistical or psychometric models (IRT, low-rank completion, tabular generative models) on the schema and your own practice data; build features from the `GIVEN` block and the item descriptions.

The two obvious levers over the crowd-marginal baseline are the skip logic,
which determines a gated item's answer whenever its parent is visible, and
whatever the `GIVEN` block tells you about a respondent you have never seen.

The organizers provide `torch_measure` in the runtime image for latent-trait /
IRT-style modeling of survey responses. Use it only if it helps your approach.

---

## The hosted runtime

Every submission runs on the same hardware. There is no routing, no tier
selection and no way to request different hardware — a `gpu:` line in `metadata`
is ignored. Resource exhaustion fails a submission:

| | |
|---|---|
| GPU | 1 × H100 |
| Memory | 16 GB |
| CPU | 8 cores |
| Wall-clock budget | 900 s in Development, 3600 s in Final |
| Network | none |
| Python | 3.13, with pre-installs `numpy  pandas  pyarrow  scipy  scikit-learn
torch  torchvision  Pillow
transformers  sentence-transformers  tokenizers  sentencepiece  tiktoken
huggingface_hub  accelerate  safetensors  bitsandbytes  autoawq  protobuf
torch_measure` |

Memory is a hard limit, not a target: exceeding it terminates the run. The data
itself is small — about 50 MB for all three instruments — so the budget is there
for your model. The wall-clock budget covers all three instruments together, not
900 s each. Each submission runs in a fresh container that is destroyed
afterwards, so module-level state does not persist between submissions.

`requirements.txt` is installed while the runtime image is built, before your
container exists and before your clock starts, so the install does not spend
your run budget — but it has its own ceiling, and exceeding it fails the
submission with `HSCC-BUILD-002`. Normal named pip requirements only: avoid pip
options, editable installs and source-build-only packages, and **pin exact
versions**, since an unpinned requirement makes pip search many candidates.

### Bringing a model

There is no network access at runtime, so nothing can be downloaded while your
code runs, and nothing is pre-fetched for you. Either bundle weights directly into your submission (`.pt`, `.pth`, `.safetensors`, `.bin`, `.ckpt`,
`.pkl`, `.joblib`, `.npy` are all accepted, for example), below 1GB. You may also use models hosted on HuggingFace, by including their identifier (e.g., `google-bert/bert-base-cased`) in `models.txt`.

---

## What happens when you submit

1. The `.zip` is validated for archive safety and layout, and your `main.py` is
   statically checked for referenced files that are missing from the `.zip`.
2. Any packages in `requirements.txt` are installed while the runtime image is
   built, before your container exists and before your clock starts, and Huggingface models are loaded with `for repo in lines:
    p = snapshot_download(repo, cache_dir=os.environ["HF_HUB_CACHE"])`
3. The orchestrator materializes the phase's hidden slice for each instrument —
   the frame with held-out cells blanked, the schema, and the canonical cell
   order. The answer key is not among them and never enters the container.
4. Your container starts, network-isolated, and imports `main.py` once.
5. For each instrument in turn: `predict(frame, schema)` is called once, the
   returned vectors are validated, and they are written out aligned to the
   canonical cell order.
6. The orchestrator scores each instrument, applies
   the phase's privacy mechanism, and posts the result to the leaderboard.

As the data is airgapped, we do not provide you with `stdout` or tracebacks. We only provide you with codes to localize the error.

A diagnostic may carry the failure phase, a sanitized exception type, the
exception text and a line number and frame context **only for load/import
diagnostics**, your file's basename, output count and type facts, safe
dependency names for load/import failures, timeout and resource facts, and
approximate progress counts. Hidden-runtime diagnostics drop to the basename
alone, such as `main.py`. They never include raw tracebacks, submitted source
line text, absolute paths, hidden item IDs, hidden item text, labels, URLs,
tokens, or hidden-derived runtime names.

```text
[HSCC-DEPS-001] Missing package: your code tried to import a module that is not installed.
Detail: ModuleNotFoundError: No module named 'missing_pkg'
Participant frames: main.py:1 in <module>
Facts: missing module: missing_pkg.

[HSCC-PREDICT-001] Runtime error in predict(): your predict() function raised KeyError at main.py.
Participant file: main.py

[HSCC-PREDICT-002] Invalid predict() output: predict() must return one probability vector per blank cell.
Participant file: main.py
Facts: vector count: returned 1, expected 144 (one per blank cell).
```

Note what the second one does **not** say. Your exception's message is dropped —
only its type survives — because a message can carry hidden data the moment your
code interpolates a value into it. Same reason there is no traceback and no line
number inside `predict()`. Debug locally, where you get all three.

| Code family | What it means | What to fix |
|---|---|---|
| `HSCC-ZIP-*` | The uploaded `.zip` layout is wrong or unsafe (as deemed by our code analysis). | Put `main.py` at the `.zip` root; do not upload a folder-wrapped `.zip` or a `.zip` containing another `.zip`. |
| `HSCC-ARTIFACT-001` | Your code referenced a local file that was not bundled. | Add the named file, such as `ncf_head.pt` or `features.npy`, to the `.zip` or update the path in your code. |
| `HSCC-IMPORT-*` / `HSCC-DEPS-*` | `main.py` could not load. | Fix imports, syntax, missing packages, or module-level setup; rerun `tools/check_submission_zip.py`. |
| `HSCC-HF-CACHE` | Your code tried to download model files at runtime. | Bundle the weights in the `.zip` and load them from a local path. Nothing is pre-fetched for you. |
| `HSCC-BUILD-001` | The hosted runtime image could not be built from your dependency choices. | Simplify `requirements.txt`, remove unsupported packages or pins, or use pre-installed packages. |
| `HSCC-BUILD-002` | The dependency install exceeded its own time ceiling. | Pin exact versions so pip does not search many candidates. |
| `HSCC-NETWORK-*` | Runtime code tried to make a blocked third-party network call. | Bundle what you need in the `.zip`; do not fetch internet resources inside `predict()`. |
| `HSCC-PREDICT-*` | `predict()` raised, or returned the wrong number of vectors, the wrong width, or non-finite / negative / all-zero values. | Return one vector per blank cell in canonical order, each as wide as that item's option list plus the gate slot; test on all three schemas. |
| `HSCC-SCORING-*` | The returned vectors could not be matched to the scored cells. | Ensure `predict()` returns a vector for every blank cell, in the frame's canonical cell order. |
| `HSCC-TIMEOUT-*` / `HSCC-CONTAINER-*` | The run timed out, exited early, or likely ran out of memory. | Move training offline, load compact artifacts at module import, and reduce per-call work. |
| `HSCC-INFRA-*` | The platform could not queue, archive, collect, or retain enough run detail. | Retry once, then start a forum post |
| `HSCC-UNKNOWN-001` | The failure did not match a safe known pattern. | Run the local tools and check `main.py`, `requirements.txt`, and bundled files before starting a forum post |

The numeric failure modes ingestion distinguishes:

| Code | Meaning |
|---:|---|
| `10` | No entry module — `main.py` was not at the `.zip` root |
| `11` | `main.py` could not be imported, or defines no callable `predict(frame, schema)` |
| `20` | Staged instrument data was missing or unreadable (organizer-side; retry, then report it) |
| `40` | `predict()` raised an exception |
| `41` | `predict()` exceeded a per-instrument cap, when one is configured |
| `42` | `predict()` returned invalid output — wrong type, wrong vector count, wrong width, non-finite, negative, or all-zero |
| `50` | The run exceeded the phase's wall-clock budget |
| `1` | Unexpected error |

`HSCC-INFRA-*` and `HSCC-UNKNOWN-001` mean the platform could not classify the
failure safely: retry once, then start a forum post.

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
probability falls below `scoring.floor` in `config.yml` while the vector still
sums to 1:

```text
p  <-  p / sum(p)
p  <-  floor + (1 - K*floor) * p          floor = 1e-3
```

Mixing rather than clipping is what keeps both promises at once, and it is what
bounds the cost of a single cell. A zero therefore costs `log(1e-3)` ≈ −6.9
rather than negative infinity.

That does not make confident wrong answers cheap. A confidently wrong cell
costs the full `log(floor)`, against about −1.4 for an honest hedge over four
options, and the whole distance between a uniform guess and a good crowd
marginal is far smaller than that.

The rule is proper: your best expected score comes from reporting what you
actually believe.

The skills across datasets are taken as a flat mean for the grand prize number.

In **phase 1** you get back your skill plus a Laplace draw, rounded to
`phases.1.round_to`. In **phase 2** you get the exact number.

## Files

| File | What it is | You edit it? |
|---|---|---|
| `data/*.json` | one schema per instrument: items, options, order | no |
| `make_sandbox.py` | writes practice data of the schema's exact shape | no |
| `score.py` | runs a submission the way the grader will, and scores it | no, run it |
| `tools/check_submission_zip.py` | validates an upload `.zip` against the contract — says nothing about your score | no, run it |
| `config.yml` | phases, time limits, privacy, sandbox knobs | no |
| `baseline/marginal_counts/` | reference submission: the crowd marginal, the thing to beat | copy it |
| `baseline/bundled_artifact/` | the same model, reading a bundled artifact — the shape to copy if you fitted something offline | copy it |
| `tutorials/{ar,zh,en,fr,ru,es}.ipynb` | the task on `data/sample.json`, then what a good model buys a survey | no, run it |
| `tutorials/build.py`, `tutorials/translations.yml` | one source for all six notebooks | only to change the tutorial |

The two baselines differ only in where their numbers come from.
`marginal_counts` reads the visible answers and nothing else;
`bundled_artifact` adds prior weights it loads from `artifacts/prior_weights.csv`
at module import, which is the mechanism a bundled model would use — the CSV is
a stand-in for a `.joblib`, `.safetensors` or `.pt` file, and only the loader
line changes.

### `make_sandbox.py`

```
python make_sandbox.py --schema data/unicef.json --out _sandbox/unicef
```

Writes two files, which together are what a delivered dataset looks like:

| File | Holds |
|---|---|
| `respondents.parquet` | every respondent, plus a `role` column |
| `schema.json` | what `predict()` receives |

Roles are assigned here, once, in the counts the schema's `split` block declares
— not in `score.py`, which only looks them up. Nothing is sampled: the first
`n_train` rows are `TRAIN`, the next `n_dev` are `DEV` and the remaining
`n_test` are `TEST`, so the file's composition is exactly what the schema says:

| Role | Phase 1 | Phase 2 |
|---|---|---|
| `TRAIN` | visible, answers included | visible, answers included |
| `DEV` | `GIVEN` only, `PREDICT` masked and scored | visible, answers included |
| `TEST` | not shipped at all | `GIVEN` only, `PREDICT` masked and scored |

Whole respondents go one way: splitting cells instead would leave a gated child
visible while its parent was hidden, which gives the parent away. Row count
comes from `dataset.n_rows`: the sandbox is the shape of the real file, not a
sample, and there is no flag to change it.

### `score.py`

```
python score.py --submission baseline/marginal_counts --data _sandbox/unicef \
                --schema data/unicef.json --phase 1
```

`--data` is a directory holding `respondents.parquet`. In order:

1. **Ingests** the file and type checks it against the schema — every declared
   column present, `respondent_id` unique, every role one of the three and
   present in the count `split` declares for it, every value one the schema
   lists. A bad dataset fails in a second rather than an hour into an H100.
2. **Selects** this phase's visible and hidden roles — all of them, nothing
   subsampled — and blanks every `PREDICT` cell of the hidden rows.
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

### `tools/check_submission_zip.py`

```
python tools/check_submission_zip.py my_submission.zip
```

**This validates the `.zip`, not your model.** It answers one question — *would
the platform accept this archive and get legal output out of it?* — and says
nothing whatever about your score. `score.py` is the tool for that, and the two
are not substitutes: a submission can pass this and score below a uniform guess,
or score well and still be rejected for a layout mistake that costs you a day's
quota.

It takes the `.zip` itself, not a directory, because the archive is what gets
uploaded and most rejections are properties of the archive. It runs the same
checks hosted ingestion runs, in the order ingestion runs them:

| Check | Catches |
|---|---|
| Archive safety and layout | absolute or parent-escaping paths, a `.zip` inside the `.zip`, `main.py` missing or nested inside a folder |
| `requirements.txt` | a file outside the root, and lines that are not plain named packages — pip options, URLs, local paths, nested requirements |
| Bundled artifacts | a literal path your code opens that is not in the `.zip` — the `HSCC-ARTIFACT-001` failure, found before it costs you a submission |
| Import | `main.py` failing to import, or defining no callable `predict(frame, schema)` |
| `predict()` | an exception, and a return value that is the wrong type, count, or width, or holds non-finite or negative entries |

The last two run `predict()` on a tiny instrument built in-process — three
respondents, one gated item — so the vectors are checked against a real option
list with a sentinel slot. That instrument carries **no signal**: passing it
means your code is well-formed, not that it predicts anything.

Prints `OK` and exits 0, or one `ERROR:` line and exits 1. Unlike a hosted run,
you get the whole message, so debug here rather than against an `HSCC-*` code.

### Running under Docker (Optional)

By default the network is cut inside the interpreter. `--docker` runs the same driver
under `docker run --network=none --read-only`, enforcing it at the kernel and
building your `requirements.txt` into a clean `python:3.13-slim` — the Python of
the hosted image. The container is capped at the worker's own limits, read from
`runner` in `config.yml`: **16 GB and 8 CPUs**. A run that fits here fits there,
and one that does not is killed here, where you can see why.

The base image is bare, though, where the hosted one arrives with torch,
transformers and the rest already installed. Anything you want under `--docker`
has to be in your `requirements.txt`, even if the hosted image would have
provided it. There is no GPU in the local container; the worker has one H100.

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
- Every import is either in the hosted image or in `requirements.txt`, pinned.
- Every weight file, lookup table and fitted artifact your code opens is inside
  the zip, loaded from a local path. Nothing is downloaded at runtime.
- You tuned on cells you hid from yourself, not on the cells you are scored on.
- `main.py` is at the **top level** of the zip, not inside a folder, and the zip
  contains no other zip. Build it with `cd my_submission && zip -r ../sub.zip .`.

Which starter to copy:

- **`baseline/marginal_counts/`** — the crowd marginal: each item's smoothed
  visible distribution, ignoring the individual respondent. This is what you
  have to beat. Start here.
- **`baseline/bundled_artifact/`** — the same model, plus prior weights read
  from a bundled CSV at module import. Copy this shape if you fitted something
  offline; the CSV stands in for a model file, and only the loader changes.

Copy one, then build the `.zip` from inside it and run both checks:

```bash
cp -R baseline/marginal_counts my_submission
(cd my_submission && zip -r ../my_submission.zip .)

python tools/check_submission_zip.py my_submission.zip    # will the platform accept it?
python score.py --submission my_submission \
                --data _sandbox/unicef --schema data/unicef.json --phase 1
```

The two answer different questions and you want both. The first validates the
archive against the contract — layout, requirements, bundled files, import,
`predict()` output — and tells you nothing about your score. The second gives
you a score, on practice data, and does not look at your `.zip` at all. Run the
second on all three instruments: a `predict()` that assumes one instrument's
shape fails on the others.

## Getting help

Use the forum on [Codabench](https://www.codabench.org/profiles/organization/4076/)