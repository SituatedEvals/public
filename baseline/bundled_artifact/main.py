"""Reference submission: use an artifact you bundled inside the ZIP.

There is no network at runtime and nothing is pre-fetched on your behalf, so
anything your code needs has to be in the ZIP:

    submission.zip/
      main.py
      requirements.txt        optional -- packages the runtime image lacks
      artifacts/
        prior_weights.csv     whatever you fitted offline

The artifact here is a CSV of prior pseudo-counts, one row per (item, option),
which is about the smallest thing that is still genuinely fitted rather than
decorative. **It could just as well be a model** -- `model.joblib`,
`weights.safetensors`, a small transformer -- and nothing below would change
except the two marked lines: load it at import, consult it per cell. The ZIP is
capped at 1 GB and the machine has 16 GB of RAM, so a fitted statistical model,
a lookup table or a small transformer all fit; a large language model does not.

Load it at **module import**, not inside predict(). Module-level code runs once
while the container starts; predict() is on the clock, and it is called once per
instrument, so per-call work is multiplied by three.

The prior is applied only to items the artifact knows about, and the crowd
marginal carries every other item. That fallback is what lets one submission
face all three instruments: an artifact fitted on one of them does not have to
say anything about the others.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "artifacts" / "prior_weights.csv"

# How much the prior is worth, in units of respondents. The visible answers are
# real counts, so a prior of 20 is worth 20 people saying so, and the data wins
# as soon as an item has more than that.
PRIOR_STRENGTH = 20.0


def _load_prior(path):
    """Read the bundled artifact into {item: {option: weight}}.

    ── Swap this for your own artifact ──────────────────────────────────────
    joblib.load(path), torch.load(path, map_location="cpu"),
    safetensors.torch.load_file(path) -- all equivalent here. The only contract
    is that predict() can turn the result into one number per option.
    """
    if not path.exists():
        return {}
    prior = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            weight = float(row["weight"])
            if weight < 0:
                raise ValueError("prior weights must be non-negative")
            prior.setdefault(row["item"], {})[row["option"]] = weight
    return prior


# ── Module-level init: runs once, off the clock ─────────────────────────────
PRIOR = _load_prior(ARTIFACT)
print("[bundled_artifact] %s: %d item(s) with a prior"
      % (ARTIFACT.name if PRIOR else "no artifact bundled", len(PRIOR)),
      flush=True)


def _options(schema, item):
    """The item's answer space, in the order the grader reads your vector."""
    record = schema["items"][item]
    values = list(record["values"])
    if record.get("gate"):
        # Being never asked is a real answer, and its slot goes last.
        values.append(schema["gated_value"])
    return values


def predict(frame, schema):
    """One probability vector per blank cell, in canonical order.

    Crowd marginal plus the bundled prior: for each item, count the answers
    that are visible and add the artifact's pseudo-counts before normalizing.
    An item the artifact does not mention keeps the half-count smoothing, so
    the submission is never worse off for a missing row.
    """
    items = [name for name, record in schema["items"].items()
             if record["class"] in ("GIVEN", "PREDICT")]
    options = {item: _options(schema, item) for item in items}

    marginals = {}
    for item in items:
        counts = frame[item].value_counts()
        # Half a count on every option, so an option nobody chose is unlikely
        # rather than impossible. A zero here would cost you the run.
        weights = np.array([counts.get(option, 0) + 0.5
                            for option in options[item]], float)

        # ── Where the artifact earns its place ───────────────────────────────
        # A model would be consulted here instead, per respondent rather than
        # per item, and would return one number per option either way.
        prior = PRIOR.get(item)
        if prior:
            mass = np.array([prior.get(option, 0.0)
                             for option in options[item]], float)
            if mass.sum() > 0:
                weights = weights + PRIOR_STRENGTH * mass / mass.sum()

        marginals[item] = weights / weights.sum()

    values = frame[items].to_numpy(dtype=object)
    return [marginals[items[column]]
            for row in range(values.shape[0])
            for column in range(len(items))
            if pd.isna(values[row, column])]
