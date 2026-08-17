#!/usr/bin/env python3
"""Build the tutorial notebook in each official UN language.

    python tutorials/build.py

One source, six notebooks. The code is identical in all of them -- same
identifiers, same schema values, same numbers -- and only the prose, the
comments and the printed labels change. Anything that is a value the harness
reads (`TRAIN`, `NA_GATED`, item names, option strings) stays in English,
because it is data rather than language.

To change the tutorial, change TEMPLATE below and then the six blocks in
translations.yml. To add a language, copy the `en` block and translate it: the
builder checks that every key is present, so a missed one is an error rather
than a silently English paragraph.

Notebooks are written without outputs. Execute them to fill those in:

    pip install jupyter nbclient
    python -m nbclient tutorials/en.ipynb --output tutorials/en.ipynb
"""

import json
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSLATIONS = os.path.join(HERE, "translations.yml")

# Official languages of the United Nations, by ISO 639-1 code.
LANGUAGES = {"ar": "Arabic", "zh": "Chinese", "en": "English",
             "fr": "French", "ru": "Russian", "es": "Spanish"}


# --------------------------------------------------------------- template --
#
# `@@key@@` is substituted from the language pack. Everything else is shared.

TEMPLATE = [
    ("md", "@@title@@"),
    ("code", '''
import os
import sys
from pathlib import Path

# @@c_root@@
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "make_sandbox.py").exists()), None)
if ROOT is None:
    raise RuntimeError("@@c_root_error@@")
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# @@c_modules@@
from make_sandbox import (ROLE_COLUMN, generated_items, load_config,
                          load_schema, options_for, write_sandbox)
from score import floored, load_frames, sample_rows
from score import score as grade

SEED = 0
PHASE = 1
config = load_config("config.yml")

pd.set_option("display.width", 200, "display.max_columns", 50)
'''),
    ("md", "@@part1@@"),
    ("code", '''
sample = load_schema("data/sample.json", config)
write_sandbox(sample, config, "_sandbox/sample", seed=SEED)

respondents = load_frames("_sandbox/sample", sample)
print(sample["dataset"]["description"])
print()

MEANING = {"TRAIN": "@@role_train@@",
           "DEV": "@@role_dev@@",
           "TEST": "@@role_test@@"}
counts = respondents[ROLE_COLUMN].value_counts()
print(pd.DataFrame({"@@col_respondents@@": counts,
                    "@@col_meaning@@": [MEANING[r] for r in counts.index]}).to_string())
'''),
    ("md", "@@schema_declares@@"),
    ("code", '''
rows = []
for name, rec in sample["items"].items():
    gate = rec.get("gate") or {}
    rows.append({"@@col_item@@": name,
                 "@@col_class@@": rec["class"],
                 "K": len(options_for(sample, name)) if rec["values"] else 0,
                 "@@col_gate@@": gate.get("parent", "-"),
                 "@@col_asked_if@@": ", ".join(gate.get("observed_if", [])) or "-",
                 "@@col_options@@": " | ".join(map(str, rec["values"] or ["-"]))})
print(pd.DataFrame(rows).to_string(index=False))
'''),
    ("md", "@@k_and_gates@@"),
    ("code", '''
chain = ["visited_clinic", "clinic_wait", "would_return"]
print(respondents[chain].value_counts().to_frame("@@col_respondents@@").head(12).to_string())
'''),
    ("md", "@@handed@@"),
    ("code", '''
frame, cells, truth = sample_rows(sample, respondents, PHASE)
shown = [n for n, r in sample["items"].items() if r["class"] != "EXCLUDE"]

print("@@lbl_frame@@", frame.shape, " @@lbl_cells@@", len(cells))
print()
print(pd.concat([frame[["respondent_id"] + shown].head(3),
                 frame[["respondent_id"] + shown].tail(3)]).to_string(index=False))
'''),
    ("md", "@@canonical@@"),
    ("code", '''
print(pd.DataFrame(cells, columns=["@@col_row@@", "respondent_id", "@@col_item@@"]).head(8)
      .to_string(index=False))
'''),
    ("md", "@@score_intro@@"),
    ("code", '''
def hidden_cells(frame, items):
    \'\'\'@@doc_hidden_cells@@\'\'\'
    values = frame[items].to_numpy(dtype=object)
    ids = frame["respondent_id"].to_numpy(dtype=object)
    return [(row, ids[row], items[col])
            for row in range(values.shape[0])
            for col in range(len(items))
            if pd.isna(values[row, col])]


def crowd_for(sch, frame):
    items = generated_items(sch)
    tables = {}
    for item in items:
        counts = frame[item].value_counts()
        n = np.array([counts.get(o, 0) for o in options_for(sch, item)], float)
        tables[item] = (n + 0.5) / (n + 0.5).sum()
    return [tables[item] for _, _, item in hidden_cells(frame, items)]


vectors = floored(crowd_for(sample, frame), sample, cells, config["scoring"]["floor"])
result = grade(sample, config, vectors, truth, cells)

def table(rows):
    """Print label/value pairs, aligned however long the labels happen to be."""
    pad = max(len(label) for label, _ in rows)
    for label, value in rows:
        print("%-*s  %s" % (pad, label, value))


table([("@@lbl_uniform@@", "%.4f" % result["uniform_reference"]),
       ("@@lbl_logscore@@", "%.4f" % result["log_score"]),
       ("@@lbl_skill@@", "%.4f" % result["skill"])])
print()
print("@@lbl_skill_note@@")
'''),
    ("md", "@@part2@@"),
    ("code", '''
# @@c_estimand@@
TARGET, POSITIVE = "trusts_health_advice", ["Somewhat", "A lot"]
GIVEN = [n for n, r in sample["items"].items() if r["class"] == "GIVEN"]

Y = respondents[TARGET].isin(POSITIVE).to_numpy(float)
design = pd.get_dummies(respondents[GIVEN].astype(str), drop_first=True)
X = np.column_stack([np.ones(len(design)), design.to_numpy(float)])

# @@c_model_fit@@
past = (respondents[ROLE_COLUMN] == "TRAIN").to_numpy()
ridge = np.linalg.solve(X[past].T @ X[past] + 5 * np.eye(X.shape[1]),
                        X[past].T @ Y[past])
predicted = X @ ridge          # @@c_predicted@@

# @@c_frame_rows@@
frame_rows = np.flatnonzero(~past)
TRUTH = Y[frame_rows].mean()   # @@c_truth@@

table([("@@lbl_fitted_on@@", "%d" % past.sum()),
       ("@@lbl_frame_size@@", "%d" % len(frame_rows)),
       ("@@lbl_corr@@", "%.2f"
        % np.corrcoef(predicted[frame_rows], Y[frame_rows])[0, 1]),
       ("@@lbl_true_share@@", "%.3f" % TRUTH)])
'''),
    ("md", "@@now_draw@@"),
    ("code", '''
def estimates(f, interviewed, rest):
    \'\'\'@@doc_estimates@@\'\'\'
    y = Y[interviewed]
    classical = (y.mean(), y.std(ddof=1) / np.sqrt(len(y)))

    correction = f[interviewed].mean() - y.mean()
    powered = (f[rest].mean() - correction,
               np.sqrt(f[rest].var(ddof=1) / len(rest)
                       + (f[interviewed] - y).var(ddof=1) / len(interviewed)))
    return classical, powered


def band(estimate):
    point, se = estimate
    return "%.3f  [%.3f, %.3f]  @@lbl_width@@ %.3f" % (
        point, point - 1.96 * se, point + 1.96 * se, 2 * 1.96 * se)


N_INTERVIEWS = 300
draw = np.random.default_rng(SEED).permutation(frame_rows)
interviewed, rest = draw[:N_INTERVIEWS], draw[N_INTERVIEWS:]

classical, powered = estimates(predicted, interviewed, rest)
table([("@@lbl_truth@@", "%.3f" % TRUTH),
       ("@@lbl_interview_only@@", band(classical)),
       ("@@lbl_powered@@", band(powered)),
       ("@@lbl_model_only@@", "%.3f  @@lbl_no_interval@@"
        % predicted[frame_rows].mean())])
'''),
    ("md", "@@one_draw@@"),
    ("code", '''
def repeat(f, n_interviews=N_INTERVIEWS, draws=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        shuffled = rng.permutation(frame_rows)
        classical, powered = estimates(f, shuffled[:n_interviews],
                                       shuffled[n_interviews:])
        out.append(classical + powered)
    return np.array(out)          # @@c_repeat_return@@


def summarize(trials, label):
    rows = []
    for name, point, se in (("@@lbl_interview_only@@", trials[:, 0], trials[:, 1]),
                            ("@@lbl_powered@@", trials[:, 2], trials[:, 3])):
        rows.append({"@@col_method@@": name,
                     "@@col_width@@": (2 * 1.96 * se).mean(),
                     "@@col_covers@@": np.mean(np.abs(point - TRUTH) <= 1.96 * se)})
    out = pd.DataFrame(rows)
    print(label)
    print(out.to_string(index=False, float_format="%.3f"))
    return out


good = summarize(repeat(predicted), "@@lbl_good_model@@")
narrower = 1 - good.loc[1, "@@col_width@@"] / good.loc[0, "@@col_width@@"]
print("\\n@@lbl_narrower@@" % (100 * narrower, N_INTERVIEWS))
print("@@lbl_equivalent@@" % round(N_INTERVIEWS / (1 - narrower) ** 2))
'''),
    ("md", "@@both_cover@@"),
    ("code", '''
from make_sandbox import make_sandbox

elsewhere = make_sandbox(sample, config, seed=99)      # @@c_elsewhere@@
other_design = pd.get_dummies(elsewhere[GIVEN].astype(str), drop_first=True)
other_X = np.column_stack([np.ones(len(other_design)), other_design.to_numpy(float)])
other_Y = elsewhere[TARGET].isin(POSITIVE).to_numpy(float)

wrong = np.linalg.solve(other_X.T @ other_X + 5 * np.eye(other_X.shape[1]),
                        other_X.T @ other_Y)
mispredicted = X @ wrong

table([("@@lbl_corr@@", "%.2f"
        % np.corrcoef(mispredicted[frame_rows], Y[frame_rows])[0, 1]),
       ("@@lbl_model_only@@", "%.3f   @@lbl_vs_truth@@ %.3f   <- @@lbl_off_by@@ %+.3f"
        % (mispredicted[frame_rows].mean(), TRUTH,
           mispredicted[frame_rows].mean() - TRUTH))])
print()
summarize(repeat(mispredicted), "@@lbl_bad_model@@")
'''),
    ("md", "@@read_together@@"),
]

KEYS = sorted(set(re.findall(r"@@(\w+)@@", "".join(t for _, t in TEMPLATE))))


def _existing(path):
    """The notebook already on disk, or []."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)["cells"]
    except Exception:
        return []


def _hand_edited(existing, cells):
    """Has someone edited the notebook since it was last built?

    Compares cell sources only, ignoring outputs, which change on every run.
    Worth the check: a notebook is an obvious thing to edit in place, and a
    rebuild would overwrite that silently and unrecoverably.
    """
    return bool(existing) and (
        [c["source"] for c in existing] != [c["source"] for c in cells])


def _keep_outputs(existing, cells):
    """Carry results across a rebuild wherever the source did not change.

    Executing a notebook is minutes of work and the outputs are part of what a
    reader gets; a rebuild that silently emptied them would be its own kind of
    overwrite. A cell whose source moved keeps no stale result. The cell id
    comes across too, so that rebuilding an unchanged notebook is a no-op on
    disk rather than a diff of regenerated identifiers.
    """
    for old, new in zip(existing, cells):
        if old.get("source") != new["source"]:
            continue
        if "id" in old:
            new["id"] = old["id"]
        if new["cell_type"] == "code":
            new["outputs"] = old.get("outputs", [])
            new["execution_count"] = old.get("execution_count")
    return cells


def build(code, pack, force=False):
    missing = [key for key in KEYS if key not in pack]
    if missing:
        raise SystemExit("%s is missing %d key(s): %s"
                         % (code, len(missing), ", ".join(missing)))

    cells = []
    for kind, text in TEMPLATE:
        body = re.sub(r"@@(\w+)@@", lambda m: pack[m.group(1)], text).strip("\n")
        cell = {"cell_type": "markdown" if kind == "md" else "code",
                "metadata": {},
                "source": body.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3",
                                    "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4, "nbformat_minor": 5}

    path = os.path.join(HERE, code + ".ipynb")
    existing = _existing(path)
    if _hand_edited(existing, cells) and not force:
        raise SystemExit(
            "%s has been edited since it was last built, and rebuilding would\n"
            "discard those edits. Move the edit into translations.py (or into\n"
            "TEMPLATE, if it belongs in every language), or pass --force to\n"
            "overwrite." % os.path.relpath(path))
    notebook["cells"] = _keep_outputs(existing, cells)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(notebook, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    return path


def main():
    with open(TRANSLATIONS, encoding="utf-8") as fh:
        packs = yaml.safe_load(fh)

    force = "--force" in sys.argv[1:]
    unknown = set(packs) - set(LANGUAGES)
    if unknown:
        raise SystemExit("not a UN language code: %s" % ", ".join(sorted(unknown)))
    for code in sorted(LANGUAGES):
        if code not in packs:
            raise SystemExit("no translation for %s (%s)" % (code, LANGUAGES[code]))
        print("%-3s %-8s -> %s"
              % (code, LANGUAGES[code],
                 os.path.relpath(build(code, packs[code], force=force))))


if __name__ == "__main__":
    main()
