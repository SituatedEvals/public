"""Run a submission the way the grader will, then score it.

    python score.py --submission baseline/marginal_counts \
                    --data _sandbox --schema data/unicef.json --phase 1

Reads a delivered dataset -- respondents.parquet, roles already assigned --
takes this phase's rows, blanks what has to be predicted, installs the
submission's requirements with the network up, cuts the network, calls
predict(), and returns one noised number. See README.md.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from make_sandbox import (ROLE_COLUMN, ROLE_COUNTS, ROLES, generated_items,
                          load_config, load_schema, options_for, scored_items)

# Runs inside the prepared environment. Reads what score.py staged, calls
# predict once, writes the vectors back out. Nothing else.
#
# The network is removed in the first statements, before anything else is
# imported, so no submission code and no package a submission pulls in has a
# socket to use. This is done here rather than through sitecustomize because a
# sitecustomize in the interpreter's own stdlib silently shadows one dropped
# into a virtual environment, and a guard that quietly does not run is worse
# than no guard at all.
DRIVER = '''\
import socket


class NetworkAccessDenied(OSError):
    pass


def _denied(*args, **kwargs):
    raise NetworkAccessDenied(
        "the submission tried to use the network; scoring runs offline")


class _DeniedSocket(socket.socket):
    """Refuses to be opened.

    Stays a class rather than becoming a function because the standard library
    subclasses socket.socket -- ssl does `class SSLSocket(socket)` at import
    time -- and a function there raises a baffling TypeError instead of a
    message that tells the participant what they actually did wrong.
    """

    def __init__(self, *args, **kwargs):
        _denied()


socket.socket = _DeniedSocket

for _name in ("create_connection", "socketpair", "getaddrinfo",
              "gethostbyname", "gethostbyname_ex", "create_server"):
    if hasattr(socket, _name):
        setattr(socket, _name, _denied)

import json
import sys

import pandas as pd

work, submission = sys.argv[1], sys.argv[2]
sys.path.insert(0, submission)

with open(work + "/data.json", encoding="utf-8") as fh:
    payload = json.load(fh)
frame = pd.DataFrame(payload["data"], columns=payload["columns"])

with open(work + "/schema.json", encoding="utf-8") as fh:
    schema = json.load(fh)

import main

vectors = main.predict(frame, schema)

with open(work + "/predictions.json", "w", encoding="utf-8") as fh:
    json.dump([[float(x) for x in vector] for vector in vectors], fh)
'''


# What the driver above needs to read the staged frame, and nothing else. It is
# the floor of the container built by --docker, standing in for the much larger
# set the hosted image pre-installs.
DRIVER_PACKAGES = ("numpy", "pandas")


class SubmissionError(Exception):
    """The submission broke a rule. Reported as FAIL, never scored."""


# ------------------------------------------------------------------ ingest --

def _read(path):
    if path.endswith(".csv"):
        return pd.read_csv(path, dtype=object).astype(object)
    return pd.read_parquet(path).astype(object)


def load_frames(path, schema):
    """Read respondents.parquet and check it against the schema.

    A delivered dataset is one file carrying every respondent and the role that
    says what they are for. The roles are decided when the dataset is built,
    not here.

    The checks are the point. A column that has drifted from the schema, or a
    value nobody declared, would otherwise surface much later as an unscoreable
    cell, and by then the run has cost an hour of H100 time.
    """
    for extension in (".parquet", ".csv"):
        candidate = os.path.join(path, "respondents" + extension)
        if os.path.exists(candidate):
            frame = _read(candidate)
            break
    else:
        raise ValueError("%s holds no respondents.parquet or respondents.csv"
                         % path)

    items = generated_items(schema)
    if "respondent_id" not in frame.columns:
        raise ValueError("%s has no respondent_id column" % path)
    if ROLE_COLUMN not in frame.columns:
        raise ValueError("%s has no %s column. A delivered dataset says what "
                         "each respondent is for." % (path, ROLE_COLUMN))
    missing = [item for item in items if item not in frame.columns]
    if missing:
        raise ValueError("%s is missing %d column(s) the schema declares: %s"
                         % (path, len(missing), ", ".join(missing[:5])))

    stray_roles = set(frame[ROLE_COLUMN].unique()) - set(ROLES)
    if stray_roles:
        raise ValueError("%s holds role(s) outside %s: %s"
                         % (path, ", ".join(ROLES),
                            ", ".join(sorted(map(str, stray_roles)))))

    # The schema declares each role as a count, so the count is checkable and
    # is checked: a file that ships a different number of DEV respondents than
    # the schema says was built from a different schema, and every number a
    # participant read off the split block is wrong. A role that is absent
    # altogether is not an error -- a phase-1 delivery ships no TEST rows at
    # all -- and which roles a phase actually needs is sample_rows's business.
    present = frame[ROLE_COLUMN].value_counts()
    for role, key in zip(ROLES, ROLE_COUNTS):
        found, declared = int(present.get(role, 0)), schema["split"][key]
        if found not in (0, declared):
            raise ValueError("%s holds %d %s respondents; the schema declares "
                             "%s = %d" % (path, found, role, key, declared))

    for item in items:
        allowed = set(options_for(schema, item))
        column = frame[item]
        stray = set(column[column.notna()].unique()) - allowed
        if stray:
            raise ValueError(
                "%s: %s holds %d value(s) the schema does not list, e.g. %r"
                % (path, item, len(stray), sorted(stray, key=str)[:3]))

    if frame["respondent_id"].duplicated().any():
        raise ValueError("%s repeats a respondent_id" % path)

    # Schema order, not file order: everything downstream goes by position.
    return frame[["respondent_id", ROLE_COLUMN] + items].reset_index(drop=True)


# ------------------------------------------------------------------ sample --

PHASE_ROLES = {1: {"visible": ("TRAIN",), "hidden": "DEV"},
               2: {"visible": ("TRAIN", "DEV"), "hidden": "TEST"}}


def sample_rows(schema, respondents, phase):
    """Take this phase's rows and blank what the submission has to predict.

    Roles decide it, and they were decided when the dataset was built. Phase 1
    shows TRAIN complete and scores DEV; phase 2 shows TRAIN and DEV complete,
    answers included, which is where DEV is worth most, and scores TEST. So a
    phase-1 leaderboard probed all through development is not an answer key for
    phase 2, and TEST answers never enter a container while they are still the
    thing being predicted.

    Nothing is sampled here and nothing is thinned. A phase takes every row of
    every role it is entitled to, so every submission in a phase sees the same
    rows and is scored on exactly the same cells, which is what makes two
    leaderboard entries comparable to each other.

    Returns (frame, cells, truth). `frame` is what predict() receives: visible
    respondents complete, hidden respondents with every PREDICT cell NaN.
    `cells` is the canonical ordering -- rows top to bottom, and within a row,
    items in schema order.
    """
    items = generated_items(schema)
    scored = scored_items(schema)
    roles = PHASE_ROLES[phase]

    absent = [role for role in roles["visible"] + (roles["hidden"],)
              if not (respondents[ROLE_COLUMN] == role).any()]
    if absent:
        raise ValueError("phase %d needs %s respondents and the dataset holds "
                         "none" % (phase, " and ".join(absent)))

    visible = respondents[respondents[ROLE_COLUMN].isin(roles["visible"])]
    hidden = respondents[respondents[ROLE_COLUMN] == roles["hidden"]]

    visible = visible[["respondent_id"] + items].reset_index(drop=True)
    hidden = hidden[["respondent_id"] + items].reset_index(drop=True)

    blanked = hidden.copy()
    blanked[scored] = np.nan
    frame = pd.concat([visible, blanked], ignore_index=True)

    offset = len(visible)
    cells, truth = [], []
    ids = hidden["respondent_id"].to_numpy(dtype=object)
    on = set(scored)
    for row in range(len(hidden)):
        for item in items:
            if item in on:
                cells.append((offset + row, ids[row], item))
                truth.append(hidden[item].iloc[row])
    return frame, cells, truth


# ------------------------------------------------------------------ privacy --

def privatize(value, config, n_respondents, phase, uniform_reference,
              rng=None):
    """The one number that leaves the grader.

    Whether it is noised is a property of the phase. Phase 1 scores the DEV
    respondents, the same ones on every submission across a whole development
    period, so the leaderboard is a query channel and the answer has to be
    noised. Phase 2 scores TEST, once per team, against respondents phase 1
    never touched -- there is no sequence to difference, so the score is exact.

    That the scored set is fixed within a phase is what makes noising the right
    defence rather than a workaround: the repetition is the whole exposure, and
    it is bounded and accountable. Redrawing who is scored per submission would
    spread the exposure over every respondent instead, and buy no amplification
    in return, because the frame says outright which rows are held out.

    When noise does apply: every held-out respondent contributes exactly one
    cell per PREDICT item, so the log score is a plain mean over respondents
    and dropping one of them moves it by at most the per-cell bound over the
    respondent count. That bound exists only because `floored` puts every
    probability at or above scoring.floor -- without the floor a single
    confident miss is unbounded, and so is the sensitivity. Skill divides the
    log score by the schema's uniform reference, so its sensitivity divides by
    the same constant.

    The draw is deliberately not seeded from --seed. A participant who could
    reproduce the noise could subtract it, and the mechanism would be theatre.
    """
    settings = config["phases"][phase]
    round_to = settings.get("round_to")

    if not settings.get("noised", True):
        reported = value if not round_to else round(
            round(value / round_to) * round_to, 10)
        return float(reported), {"noised": False, "round_to": round_to}

    privacy = config["privacy"]
    sensitivity = (-np.log(config["scoring"]["floor"])
                   / (n_respondents * uniform_reference))
    scale = sensitivity / privacy["epsilon"]

    rng = np.random.default_rng() if rng is None else rng
    noised = float(value + rng.laplace(0.0, scale))
    if round_to:
        noised = round(round(noised / round_to) * round_to, 10)
    return noised, {"noised": True, "sensitivity": sensitivity, "scale": scale,
                    "epsilon": privacy["epsilon"], "round_to": round_to}


# ------------------------------------------------------------- environment --

def _declares_anything(requirements):
    """Does this requirements.txt actually ask for a package?

    A file holding only comments is the same as no file: it would otherwise
    buy an empty venv, and a submission that imports pandas -- which the hosted
    image has -- would fail here for a reason the worker does not have.
    """
    if not os.path.exists(requirements):
        return False
    with open(requirements, encoding="utf-8") as fh:
        return any(line.strip() and not line.strip().startswith("#")
                   for line in fh)


def prepare_environment(submission, workdir, python):
    """Create the venv, install requirements with the network up, then cut it.

    requirements.txt is optional, as it is on the worker: the hosted image
    already carries numpy, pandas, torch, transformers and the rest, and the
    file exists only to add what the image does not have. A submission that
    declares nothing therefore runs in the environment score.py itself is
    running in, which is what stands in for that image here -- a fresh venv
    would not even hold pandas, and the local harness would fail submissions
    the worker runs happily. A submission that does declare something gets the
    isolated venv, where an undeclared import is the ImportError it would be
    on the worker. `--docker` is the strict path either way.

    Returns the interpreter to run the driver with.
    """
    requirements = os.path.join(submission, "requirements.txt")
    if not _declares_anything(requirements):
        print("[install] nothing declared; running in this environment, "
              "which stands in for the hosted image")
        return python

    venv = os.path.join(workdir, "venv")
    binary = os.path.join(venv, "Scripts" if os.name == "nt" else "bin",
                          "python.exe" if os.name == "nt" else "python")

    subprocess.run([python, "-m", "venv", venv], check=True, capture_output=True)
    print("[install] %s, network up" % requirements)
    done = subprocess.run([binary, "-m", "pip", "install", "--quiet",
                           "-r", requirements],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise SubmissionError(
            "requirements.txt did not install:\n" + done.stderr.strip())
    return binary


def stage(schema, masked, workdir):
    """Write what the driver hands to predict().

    The schema goes across verbatim: predict() receives the same file that is in
    data/, so there is no second view of the schema to keep in step with the
    first. JSON rather than parquet for the frame, so the submission's
    environment needs nothing beyond what it declared and values arrive as the
    exact objects the schema lists rather than whatever a CSV round trip infers.
    """
    items = generated_items(schema)
    columns = ["respondent_id"] + items
    rows = [[None if pd.isna(value) else value for value in row]
            for row in masked[columns].to_numpy(dtype=object)]
    with open(os.path.join(workdir, "data.json"), "w", encoding="utf-8") as fh:
        json.dump({"columns": columns, "data": rows}, fh)
    with open(os.path.join(workdir, "schema.json"), "w", encoding="utf-8") as fh:
        json.dump(schema, fh, ensure_ascii=False)


def run_submission(binary, submission, workdir, timeout, image, runner,
                   docker=False, verbose=True):
    """Call predict() with the network off. Returns the raw vectors.

    `verbose` is the phase's logging level. A crash in phase 1 comes back with
    its whole traceback, because development is when a participant has to be
    able to fix things. In phase 2 they get the exception line and nothing
    else: the traceback of a run over the full data can carry values out of it.
    The full text goes to the run log either way.
    """
    driver = os.path.join(workdir, "driver.py")
    with open(driver, "w", encoding="utf-8") as fh:
        fh.write(DRIVER)

    if docker:
        command = _docker_command(submission, workdir, image, runner)
        print("[run] docker --network=none, %dg, %d cpus, network off"
              % (runner["memory_gb"], runner["cpus"]))
    else:
        command = [binary, driver, workdir, os.path.abspath(submission)]
        print("[run] sockets disabled in-process, network off")

    environment = dict(os.environ)
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "HF_HUB_DISABLE_TELEMETRY": "1", "WANDB_MODE": "disabled",
                        "PYTHONDONTWRITEBYTECODE": "1"})

    try:
        done = subprocess.run(command, capture_output=True, text=True,
                              timeout=timeout, env=environment)
    except subprocess.TimeoutExpired:
        raise SubmissionError("predict() did not finish within %ds" % timeout)

    if done.returncode != 0:
        # Lead with the exception itself. Python puts it on the last line, and
        # a participant reading a wall of traceback should not have to hunt.
        trace = done.stderr.strip()
        reason = trace.splitlines()[-1] if trace else "no output"
        error = SubmissionError("predict() failed: %s%s"
                                % (reason, "\n\n" + trace if verbose else ""))
        error.detail = trace
        raise error

    output = os.path.join(workdir, "predictions.json")
    if not os.path.exists(output):
        raise SubmissionError("predict() returned nothing the driver could write")
    with open(output, encoding="utf-8") as fh:
        return json.load(fh)


def _docker_command(submission, workdir, image, runner):
    """The container, capped at the worker's own limits.

    Memory and CPU come from `runner` in config.yml rather than being written
    here, so the local rehearsal is bounded the way the worker is: a submission
    that fits locally fits there, and one that does not is killed here, where
    the participant can see why.
    """
    if shutil.which("docker") is None:
        raise SubmissionError("--docker was requested but docker is not installed")
    context = os.path.join(workdir, "image")
    os.makedirs(context, exist_ok=True)
    # The driver itself reads the frame with pandas, so the base image needs it
    # whether or not the submission declares anything. That is the floor, and
    # nothing above it is implied: the hosted image ships far more, and a
    # submission that leans on the rest of it has to say so in requirements.txt
    # to see it here.
    lines = ["FROM %s" % image,
             "RUN pip install --no-cache-dir %s" % " ".join(DRIVER_PACKAGES)]
    requirements = os.path.join(submission, "requirements.txt")
    if os.path.exists(requirements):
        shutil.copy(requirements, context)
        lines += ["COPY requirements.txt .",
                  "RUN pip install --no-cache-dir -r requirements.txt"]
    with open(os.path.join(context, "Dockerfile"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    # The build is the install phase: it is the only step with a network.
    print("[install] docker build, network up")
    subprocess.run(["docker", "build", "--quiet", "-t", "sbench-submission",
                    context], check=True, capture_output=True)
    return ["docker", "run", "--rm", "--network=none", "--read-only",
            "--memory=%dg" % runner["memory_gb"],
            "--cpus=%d" % runner["cpus"], "--pids-limit=256",
            "--tmpfs", "/tmp",
            "-v", "%s:/work" % os.path.abspath(workdir),
            "-v", "%s:/submission:ro" % os.path.abspath(submission),
            "sbench-submission", "python", "/work/driver.py", "/work",
            "/submission"]


# ---------------------------------------------------------------- scoring --

def check(schema, vectors, cells):
    """Every rule the grader enforces. Raises on the first thing that is wrong."""
    if len(vectors) != len(cells):
        raise SubmissionError(
            "predict() returned %d vectors for %d hidden cells. Check the "
            "canonical order: rows top to bottom, items in schema key order."
            % (len(vectors), len(cells)))

    for index, (vector, (_, _, item)) in enumerate(zip(vectors, cells)):
        width = len(options_for(schema, item))
        p = np.asarray(vector, dtype=float)
        if p.shape != (width,):
            raise SubmissionError(
                "vector %d is for %s and should have %d entries, not %d"
                % (index, item, width, p.size))
        if not np.isfinite(p).all():
            raise SubmissionError("vector %d for %s contains NaN or infinity"
                                  % (index, item))
        if (p < 0).any():
            raise SubmissionError("vector %d for %s contains a negative value"
                                  % (index, item))
        if p.sum() <= 0:
            raise SubmissionError("vector %d for %s sums to zero" % (index, item))


def floored(vectors, schema, cells, floor):
    """Renormalise, then mix with a flat vector so nothing is ever zero.

    Mixing rather than clipping keeps both promises at once: every entry is at
    least `floor` and the vector still sums to 1. Clip-then-renormalise
    pushes the clipped entries back under the floor and keeps neither.
    """
    out = []
    for vector, (_, _, item) in zip(vectors, cells):
        p = np.asarray(vector, dtype=float)
        p = p / p.sum()
        width = p.size
        out.append(floor + (1.0 - width * floor) * p)
    return out


def uniform_reference(schema):
    """Mean surprisal of a uniform guess, in nats, from the schema alone.

    log K averaged over the scored items, where K counts an item's options with
    the gate sentinel included. No data is involved: this is a property of the
    instrument, fixed before anybody submits anything, and it is what makes the
    three instruments comparable. ERPIS asks 213 mostly-binary questions and
    the Skills Assessment asks 73 much wider ones, so a nat is not worth the
    same in each.
    """
    scored = scored_items(schema)
    if not scored:
        raise ValueError("schema has no PREDICT items")
    return float(np.mean([np.log(len(options_for(schema, item)))
                          for item in scored]))


def score(schema, config, vectors, truth, cells, seed=0):
    """Mean log score and the skill it normalises to.

    The metric is the log probability the submission gave the answer that was
    actually there, averaged over blank cells. It is at most 0 and at least
    log(floor); higher is better.

    `skill` puts that on a scale the three instruments share: 0 is a uniform
    guess, 1 is perfect, negative is worse than guessing. Both terms come from
    the log score and the schema, so nothing about it can be tuned.
    """
    logp, briers, items = [], [], []

    for vector, actual, (_, _, item) in zip(vectors, truth, cells):
        options = options_for(schema, item)
        if actual not in options:
            raise ValueError("%s holds %r, which is not one of its options"
                             % (item, actual))
        k = options.index(actual)
        logp.append(np.log(vector[k]))
        target = np.zeros(len(options))
        target[k] = 1.0
        briers.append(float(((vector - target) ** 2).sum()))
        items.append(item)

    if not logp:
        raise ValueError("nothing was held out, so there is nothing to score")

    logp = np.asarray(logp)
    uniform = uniform_reference(schema)
    by_item = (pd.DataFrame({"item": items, "log_score": logp})
               .groupby("item")["log_score"].agg(["mean", "size"]))

    return {
        "log_score": float(logp.mean()),
        "uniform_reference": uniform,
        "skill": 1.0 + float(logp.mean()) / uniform,
        "std_error": cluster_std_error(
            logp, [cell[1] for cell in cells],
            draws=config["scoring"]["bootstrap_draws"], seed=seed),
        "item_normalized": float(by_item["mean"].mean()),
        "brier": float(np.mean(briers)),
        "n_cells": int(logp.size),
        "by_item": by_item,
    }


def cluster_std_error(values, respondents, draws=2000, seed=0):
    """Standard error that respects clustering of cells within respondents.

    One respondent contributes many cells and their scores move together.
    Treating cells as independent understates the uncertainty, sometimes by a
    factor of two. Resample whole respondents instead.
    """
    values = np.asarray(values, dtype=float)
    _, index = np.unique(np.asarray(respondents, dtype=object),
                         return_inverse=True)
    totals = np.bincount(index, weights=values)
    counts = np.bincount(index).astype(float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, totals.size, size=(draws, totals.size))
    means = totals[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return float(means.std(ddof=1))


def baselines(schema, truth, cells, floor):
    """What a model has to beat, as log scores: uniform, and the crowd marginal.

    Both are higher-is-better, on the same scale as what a submission gets.
    """
    uniform, counts = [], {}
    for actual, (_, _, item) in zip(truth, cells):
        counts.setdefault(item, {}).setdefault(actual, 0)
        counts[item][actual] += 1
        uniform.append(-np.log(len(options_for(schema, item))))

    crowd = []
    for actual, (_, _, item) in zip(truth, cells):
        table = counts[item]
        total = sum(table.values())
        crowd.append(np.log(max(table[actual] / total, floor)))
    return float(np.mean(uniform)), float(np.mean(crowd))


# -------------------------------------------------------------------- main --

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--submission", default="baseline/marginal_counts",
                        help="directory holding main.py and requirements.txt")
    parser.add_argument("--data", required=True,
                        help="the delivered parquet, or a directory holding "
                             "data.parquet (what make_sandbox.py writes)")
    parser.add_argument("--schema", default="data/unicef.json")
    parser.add_argument("--config", default="config.yml",
                        help="organizer-side configuration")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1,
                        help="competition phase; sets which roles are visible "
                             "and which are scored, the timeout, the "
                             "rounding and the logging level")
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds the clustered bootstrap behind std_error. "
                             "Who is held out does not depend on it: roles are "
                             "in the delivered file, not drawn here")
    parser.add_argument("--timeout", type=int, default=None,
                        help="override the phase's timeout, in seconds")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter to build the submission's venv from")
    parser.add_argument("--docker", action="store_true",
                        help="enforce the network cut with a container")
    parser.add_argument("--keep", action="store_true",
                        help="keep the work directory for inspection")
    parser.add_argument("--log", default=None,
                        help="write the organizer-side log dict here as JSON")
    parser.add_argument("--show-log", action="store_true",
                        help="print the organizer-side log. Never available to "
                             "a participant; this is the local rehearsal only")
    args = parser.parse_args()

    started = time.time()
    config = load_config(args.config)
    settings = config["phases"][args.phase]
    timeout = args.timeout if args.timeout is not None \
        else settings["timeout_seconds"]
    verbose = settings["logging"] == "verbose"

    schema = load_schema(args.schema, config)
    # No row-count warning here: load_frames has already checked every role
    # against the count the schema declares for it, which says the same thing
    # and says which role is wrong.
    respondents = load_frames(args.data, schema)
    masked, cells, truth = sample_rows(schema, respondents, args.phase)
    # Held-out respondents, whatever role they carry: DEV in phase 1, TEST
    # in phase 2. Not to be read as "the TEST rows".
    n_held_out = len({cell[1] for cell in cells})
    print("[phase] %d (%s): %s visible, %s scored, %ds for predict()"
          % (args.phase, settings["name"],
             "+".join(PHASE_ROLES[args.phase]["visible"]),
             PHASE_ROLES[args.phase]["hidden"], timeout))
    print("[data] %s against %s: %d respondents, %d held out, %d cells"
          % (args.data, args.schema, len(masked), n_held_out, len(cells)))

    # Everything the grader learns. Written to the run log, never returned:
    # only `score` below goes back to the participant.
    logs = {"data": args.data, "schema": args.schema, "phase": args.phase,
            "seed": args.seed,
            "n_respondents": len(masked), "n_held_out": n_held_out,
            "n_cells": len(cells)}

    workdir = tempfile.mkdtemp(prefix="sbench-")
    try:
        stage(schema, masked, workdir)
        # In docker mode the image build is the install stage and the container
        # is the isolation, so there is no venv to make.
        binary = (None if args.docker
                  else prepare_environment(args.submission, workdir,
                                           args.python))
        vectors = run_submission(binary, args.submission, workdir, timeout,
                                 config["scoring"]["docker_image"],
                                 config["runner"],
                                 docker=args.docker, verbose=verbose)
        check(schema, vectors, cells)
        vectors = floored(vectors, schema, cells,
                          config["scoring"]["floor"])
        result = score(schema, config, vectors, truth, cells, seed=args.seed)
    except SubmissionError as exc:
        logs["status"] = "FAIL"
        logs["error"] = getattr(exc, "detail", str(exc))
        write_log(logs, args, verbose)
        print("\nFAIL  %s" % exc)
        return 1
    finally:
        if args.keep:
            print("[work] %s" % workdir)
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    uniform, crowd = baselines(schema, truth, cells,
                               config["scoring"]["floor"])
    reported, mechanism = privatize(result["skill"], config, n_held_out,
                                    args.phase, result["uniform_reference"])
    elapsed = time.time() - started

    logs.update({
        "status": "PASS",
        "skill": result["skill"],
        "reported_skill": reported,
        "log_score": result["log_score"],
        "uniform_reference": result["uniform_reference"],
        "privacy": mechanism,
        "std_error": result["std_error"],
        "item_normalized": result["item_normalized"],
        "brier": result["brier"],
        "baseline_uniform": uniform,
        "baseline_crowd": crowd,
        "seconds": elapsed,
        "by_item": {item: {"log_score": float(row["mean"]),
                           "n_cells": int(row["size"])}
                    for item, row in result["by_item"].iterrows()},
    })
    write_log(logs, args, verbose)

    # The whole of what a participant gets back.
    print("\nPASS  %.4f  (%.1fs)" % (reported, elapsed))
    return 0


def write_log(logs, args, verbose):
    """Put the organizer-side quantities somewhere they are kept, not returned.

    `verbose` drops the per-item breakdown in phase 2. It is the most useful
    thing in here and the most re-identifying: a per-item loss over a small
    held-out set is close to a query about particular people.
    """
    if not verbose:
        logs.pop("by_item", None)
    if args.log:
        with open(args.log, "w", encoding="utf-8") as fh:
            json.dump(logs, fh, indent=2, ensure_ascii=False, default=str)
            fh.write("\n")
        print("[log] %s" % args.log)
    if args.show_log:
        print("\n--- organizer-side log, not returned to the participant ---")
        print(json.dumps(logs, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    sys.exit(main())
