"""Build a synthetic practice dataset from an instrument schema.

    python make_sandbox.py --schema data/unicef.json --out _sandbox

Reads one of the schemas in data/ and writes a parquet file of invented
respondents, as many rows as the delivered file has. score.py imports
load_config, load_schema, options_for and make_sandbox from here, so the
generator and the grader cannot disagree about option order.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

# The one stream the sandbox draws from. Roles do not draw at all -- they are
# counted off the schema -- so there is no second stream to keep separate from
# the hidden trait that decides how a respondent answers.
STREAM_DATA = 11

# What a respondent is for, decided once when the dataset is built and never
# recomputed. TRAIN is visible in both phases; DEV is scored in phase 1 and
# becomes visible in phase 2; TEST is scored in phase 2 and is not shipped at
# all before then, so those answers never enter a submission's container while
# they are still the thing being predicted.
ROLES = ("TRAIN", "DEV", "TEST")
ROLE_COUNTS = ("n_train", "n_dev", "n_test")
ROLE_COLUMN = "role"


def load_config(path="config.yml"):
    """Read the organizer-side configuration and check the phases are sane."""
    with open(path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    for number, phase in config["phases"].items():
        # Explicit, never defaulted: whether a phase's score is noised decides
        # whether the leaderboard leaks, and it should not be silently on.
        if not isinstance(phase.get("noised"), bool):
            raise ValueError("phase %s: noised must be true or false" % number)
    if config["privacy"]["mechanism"] != "laplace":
        raise ValueError("unknown privacy mechanism %r"
                         % config["privacy"]["mechanism"])
    return config


def load_schema(path, config):
    """Read and validate schema.

    Option lists are taken as given. The delivered files are preprocessed so
    that every variable arrives categorical with a declared set of levels,
    which is why there is no numeric case here: a schema that enumerated what a
    continuous variable contained would be listing observed responses, and
    banding it is the preprocessing step's job rather than the grader's.

    Preprocessing also means no option is ever null. NaN in the frame has
    exactly one meaning -- this cell is held out, predict it -- and an item
    whose option list contained a missing value would make a blank cell
    ambiguous between "predict this" and "they did not answer". Genuine item
    non-response is a level like any other: "Prefer not to answer",
    "98. I don't know", "99. Refused to answer". The check below enforces it.
    """
    with open(path, encoding="utf-8") as fh:
        schema = json.load(fh)

    # One definition of the sentinel for all three instruments.
    schema["gated_value"] = config["gated_value"]

    items = schema["items"]
    for name, rec in items.items():
        if rec["class"] in ("GIVEN", "PREDICT") and not rec.get("values"):
            raise ValueError("%s is %s but has no values to draw from"
                             % (name, rec["class"]))
        if any(v is None for v in rec.get("values") or ()):
            raise ValueError(
                "%s lists a null option. NaN means 'held out, predict this' "
                "and cannot also mean an answer: give non-response its own "
                "level instead." % name)
        stray = [v for v in rec.get("values") or () if not isinstance(v, str)]
        if stray:
            raise ValueError(
                "%s lists %r as a number. Options have to be strings: a gated "
                "column carries the sentinel alongside them, which no numeric "
                "column can hold, and a CSV round trip would bring them back "
                "as text and stop matching. Quote them." % (name, stray[:3]))
        if schema["gated_value"] in (rec.get("values") or ()):
            raise ValueError(
                "%s lists %r among its values. The sentinel is appended by "
                "options_for, never enumerated." % (name, schema["gated_value"]))
        gate = rec.get("gate")
        if gate and gate["parent"] not in items:
            raise ValueError("%s gates on %r, which is not in the schema"
                             % (name, gate["parent"]))

    dataset = schema["dataset"]
    for key in ("n_rows", "version", "description"):
        if key not in dataset:
            raise ValueError("dataset is missing %r" % key)
    if not isinstance(dataset["n_rows"], int) or dataset["n_rows"] <= 0:
        raise ValueError("dataset.n_rows must be a positive integer, not %r"
                         % (dataset["n_rows"],))

    # Counts, not shares. How many respondents carry each role is the thing
    # worth reading -- it says outright how much is visible in a phase and how
    # much is scored -- and a count cannot drift from n_rows through a rounding
    # step the way a share can. All three are written out, so the one thing
    # that can go wrong is that they stop agreeing with n_rows.
    split = schema["split"]
    missing = [key for key in ROLE_COUNTS if key not in split]
    if missing:
        raise ValueError("split is missing %s" % ", ".join(missing))
    for key in ROLE_COUNTS:
        count = split[key]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("split.%s must be a count of respondents, a "
                             "non-negative integer, not %r" % (key, count))
    total = sum(split[key] for key in ROLE_COUNTS)
    if total != dataset["n_rows"]:
        raise ValueError("%s sum to %d, but dataset.n_rows is %d: every "
                         "respondent has exactly one role"
                         % (" + ".join(ROLE_COUNTS), total, dataset["n_rows"]))
    for role, key in zip(ROLES[1:], ROLE_COUNTS[1:]):
        if split[key] < 1:
            raise ValueError("split.%s must be at least 1: a phase with no %s "
                             "respondents has nothing to score" % (key, role))

    return schema


def generated_items(schema):
    """Items the sandbox invents, in schema order.

    EXCLUDE items are identifiers, record keys, free text and administrative
    fields. They are never generated, never shown and never scored: the frame
    carries its own respondent_id, so a delivered key column is one more thing
    a submission could key on and nothing it could learn from.
    """
    return [name for name, rec in schema["items"].items()
            if rec["class"] in ("GIVEN", "PREDICT")]


def options_for(schema, name):
    """The option list for an item, in the one order that counts.

    A gated item can legitimately be "never asked", so the gate sentinel is a
    real option for it rather than a missing value, and it goes last. This is
    the single definition of option order: the generator, the grader and your
    predict() all have to agree about it, because your probability vector is
    read in this order.
    """
    rec = schema["items"][name]
    options = list(rec["values"])
    if rec.get("gate"):
        options.append(schema["gated_value"])
    return options


def scored_items(schema):
    """Items that can be held out and scored."""
    return [name for name, rec in schema["items"].items()
            if rec["class"] == "PREDICT"]


def _generation_order(schema, items):
    """Items sorted so that every gate parent precedes its children."""
    records = schema["items"]
    remaining = list(items)
    placed, order = set(), []
    while remaining:
        ready = [name for name in remaining
                 if not records[name].get("gate")
                 or records[name]["gate"]["parent"] not in remaining]
        if not ready:
            raise ValueError("gate definitions form a cycle among: %s"
                             % ", ".join(sorted(remaining)))
        for name in ready:
            order.append(name)
            placed.add(name)
        remaining = [name for name in remaining if name not in placed]
    return order


# -------------------------------------------------------------- generation --

def make_sandbox(schema, config, seed=0):
    """Invent the schema's worth of respondents, obeying its supports and skips.

    The row count comes from the schema rather than the caller. The sandbox is
    meant to be exactly the shape of the delivered file, and a size that can be
    passed in is a size that will eventually disagree with it.
    """
    settings = config["sandbox"]
    n = schema["dataset"]["n_rows"]
    rng = np.random.default_rng([seed, STREAM_DATA])
    items = generated_items(schema)
    if not items:
        raise ValueError("schema has no GIVEN or PREDICT items")

    # One hidden number per respondent, revealed by no column, nudging many
    # answers at once. Without it a respondent's answers would be independent
    # given their demographics, and the sandbox would be hostile to every
    # latent-factor method by construction.
    trait = rng.normal(size=n)

    columns = {}
    for name in _generation_order(schema, items):
        # Draw from the instrument's own support, not from options_for: the
        # gate sentinel is a legitimate answer for the grader to score, but it
        # is only ever produced by the gate below, never drawn at random.
        options = list(schema["items"][name]["values"])
        width = len(options)

        # An invented marginal for this item, skewed the way survey items are.
        base = np.log(rng.dirichlet(np.full(width, settings["dirichlet_alpha"]))
                      + 1e-12)
        logits = np.tile(base, (n, 1))

        # Everyone's answers shift together with the hidden trait.
        logits += np.outer(trait, rng.normal(scale=settings["trait_scale"],
                                             size=width))

        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        chosen = (weights.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
        values = np.asarray(options, dtype=object)[chosen]

        # Skip logic. A respondent whose gate did not open was never asked, so
        # the true value of the cell is the sentinel, not a missing value. A
        # parent that is itself gated carries the sentinel, which is never in
        # observed_if, so chains close without any special handling.
        gate = schema["items"][name].get("gate")
        if gate:
            parent = np.asarray(columns[gate["parent"]], dtype=object)
            skipped = ~np.isin(parent, np.asarray(gate["observed_if"],
                                                  dtype=object))
            values = np.where(skipped, schema["gated_value"], values)

        columns[name] = values

    frame = pd.DataFrame({name: columns[name] for name in items})
    frame.insert(0, "respondent_id", ["R%06d" % i for i in range(1, n + 1)])
    return frame.astype(object)


def assign_roles(schema, frame):
    """Give every respondent one role, in the counts the schema declares.

    Nothing is sampled. The first `n_train` rows are TRAIN, the next `n_dev`
    are DEV and the rest are TEST, so the composition of the file is exactly
    what the schema says it is rather than what a draw happened to produce, and
    the counts printed here are the counts a participant reads in the schema.

    Decided once and written to disk, not something the grader recomputes on
    every run. Fixing it is what keeps the leaderboard paired: every submission
    is scored on the same cells, so the variability they share cancels in the
    differences between them, which is all a leaderboard reports.

    Whole respondents go one way: a scored respondent has every PREDICT answer
    withheld, and splitting cells instead would leave a gated child visible
    while its parent was hidden, which gives the parent away.
    """
    counts = [schema["split"][key] for key in ROLE_COUNTS]
    if sum(counts) != len(frame):
        raise ValueError("split assigns %d roles but the frame has %d rows"
                         % (sum(counts), len(frame)))
    out = frame.copy()
    out[ROLE_COLUMN] = np.repeat(np.asarray(ROLES, dtype=object), counts)
    return out


def write_sandbox(schema, config, out, seed=0):
    """Write respondents.parquet and schema.json into `out`.

    One file with a role column rather than one file per role, because the
    thing worth checking is a property of the whole set -- every respondent has
    exactly one role -- and that is checkable in a single file and merely
    conventional across several.
    """
    os.makedirs(out, exist_ok=True)
    frame = assign_roles(schema, make_sandbox(schema, config, seed=seed))

    path = os.path.join(out, "respondents.parquet")
    try:
        frame.to_parquet(path, index=False)
    except (ImportError, ValueError) as exc:
        path = os.path.join(out, "respondents.csv")
        frame.to_csv(path, index=False)
        print("parquet unavailable (%s); wrote CSV instead" % exc)

    # The schema is what predict() receives, verbatim. There is no thinned-down
    # view of it: every key in the file is already public, and a second copy of
    # the schema would be a second thing to keep in step with the first.
    schema_path = os.path.join(out, "schema.json")
    with open(schema_path, "w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path, schema_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--schema", default="data/unicef.json",
                        help="instrument schema to build from")
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="_sandbox",
                        help="directory to write into (git-ignored)")
    args = parser.parse_args()

    config = load_config(args.config)
    schema = load_schema(args.schema, config)
    path, _ = write_sandbox(schema, config, args.out, seed=args.seed)
    items = generated_items(schema)
    frame = _read_any(path)
    counts = frame[ROLE_COLUMN].value_counts()
    print("%s -> %s" % (args.schema, args.out))
    print("  %d respondents, %d items (%d scored), %d gated"
          % (schema["dataset"]["n_rows"], len(items), len(scored_items(schema)),
             sum(1 for name in items if schema["items"][name].get("gate"))))
    print("  respondents.parquet")
    for role, held in (("TRAIN", "visible in both phases"),
                       ("DEV", "scored in phase 1, visible in phase 2"),
                       ("TEST", "scored in phase 2, not shipped before then")):
        print("    %-6s %6d rows   %s" % (role, counts.get(role, 0), held))
    print("  schema.json    what predict() receives")


def _read_any(path):
    return (pd.read_csv(path, dtype=object) if path.endswith(".csv")
            else pd.read_parquet(path))


if __name__ == "__main__":
    main()
