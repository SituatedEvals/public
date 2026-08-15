"""External model baseline: a Hugging Face LLM supplies the prior, the visible
answers supply the likelihood.

The model is named in `models.txt`, one repo id per line. The platform downloads
it into the image cache before the container starts, so `from_pretrained()`
finds it offline at runtime -- no network while `predict()` runs, nothing to
bundle in the ZIP.

Two ideas, in descending order of what they are worth.

  1. GATE RECURSION. Whether an item was asked is a deterministic function of
     its parent's answer, so P(asked) is a sum over the parent's distribution,
     computed down the gate chain. When the parent is GIVEN it is an indicator
     and the sentinel slot is free score.

  2. DIRICHLET BACKOFF ONTO AN LLM PRIOR. Smooth stratum counts into coarser
     counts into a base measure: (counts + ALPHA * coarser) / (total + ALPHA).
     Strata are the GIVEN columns carrying the most mutual information with the
     item. The bottom of that ladder is the model's distribution over the
     item's options, read from next-token logits in one forward pass per item.
     The schema carries real question wording and real option text, so this is
     ordinary world knowledge.

ALPHA is what the prior is worth in people, so a bad prior is diluted by the
first real counts rather than believed. Everything degrades: no model named, no
torch, or a load failure, and the base measure falls back to uniform with the
rest of the pipeline unchanged.

**Do not expect this to beat `baseline/marginal_counts` on `data/sample.json`,
and do not read that as the prior failing.** The sandbox draws each item's
marginal from a random Dirichlet and its dependencies from random loadings on
an invented latent trait. What the LLM contributes is knowledge of how people
actually answer -- roughly what share of households hold property documents,
where a Big Five subscale centres -- and the sandbox's answers have no
relationship to the world at all. A prior that is *right about reality* is
therefore uninformative there by construction, and the more accurate it is the
less it can help. On top of that, ALPHA pseudo-counts against the thousands of
visible answers in a sandbox frame is a rounding error either way.

So use the sandbox for what it can show: that this runs, inside the time
budget, and returns legal vectors on all three schemas. Whether the prior earns
its place is a question only the real instruments can answer, and finding out
is the competition.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models.txt")

ALPHA = 2.0        # pseudocounts carried down each backoff level
STRATA = 2         # GIVEN columns per item, most informative first
MIN_COUNT = 8      # a column with less than this to go on is not worth ranking
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def declared_model():
    """The first Hugging Face repo id in models.txt, or None if there is none.

    The platform downloads everything named in that file into the image cache
    before the container starts, so `from_pretrained(repo_id)` resolves offline
    at runtime. Blanks and comments are skipped here, but keep the shipped file
    bare: the hosted pre-download reads it a line at a time and would try to
    fetch a comment as if it were a repo.
    """
    try:
        with open(MODELS, encoding="utf-8") as fh:
            for line in fh:
                repo = line.strip()
                if repo and not repo.startswith("#"):
                    return repo
    except OSError:
        pass
    return None


def llm_prior(schema, targets, values):
    """{item: distribution over its values}, or {} if no model is available.

    One prompt per item, one forward pass, softmax over the option letters'
    first-token logits. It never generates. Anything at all going wrong here
    returns {}, and predict() falls back to a uniform base measure.
    """
    repo = declared_model()
    if repo is None:
        return {}
    # The cache is already populated; this stops transformers reaching out to
    # check for a newer revision, which would fail on an isolated container.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(repo)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Cast after loading rather than through a from_pretrained keyword:
        # transformers renamed that argument between 4.x and 5.x, and this
        # works on both.
        model = AutoModelForCausalLM.from_pretrained(repo).to(
            device=device,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32).eval()
    except Exception:
        return {}

    # Every token an option letter might arrive as, so the answer can be read
    # straight off the logits: "A" and " A" are different tokens.
    letter_ids = []
    for letter in LETTERS:
        ids = {tokenizer.encode(form, add_special_tokens=False)[0]
               for form in (letter, " " + letter)
               if tokenizer.encode(form, add_special_tokens=False)}
        letter_ids.append(sorted(ids))

    survey = schema.get("dataset", {}).get("description", "")[:600]
    prior = {}
    try:
        for item in targets:
            options = values[item][:len(LETTERS)]
            body = "\n".join(
                ["You are estimating how a survey respondent answered.", "",
                 "Survey: " + survey, "",
                 "Question: " + schema["items"][item].get("question", item), "",
                 "Options:"]
                + ["%s. %s" % (LETTERS[i], option)
                   for i, option in enumerate(options)]
                + ["", "Reply with the single letter of the most likely answer."])
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": body}],
                    tokenize=False, add_generation_prompt=True)
            except Exception:          # not an instruct model
                prompt = body + "\nAnswer:"

            with torch.no_grad():
                logits = model(**tokenizer(prompt, return_tensors="pt").to(device)
                               ).logits[0, -1].float().cpu().numpy()

            scores = np.array([max(logits[i] for i in letter_ids[k])
                               for k in range(len(options))], float)
            p = np.exp(scores - scores.max())
            p = p / p.sum()
            if len(options) < len(values[item]):      # more than 26 options
                p = np.concatenate([p, np.full(len(values[item]) - len(options),
                                               p.mean())])
            prior[item] = p / p.sum()
    except Exception:
        return {}
    return prior


def codes(series, levels):
    """Answers as indices into `levels`; -1 for the sentinel or a blank."""
    lookup = {level: i for i, level in enumerate(levels)}
    return series.map(lookup).fillna(-1).astype(np.int64).to_numpy()


def mutual_information(x, y, nx, ny):
    """Plug-in mutual information in nats, over the rows where both are known."""
    both = (x >= 0) & (y >= 0)
    x, y = x[both], y[both]
    if x.size < MIN_COUNT:
        return 0.0
    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny) / x.size
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())


def predict(frame, schema):
    # ---- what the instrument is ------------------------------------------
    items = [n for n, r in schema["items"].items()
             if r["class"] in ("GIVEN", "PREDICT")]
    given = [n for n in items if schema["items"][n]["class"] == "GIVEN"]
    targets = [n for n in items if schema["items"][n]["class"] == "PREDICT"]
    values = {n: list(schema["items"][n]["values"]) for n in items}
    gate = {n: schema["items"][n].get("gate") for n in items}

    # Gate parents before their children, so the recursion below can rely on
    # the parent's distribution already being computed.
    def depth(name):
        levels, seen = 0, set()
        while gate.get(name) and name not in seen:
            seen.add(name)
            name = gate[name]["parent"]
            levels += 1
        return levels

    # ---- the two sources of information ----------------------------------
    prior = llm_prior(schema, targets, values)

    visible = frame.loc[~frame[targets].isna().any(axis=1)]
    if len(visible) < 50:                      # nothing to count: prior only
        visible = frame
    seen_codes = {g: codes(visible[g], values[g]) for g in given}
    card = {g: len(values[g]) for g in given}

    n = len(frame)
    frame_codes = {g: codes(frame[g], values[g]) for g in given}

    # ---- P(answer | asked), one item at a time ---------------------------
    answer = {}
    for item in targets:
        y = codes(visible[item], values[item])
        asked = y >= 0                         # the sentinel codes as -1
        width = len(values[item])

        ranked = sorted(given, reverse=True, key=lambda g: mutual_information(
            seen_codes[g][asked], y[asked], card[g], width))
        ladder = ranked[:STRATA]

        # Start from the prior, then let each backoff level pull towards the
        # counts it actually has. ALPHA is what the prior is worth in people.
        running = np.tile(prior.get(item, np.full(width, 1.0 / width)), (n, 1))
        key_fit = np.zeros(int(asked.sum()), dtype=np.int64)
        key_all = np.zeros(n, dtype=np.int64)
        size = 1
        for column in [None] + ladder:
            if column is not None:
                key_fit = key_fit * card[column] + np.maximum(
                    seen_codes[column][asked], 0)
                key_all = key_all * card[column] + np.maximum(
                    frame_codes[column], 0)
                size *= card[column]
            table = np.bincount(key_fit * width + y[asked],
                                minlength=size * width).astype(float)
            counts = table.reshape(size, width)[np.minimum(key_all, size - 1)]
            running = ((counts + ALPHA * running)
                       / (counts.sum(axis=1, keepdims=True) + ALPHA))
        answer[item] = running

    # ---- P(asked), down the gate chain -----------------------------------
    asked_prob = {}
    for item in sorted(targets, key=depth):
        if gate[item] is None:
            asked_prob[item] = np.ones(n)
            continue
        parent, allowed = gate[item]["parent"], gate[item]["observed_if"]
        if schema["items"][parent]["class"] == "GIVEN":
            # The parent is visible for everybody, so this is not a guess.
            asked_prob[item] = np.isin(frame[parent].to_numpy(dtype=object),
                                       allowed).astype(float)
        else:
            opens = [values[parent].index(v) for v in allowed
                     if v in values[parent]]
            asked_prob[item] = asked_prob[parent] * (
                answer[parent][:, opens].sum(axis=1) if opens else 0.0)

    # ---- one vector per blank cell, sentinel last ------------------------
    full = {}
    for item in targets:
        if gate[item] is None:
            full[item] = answer[item]
        else:
            p = asked_prob[item][:, None]
            full[item] = np.concatenate([p * answer[item], 1.0 - p], axis=1)

    grid = frame[items].to_numpy(dtype=object)
    return [full[items[column]][row]
            for row in range(n)
            for column in range(len(items))
            if pd.isna(grid[row, column])]
