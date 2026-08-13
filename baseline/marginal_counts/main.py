"""Your submission. Edit predict(). See README.md for the rules and the
contract: what you are given, what you return, and in what order.
"""

import numpy as np
import pandas as pd


def predict(frame, schema):
    """Return one probability vector per blank cell, in canonical order.

    frame   respondent_id plus one column per item. NaN means "held out,
            predict this"; every other cell is an answer you may use.
    schema    the instrument schema, as in data/, plus schema["gated_value"].

    The baseline here predicts the crowd: for each item, the smoothed
    distribution of the answers that are visible. It ignores everything about
    the individual respondent, which is exactly what you are trying to beat.
    """
    items = [name for name, record in schema["items"].items()
             if record["class"] in ("GIVEN", "PREDICT")]

    options = {}
    for item in items:
        record = schema["items"][item]
        options[item] = list(record["values"]) + (
            [schema["gated_value"]] if record.get("gate") else [])

    marginals = {}
    for item in items:
        counts = frame[item].value_counts()
        # Half a count on every option, so an option nobody chose is unlikely
        # rather than impossible. A zero here would cost you the run.
        weights = np.array([counts.get(option, 0) + 0.5
                            for option in options[item]], float)
        marginals[item] = weights / weights.sum()

    values = frame[items].to_numpy(dtype=object)
    return [marginals[items[column]]
            for row in range(values.shape[0])
            for column in range(len(items))
            if pd.isna(values[row, column])]
