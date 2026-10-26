import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import make_sandbox as ms
import score as scoring


@pytest.mark.parametrize("phase", [1, 2])
def test_public_baseline_is_zero_and_release_preserves_original_noise(tmp_path, phase):
    config = ms.load_config(str(ROOT / "config.yml"))
    schema = ms.load_schema(str(ROOT / "data/sample.json"), config)
    ms.write_sandbox(schema, config, str(tmp_path), seed=0)
    respondents = scoring.load_frames(str(tmp_path), schema)
    frame, cells, truth = scoring.sample_rows(schema, respondents, phase)
    spec = importlib.util.spec_from_file_location("baseline", ROOT / "baseline/marginal_counts/main.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    vectors = baseline.predict(frame.copy(), schema)
    vectors = scoring.floored(vectors, schema, cells, config["scoring"]["floor"])
    reference = scoring.marginal_reference(schema, frame, cells, truth, config["scoring"]["floor"])
    result = scoring.score(schema, config, vectors, truth, cells, reference)
    assert result["skill"] == pytest.approx(0, abs=1e-14)
    uniform = scoring.uniform_reference(schema)
    old = 1 + result["log_score"] / uniform
    released, mechanism = scoring.privatize(old, config, len({cell[1] for cell in cells}), phase, uniform, rng=np.random.default_rng(42))
    converted = scoring.normalize_released(released, uniform, reference)
    assert converted == 1 + (released - 1) * (uniform / reference)
    assert mechanism["noised"] == (phase == 1)
    if phase == 2:
        assert converted == pytest.approx(0, abs=1e-14)


def test_public_dates_are_aoe_and_failed_attempts_are_free():
    config = ms.load_config(str(ROOT / "config.yml"))
    dev, test = config["phases"][1], config["phases"][2]
    assert dev["closes_exclusive_utc"] == "2026-11-14T12:00:00Z"
    assert test["opens_utc"] == "2026-11-15T12:00:00Z"
    assert test["closes_exclusive_utc"] == "2026-11-20T12:00:00Z"
    assert dev["max_submissions_per_day"] == 3
    assert test["max_submissions"] == 1
    assert not dev["failed_submissions_count"]
    assert not test["failed_submissions_count"]
