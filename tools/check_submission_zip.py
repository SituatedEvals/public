#!/usr/bin/env python3
"""Validate a SimulacraBench submission ZIP."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import math
import numbers
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path


# Keep in sync with orchestrator/codabench_shared/submissions/static_checks.py.
SENTENCE_TRANSFORMER_BASIC_MODEL_IDS = {
    "albert-base-v1",
    "albert-base-v2",
    "albert-large-v1",
    "albert-large-v2",
    "albert-xlarge-v1",
    "albert-xlarge-v2",
    "albert-xxlarge-v1",
    "albert-xxlarge-v2",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "bert-base-multilingual-cased",
    "bert-base-multilingual-uncased",
    "bert-base-uncased",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking",
    "bert-large-cased",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-uncased-whole-word-masking",
    "bert-large-uncased",
    "camembert-base",
    "ctrl",
    "distilbert-base-cased-distilled-squad",
    "distilbert-base-cased",
    "distilbert-base-german-cased",
    "distilbert-base-multilingual-cased",
    "distilbert-base-uncased-distilled-squad",
    "distilbert-base-uncased-finetuned-sst-2-english",
    "distilbert-base-uncased",
    "distilgpt2",
    "distilroberta-base",
    "gpt2-large",
    "gpt2-medium",
    "gpt2-xl",
    "gpt2",
    "openai-gpt",
    "roberta-base-openai-detector",
    "roberta-base",
    "roberta-large-mnli",
    "roberta-large-openai-detector",
    "roberta-large",
    "t5-11b",
    "t5-3b",
    "t5-base",
    "t5-large",
    "t5-small",
    "transfo-xl-wt103",
    "xlm-clm-ende-1024",
    "xlm-clm-enfr-1024",
    "xlm-mlm-100-1280",
    "xlm-mlm-17-1280",
    "xlm-mlm-en-2048",
    "xlm-mlm-ende-1024",
    "xlm-mlm-enfr-1024",
    "xlm-mlm-enro-1024",
    "xlm-mlm-tlm-xnli15-1024",
    "xlm-mlm-xnli15-1024",
    "xlm-roberta-base",
    "xlm-roberta-large-finetuned-conll02-dutch",
    "xlm-roberta-large-finetuned-conll02-spanish",
    "xlm-roberta-large-finetuned-conll03-english",
    "xlm-roberta-large-finetuned-conll03-german",
    "xlm-roberta-large",
    "xlnet-base-cased",
    "xlnet-large-cased",
}
SAFE_ARTIFACT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".csv",
    ".joblib",
    ".json",
    ".model",
    ".npy",
    ".npz",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".txt",
    ".ubj",
    ".yaml",
    ".yml",
}
LOAD_CALL_SUFFIXES = (
    "open",
    ".open",
    ".read_csv",
    ".read_parquet",
    ".read_pickle",
    ".read_json",
    ".read_excel",
    ".load",
    ".loadtxt",
    ".genfromtxt",
    ".load_model",
    ".read_text",
    ".read_bytes",
    ".with_name",
)
# A one-instrument frame in the shape predict() receives in phase 1: two
# visible TRAIN respondents to fit on, one held-out DEV respondent with its
# scored cells blank. The one TEST respondent the split declares is not shipped
# in phase 1, which is why the dataset has four rows and the frame has three.
# The gated item exercises the sentinel slot, which is a real answer and goes
# last.
SMOKE_SCHEMA = {
    "dataset": {"n_rows": 4, "version": "1.0", "description": "local smoke check"},
    "items": {
        "region": {"question": "Which region?", "class": "GIVEN",
                   "values": ["North", "South"], "gate": None},
        "visited_clinic": {"question": "Did you visit a clinic?", "class": "PREDICT",
                           "values": ["Yes", "No", "Prefer not to answer"], "gate": None},
        "clinic_wait": {"question": "How long did you wait?", "class": "PREDICT",
                        "values": ["Under 30 minutes", "Over 30 minutes"],
                        "gate": {"parent": "visited_clinic", "observed_if": ["Yes"]}},
    },
    "split": {"n_train": 2, "n_dev": 1, "n_test": 1},
    "gated_value": "NA_GATED",
}

# Canonical order: rows top to bottom, items in schema key order. Only the
# held-out respondent's PREDICT cells are blank, so there are two of them.
SMOKE_WIDTHS = [3, 3]      # visited_clinic, clinic_wait (+1 for NA_GATED)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_zip", type=Path)
    args = parser.parse_args()

    try:
        validate_submission_zip(args.submission_zip)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"OK: {args.submission_zip} looks like a valid submission ZIP.")
    return 0


def validate_submission_zip(zip_path: Path) -> None:
    if not zip_path.exists():
        raise ValueError(f"ZIP not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid ZIP file: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        normalized = {name.replace("\\", "/") for name in names}
        _reject_unsafe_members(normalized)

        if any(name.lower().endswith(".zip") for name in normalized):
            raise ValueError("Do not upload a ZIP that contains another ZIP. Upload the submission files directly.")

        if "main.py" not in normalized:
            nested_model = sorted(
                name for name in normalized
                if name.endswith("/main.py")
            )
            if nested_model:
                raise ValueError(
                    "main.py is nested inside a folder. Zip the contents of your submission directory, "
                    "not the directory itself."
                )
            raise ValueError("main.py must be at the ZIP root.")

        with tempfile.TemporaryDirectory(prefix="submission-check-") as tmpdir:
            zf.extractall(tmpdir)
            submission_dir = Path(tmpdir)
            _check_requirements(submission_dir)
            _check_missing_local_artifacts(submission_dir)
            _check_model(submission_dir)


def _reject_unsafe_members(names: set[str]) -> None:
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("ZIP contains an unsafe file path. Recreate it from the submission directory contents.")


def _check_missing_local_artifacts(submission_dir: Path) -> None:
    for source_path in _runtime_python_files(submission_dir):
        try:
            source = source_path.read_text(errors="replace")
            tree = ast.parse(source, filename=source_path.name)
        except (OSError, SyntaxError):
            continue
        constants = _module_string_constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _call_name(node.func) or ""
            if not _call_may_load_local_file(call_name):
                continue
            if _call_is_write_open(call_name, node, constants):
                continue
            rel_source_path = source_path.relative_to(submission_dir)
            for value in _literal_call_strings(node, constants, rel_source_path):
                missing = _missing_artifact_path(submission_dir, value)
                if missing:
                    rel_source = rel_source_path.as_posix()
                    artifact_name = Path(missing).name
                    raise ValueError(
                        f"{rel_source}:{node.lineno} references bundled file {artifact_name!r}, "
                        "but it is not in the ZIP. Add the file or update the relative path."
                    )


def _runtime_python_files(submission_dir: Path) -> list[Path]:
    parsed: dict[Path, ast.Module] = {}
    module_to_path: dict[str, Path] = {}
    for path in sorted(submission_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(errors="replace"), filename=path.name)
        except (OSError, SyntaxError):
            continue
        parsed[path] = tree
        relpath = path.relative_to(submission_dir).with_suffix("")
        parts = list(relpath.parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            module_to_path[".".join(parts)] = path

    entrypoints = [submission_dir / "main.py"]
    selected: list[Path] = []
    stack = [path for path in entrypoints if path in parsed]
    while stack:
        path = stack.pop()
        if path in selected:
            continue
        selected.append(path)
        tree = parsed[path]
        relpath = path.relative_to(submission_dir)
        for module_name in _local_import_candidates(tree, relpath):
            imported = module_to_path.get(module_name)
            if imported and imported not in selected:
                stack.append(imported)
    return selected or [path for path in entrypoints if path.exists()]


def _local_import_candidates(tree: ast.Module, relpath: Path) -> set[str]:
    visitor = _LocalImportCandidateVisitor(relpath)
    visitor.visit(tree)
    return visitor.candidates


class _LocalImportCandidateVisitor(ast.NodeVisitor):
    def __init__(self, relpath: Path):
        self.candidates: set[str] = set()
        current_module = ".".join(relpath.with_suffix("").parts)
        if current_module.endswith(".__init__"):
            self.current_package = current_module.rsplit(".", 1)[0]
        else:
            self.current_package = current_module.rsplit(".", 1)[0] if "." in current_module else ""

    def visit_If(self, node: ast.If) -> None:
        if _is_main_guard(node.test):
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.candidates.add(alias.name)
            self.candidates.add(alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            base = self.current_package.split(".") if self.current_package else []
            if node.level > len(base) + 1:
                return
            prefix_parts = base[:len(base) - node.level + 1]
            if node.module:
                prefix_parts.extend(node.module.split("."))
            prefix = ".".join(part for part in prefix_parts if part)
        else:
            prefix = node.module or ""
        if prefix:
            self.candidates.add(prefix)
        for alias in node.names:
            if prefix:
                self.candidates.add(f"{prefix}.{alias.name}")
            elif alias.name != "*":
                self.candidates.add(alias.name)


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        else:
            continue
        if isinstance(target, ast.Name) and isinstance(value, ast.Constant) and isinstance(value.value, str):
            constants[target.id] = value.value
        elif isinstance(target, ast.Name):
            constants.pop(target.id, None)
    return constants


def _call_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    if parts:
        return ".".join(reversed(parts))
    return None


def _call_may_load_local_file(call_name: str) -> bool:
    if call_name in {"open", "load_model", "read_text", "read_bytes"}:
        return True
    return any(call_name.endswith(suffix) for suffix in LOAD_CALL_SUFFIXES)


def _call_is_write_open(call_name: str, node: ast.Call, constants: dict[str, str]) -> bool:
    if not (call_name == "open" or call_name.endswith(".open")):
        return False
    mode = ""
    if len(node.args) >= 2:
        mode = _string_literal(node.args[1], constants) or ""
    for keyword in node.keywords:
        if keyword.arg == "mode":
            mode = _string_literal(keyword.value, constants) or mode
    return any(flag in mode for flag in ("w", "a", "x", "+"))


def _literal_call_strings(
    node: ast.Call,
    constants: dict[str, str],
    source_path: Path | None = None,
) -> list[str]:
    values: list[str] = []
    for arg in node.args[:2]:
        literal = _path_literal(arg, constants, source_path)
        if literal is not None:
            values.append(literal)
    for keyword in node.keywords:
        if keyword.arg in {"path", "filepath", "filename", "file", "fname"}:
            literal = _path_literal(keyword.value, constants, source_path)
            if literal is not None:
                values.append(literal)
    if isinstance(node.func, ast.Attribute):
        literal = _path_literal(node.func.value, constants, source_path)
        if literal is not None:
            values.append(literal)
    return values


def _string_literal(node: ast.AST, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value or None
    if isinstance(node, ast.Name):
        return constants.get(node.id) or None
    return None


def _path_literal(
    node: ast.AST,
    constants: dict[str, str],
    source_path: Path | None = None,
) -> str | None:
    literal = _string_literal(node, constants)
    if literal is not None:
        return literal
    if isinstance(node, ast.Call):
        call_name = _call_name(node.func) or ""
        if call_name in {"Path", "pathlib.Path"} and node.args:
            return _path_literal(node.args[0], constants, source_path)
        if call_name in {"os.path.join", "posixpath.join", "ntpath.join"}:
            parts: list[str] = []
            for arg in node.args:
                part = _path_literal(arg, constants, source_path)
                if part is None:
                    return None
                parts.append(part)
            return os.path.join(*parts) if parts else None
        if call_name.endswith(".with_name") and node.args:
            return _source_relative_with_name(
                node.func,
                _path_literal(node.args[0], constants, source_path),
                source_path,
            )
        if isinstance(node.func, ast.Attribute) and node.func.attr == "with_name" and node.args:
            return _source_relative_with_name(
                node.func,
                _path_literal(node.args[0], constants, source_path),
                source_path,
            )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _path_literal(node.left, constants, source_path)
        right = _path_literal(node.right, constants, source_path)
        if left is not None and right is not None:
            return left + right
    return None


def _source_relative_with_name(
    func_node: ast.AST,
    filename: str | None,
    source_path: Path | None,
) -> str | None:
    if filename is None:
        return None
    if not isinstance(func_node, ast.Attribute):
        return filename
    if source_path is None or not _path_expr_is_dunder_file(func_node.value):
        return filename
    source_dir = source_path.parent
    return (source_dir / filename).as_posix() if source_dir.parts else filename


def _path_expr_is_dunder_file(node: ast.AST) -> bool:
    if isinstance(node, ast.Name) and node.id == "__file__":
        return True
    if isinstance(node, ast.Call):
        call_name = _call_name(node.func) or ""
        if call_name in {"Path", "pathlib.Path"} and node.args:
            return _path_expr_is_dunder_file(node.args[0])
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"resolve", "absolute"}:
            return _path_expr_is_dunder_file(node.func.value)
    return False


def _missing_artifact_path(submission_dir: Path, value: str) -> str:
    value = (value or "").strip()
    if not value or "://" in value or value.startswith(("~", "$")):
        return ""
    if any(marker in value for marker in ("{", "}", "*", "?")):
        return ""
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return ""
    if path.name == "requirements.txt":
        return ""
    if re.search(
        r"secret|token|password|hidden|source[_-]?item|item[_-]?id|"
        r"source[_-]?id|label[_-]?id|ground[_-]?truth|answer",
        path.name,
        re.IGNORECASE,
    ):
        return ""
    if path.suffix.lower() not in SAFE_ARTIFACT_SUFFIXES:
        return ""
    candidate = (submission_dir / path).resolve()
    try:
        candidate.relative_to(submission_dir.resolve())
    except ValueError:
        return ""
    if candidate.exists():
        return ""
    return path.as_posix()


def _check_requirements(submission_dir: Path) -> None:
    requirements = submission_dir / "requirements.txt"
    nested = sorted(
        path.relative_to(submission_dir).as_posix()
        for path in submission_dir.rglob("requirements.txt")
        if path != requirements
    )
    if nested:
        raise ValueError("requirements.txt must be at the ZIP root, not inside a folder.")
    if not requirements.exists():
        return
    for lineno, raw_line in enumerate(requirements.read_text(errors="replace").splitlines(), start=1):
        line = re.sub(r"\s+#.*$", "", raw_line.strip()).strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-") or "://" in line or line.startswith(("./", "../", "/")):
            raise ValueError(
                f"requirements.txt line {lineno} is unsupported. Use named pip packages only; "
                "pip option lines, URLs, local paths, and nested requirements files are not supported."
            )
        if not re.match(
            r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:\[[^\]]+\])?(?:\s*(?:===|==|~=|!=|<=|>=|<|>|;).*)?$",
            line,
        ):
            raise ValueError(
                f"requirements.txt line {lineno} is unsupported. Use named pip packages only."
            )


def _pretrained_model_reference(node: ast.Call, constants: dict[str, str]) -> str | None:
    if node.args:
        literal = _string_literal(node.args[0], constants)
        if literal is not None:
            return literal
    for keyword in node.keywords:
        if keyword.arg in {
            "pretrained_model_name_or_path",
            "model_name_or_path",
            "model_name",
            "model",
            "model_id",
            "repo_id",
            "path",
        }:
            literal = _string_literal(keyword.value, constants)
            if literal is not None:
                return literal
    return None


def _is_local_model_reference(submission_dir: Path, model_ref: str) -> bool:
    if not model_ref or os.path.isabs(model_ref):
        return False
    if "://" in model_ref or model_ref.startswith(("~", "$")):
        return False
    candidate = (submission_dir / model_ref).resolve()
    try:
        candidate.relative_to(submission_dir.resolve())
    except ValueError:
        return False
    return candidate.exists()


def _declared_model_ref_for_call(model_ref: str, call_name: str) -> str:
    if (
        "SentenceTransformer" in call_name
        and "/" not in model_ref
        and model_ref.lower() not in SENTENCE_TRANSFORMER_BASIC_MODEL_IDS
    ):
        return f"sentence-transformers/{model_ref}"
    return model_ref


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    if not isinstance(node.ops[0], ast.Eq):
        return False
    left, right = node.left, node.comparators[0]
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    ) or (
        isinstance(right, ast.Name)
        and right.id == "__name__"
        and isinstance(left, ast.Constant)
        and left.value == "__main__"
    )


def _check_model(submission_dir: Path) -> None:
    entry = submission_dir / "main.py"
    if not entry.exists():
        entry = submission_dir / "main.py"
    model = _load_module(entry, "submission_model", submission_dir)
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise ValueError(f"{entry.name} must define callable predict(frame, schema).")

    import numpy as _np
    import pandas as _pd

    frame = _pd.DataFrame({
        "respondent_id": ["R1", "R2", "R3"],
        "region": ["North", "South", "North"],
        "visited_clinic": ["Yes", "No", _np.nan],
        "clinic_wait": ["Under 30 minutes", "NA_GATED", _np.nan],
    })
    try:
        value = predict(frame, dict(SMOKE_SCHEMA))
    except TypeError as exc:
        raise ValueError(
            "predict() must accept two positional arguments: "
            "predict(frame, schema)."
        ) from exc
    except Exception as exc:
        raise ValueError("predict() raised during the local smoke check.") from exc
    _assert_vectors(value, SMOKE_WIDTHS, "predict()")


def _load_module(path: Path, module_name: str, submission_dir: Path):
    previous_path = list(sys.path)
    sys.path.insert(0, str(submission_dir))
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not import {path.name}.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous_path


def _assert_vectors(value, widths: list[int], label: str) -> None:
    """Validate a predict() return exactly as the hosted ingestion does.

    One vector per blank cell, in canonical order, each as long as that item's
    option list — `values` plus a final slot for the gate sentinel when the item
    is gated. Probabilities must be finite and non-negative with a positive sum;
    the hosted scorer renormalizes and floors, so the local check must not be
    stricter than hosting.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{label} must return a list of probability vectors, one per blank "
            f"cell, got {type(value).__name__}."
        )
    if len(value) != len(widths):
        raise ValueError(
            f"{label} returned {len(value)} vectors for {len(widths)} blank "
            "cells. Canonical order is rows top to bottom, and within a row, "
            "items in schema key order."
        )
    for index, (vector, width) in enumerate(zip(value, widths)):
        try:
            numbers = [_assert_finite_number(p, label) for p in vector]
        except TypeError:
            raise ValueError(f"{label} vector {index} is not a sequence.") from None
        if len(numbers) != width:
            raise ValueError(
                f"{label} vector {index} has {len(numbers)} entries, but that "
                f"item has {width} options. Read the option list from the "
                "schema, not from the data — the gate sentinel gets a slot too."
            )
        if any(p < 0.0 for p in numbers):
            raise ValueError(f"{label} vector {index} contains a negative value.")
        if sum(numbers) <= 0.0:
            raise ValueError(f"{label} vector {index} sums to zero.")


def _assert_finite_number(value, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must return finite numeric probabilities.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must return finite numeric probabilities.") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must return finite numeric probabilities.")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
