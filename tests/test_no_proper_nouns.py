"""The core package must not know any model, site, or dataset by name.

``stages/preprocess.py`` states the rule in prose already: "No model appears by
name anywhere below. Registering a model is adding one JSON card to
``registry/models/``; a branch on ``model_id`` here would falsify it." Nothing
enforced it, and the audit in 1b56296 found five hardcoded values that a human
review had already passed over. This test is that enforcement.

It deliberately inspects **identifiers and string values only**. A comment or
docstring naming MIMIC or ECGFounder explains *why* a decision was made and is
worth keeping; ``gates.py`` needs to say which corpus motivated the
contamination check. The smell is a proper noun the code can *act on* -- a
branch on a model id, a hardcoded path, a lookup keyed by site.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: Model, dataset, site and vendor names. Lowercased before matching.
FORBIDDEN = (
    "mimic",
    "ecgfounder",
    "prophecg",
    "stemi",
    "ptbxl",
    "ptb_xl",
    "dryou",
    "dicom-miva",
    "dicom_miva",
)

CORE = Path(__file__).resolve().parent.parent / "src" / "mival"


def _source_files():
    return sorted(p for p in CORE.rglob("*.py") if "__pycache__" not in p.parts)


def _docstring_nodes(tree: ast.AST) -> set:
    """Every string node that serves as a module, class or function docstring."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    out.add(id(body[0].value))
    return out


def _offences(path: Path):
    """(line, kind, text) for every identifier or string value naming a proper noun."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = _docstring_nodes(tree)
    found = []

    def check(line, kind, text):
        low = text.lower()
        hits = [w for w in FORBIDDEN if w in low]
        if hits:
            found.append((line, kind, text if len(text) < 70 else text[:67] + "..."))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                check(node.lineno, "string", node.value)
        elif isinstance(node, ast.Name):
            check(node.lineno, "name", node.id)
        elif isinstance(node, ast.Attribute):
            check(node.lineno, "attribute", node.attr)
        elif isinstance(node, ast.arg):
            check(node.lineno, "argument", node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            check(node.lineno, "definition", node.name)
        elif isinstance(node, ast.keyword) and node.arg:
            check(node.lineno, "keyword", node.arg)
    return found


#: Offences that exist today, each with the reason it is still here. The gate
#: stays green so that a *new* one is visible immediately, and the debt stays
#: visible so that it is not mistaken for a clean bill of health. Removing an
#: entry from here is the fix; adding one requires a reason a reviewer accepts.
KNOWN_EXCEPTIONS = {
    ("stages/evaluate.py", "stemi"): (
        "OUTCOME_DIAGNOSTIC is the default value of the `outcome` report axis. "
        "It names this study's disease, so a study validating any other outcome "
        "still gets 'stemi' in its result tables. Should become a neutral "
        "sentinel ('primary') or be read from the evaluation spec."
    ),
    ("stages/misclassify.py", "stemi_mimic"): (
        "Half false positive: 'mimic' here is the clinical term (a pattern that "
        "mimics STEMI), not the dataset. The 'stemi' half is real -- "
        "DEFAULT_VERDICTS is a disease-specific vocabulary living in core code. "
        "The docstring already says a study declares its own with "
        "`misclassify.verdicts`, so the default is a documented choice rather "
        "than a hidden one."
    ),
}


def _unknown(path, offences):
    rel = str(path.relative_to(CORE))
    return [o for o in offences if (rel, o[2]) not in KNOWN_EXCEPTIONS]


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: str(p.relative_to(CORE)))
def test_core_module_names_no_model_site_or_dataset(path):
    offences = _unknown(path, _offences(path))
    assert not offences, "\n".join(
        f"{path.relative_to(CORE)}:{line} {kind} {text!r} names a model, site or dataset; "
        "it belongs in a ModelCard, study.yaml or configs/. If it must stay, add it to "
        "KNOWN_EXCEPTIONS with a reason."
        for line, kind, text in offences
    )


def test_the_gate_can_actually_fail():
    """A gate that cannot fail is not a gate."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write('MODEL = "ecgfounder"\n')
        probe = Path(fh.name)
    try:
        assert _offences(probe), "the detector missed a hardcoded model id"
    finally:
        probe.unlink()


def test_prose_is_allowed():
    """Comments and docstrings may name anything; they carry the reasoning."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write('"""Why this matters for ECGFounder on MIMIC."""\n# STEMI too\nX = 1\n')
        probe = Path(fh.name)
    try:
        assert not _offences(probe), "prose must not trip the gate"
    finally:
        probe.unlink()


def test_known_exceptions_are_still_real():
    """A stale allowlist hides the fact that a debt was already paid."""
    seen = set()
    for path in _source_files():
        rel = str(path.relative_to(CORE))
        for _, _, text in _offences(path):
            seen.add((rel, text))
    stale = sorted(set(KNOWN_EXCEPTIONS) - seen)
    assert not stale, (
        f"these KNOWN_EXCEPTIONS no longer occur and should be deleted: {stale}"
    )
