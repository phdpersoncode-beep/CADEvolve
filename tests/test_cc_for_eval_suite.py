"""Pytest wrapper around the independent CC-for evaluation suite.

The gates themselves live in ``evals/cc_for``; this module turns them into
assertions over the hand-written edge cases and the checked-in fixtures so a CI
run fails on a canonicalization regression.

Gates that measure a *known* open defect are marked xfail with a reason rather
than deleted, so a fix flips them to xpass instead of going unnoticed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "dataset_utils"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from evals.cc_for import code_metrics  # noqa: E402
from evals.cc_for.harness import evaluate_program  # noqa: E402
from utils.canonicalization.cc_for import canonicalize_code  # noqa: E402

HAS_CADQUERY = importlib.util.find_spec("cadquery") is not None
CASES = REPO_ROOT / "evals" / "cc_for" / "cases"
ZERO_TO_CAD_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "zero_to_cad"

CASE_PATHS = sorted(CASES.glob("*.py"))
CASE_IDS = [path.stem for path in CASE_PATHS]

# Cases whose canonical program is currently wrong.  Each entry names the gate
# that fails and why; see docs/cc_for_eval_suite.md for the analysis.
KNOWN_FAILURES: dict[str, dict[str, str]] = {
    "augmented_assignment_offsets": {
        "canonical_executes": "AugAssign targets are not renamed with their SSA "
        "definition, so the canonical program raises NameError",
        "parameters_preserved": "same root cause: total_span is versioned on its "
        "initializer only",
    },
    "try_except_fallback": {
        "structure": "fluent chains inside try/except bodies are not lowered",
        "chains_lowered": "fluent chains inside try/except bodies are not lowered",
    },
    "self_attribute_rebuild": {
        "canonical_executes": "self.wp is copy-propagated back to the constructor "
        "argument across self.build(), so the canonical program builds nothing",
    },
    "result_read_before_alias": {
        "canonical_executes": "a bare statement reading a loop-carried `result` is "
        "not rewritten to the renamed state variable",
    },
}

# The workplane counter continues past any existing wpN, so re-running the
# converter on its own output renumbers every step.
IDEMPOTENCE_IS_BROKEN = True


def _evaluate(path: Path, **kwargs):
    return evaluate_program(path.read_text(encoding="utf-8"), name=path.stem, **kwargs)


@pytest.mark.parametrize("path", CASE_PATHS, ids=CASE_IDS)
def test_edge_case_structural_gates(path: Path) -> None:
    """Conversion, contract, parameters, loops and literals -- no CAD needed."""

    evaluation = _evaluate(path, run_geometry=False)
    expected = KNOWN_FAILURES.get(path.stem, {})
    for gate in evaluation.gates:
        if gate.skipped or gate.name == "idempotent":
            continue
        if gate.name in expected:
            assert not gate.passed, (
                f"{path.stem}: gate {gate.name!r} now passes -- remove it from "
                f"KNOWN_FAILURES ({expected[gate.name]})"
            )
            continue
        assert gate.passed, f"{path.stem}: {gate.name} failed: {gate.error or gate.detail}"


@pytest.mark.skipif(not HAS_CADQUERY, reason="CadQuery is required for geometry gates")
@pytest.mark.parametrize("path", CASE_PATHS, ids=CASE_IDS)
def test_edge_case_geometry_gates(path: Path) -> None:
    """Original and canonical programs must build the same solid."""

    evaluation = _evaluate(path)
    expected = KNOWN_FAILURES.get(path.stem, {})
    for gate in evaluation.gates:
        if gate.skipped or gate.name == "idempotent":
            continue
        if gate.name in expected:
            continue
        assert gate.passed, f"{path.stem}: {gate.name} failed: {gate.error or gate.detail}"


@pytest.mark.skipif(not HAS_CADQUERY, reason="CadQuery is required for geometry gates")
@pytest.mark.parametrize(
    "path",
    sorted(ZERO_TO_CAD_FIXTURES.glob("*.py")),
    ids=lambda path: path.stem,
)
def test_zero_to_cad_fixtures_round_trip(path: Path) -> None:
    evaluation = _evaluate(path, run_perturbations=False)
    for gate in evaluation.gates:
        if gate.skipped or gate.name == "idempotent":
            continue
        assert gate.passed, f"{path.stem}: {gate.name} failed: {gate.error or gate.detail}"


@pytest.mark.xfail(
    IDEMPOTENCE_IS_BROKEN,
    reason="the wpN counter never restarts, so a second pass renumbers every step",
    strict=True,
)
def test_canonicalization_is_idempotent() -> None:
    source = (CASES / "nested_loops_grid.py").read_text(encoding="utf-8")
    once = canonicalize_code(source).code
    twice = canonicalize_code(once).code
    assert twice == once


def test_self_attribute_copy_propagation_is_still_open() -> None:
    """Silent geometry loss: the canonical program builds an empty part.

    Nothing raises and no structural error is reported, so only comparing the
    solids finds it.  Inverted the day the propagation is made sound.
    """

    source = (
        "import cadquery as cq\n"
        "class Part:\n"
        "    def __init__(self, workplane, size):\n"
        "        self.wp = workplane\n"
        "        self.size = size\n"
        "        self.build()\n"
        "        self.model = self.wp\n"
        "    def build(self):\n"
        "        self.wp = self.wp.box(self.size, self.size, self.size)\n"
        "result = Part(cq.Workplane('XY'), 10.0).model\n"
    )
    canonical = canonicalize_code(source)
    assert canonical.report.structural_errors == []
    assert "self.model = workplane" in canonical.code

    if not HAS_CADQUERY:
        return
    namespace: dict = {}
    exec(compile(canonical.code, "<canonical>", "exec"), namespace, namespace)
    assert namespace["result"].vals() == [], "geometry loss appears to be fixed"


def test_augmented_assignment_regression_is_still_open() -> None:
    """The smallest program the SSA renamer currently miscompiles.

    Kept as an explicit assertion so the day it is fixed, this test fails loudly
    and can be inverted rather than quietly staying green.
    """

    source = (
        "import cadquery as cq\n"
        "total = 1.0\n"
        "total += 2.0\n"
        "result = cq.Workplane('XY').box(total, 2, 2)\n"
    )
    canonical = canonicalize_code(source).code
    assert "total_1 = 1.0" in canonical
    assert "total += 2.0" in canonical, "AugAssign target was left unversioned"
    with pytest.raises(NameError):
        exec(compile(canonical, "<canonical>", "exec"), {}, {})


def test_preamble_gate_ignores_docstrings_and_loop_carried_state() -> None:
    """Guard the gate itself: it must not flag a correct preamble."""

    canonical = canonicalize_code(
        (CASES / "loop_scalar_accumulator.py").read_text(encoding="utf-8")
    )
    layout = code_metrics.preamble_layout(
        canonical.code,
        exempt=frozenset(canonical.report.loop_carried_names),
    )
    assert layout.contiguous, layout.late_parameters
    assert layout.preamble_size > 0
