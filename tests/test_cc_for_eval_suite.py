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
from utils.canonicalization.cc_for import (  # noqa: E402
    canonicalize_code,
    decompose_actions,
    join_actions,
    validate_structure,
)

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
    "loop_over_geometry_iterable": {
        "structure": "a fluent chain in a `for` header is not lowered; the loop "
        "body and the header's names are, the iterable expression is not",
    },
}

# The workplane counter continues past any existing wpN, so re-running the
# converter on its own output renumbers every step.
IDEMPOTENCE_IS_BROKEN = True


def _evaluate(path: Path, **kwargs):
    return evaluate_program(path.read_text(encoding="utf-8"), name=path.stem, **kwargs)


@pytest.mark.parametrize("placement", ("preamble", "late"))
@pytest.mark.parametrize("path", CASE_PATHS, ids=CASE_IDS)
def test_edge_case_structural_gates(path: Path, placement: str) -> None:
    """Conversion, contract, parameters, loops and literals -- no CAD needed.

    Run for both representations: CC-for and CC-step share every stage except
    parameter placement, so a defect in one is expected in the other, and a gate
    that only holds for one of them is a placement bug.
    """

    evaluation = _evaluate(path, run_geometry=False, parameter_placement=placement)
    expected = KNOWN_FAILURES.get(path.stem, {})
    for gate in evaluation.gates:
        if gate.skipped or gate.name == "idempotent":
            continue
        if gate.name in expected:
            assert not gate.passed, (
                f"{path.stem} [{placement}]: gate {gate.name!r} now passes -- "
                f"remove it from KNOWN_FAILURES ({expected[gate.name]})"
            )
            continue
        assert gate.passed, (
            f"{path.stem} [{placement}]: {gate.name} failed: "
            f"{gate.error or gate.detail}"
        )


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
@pytest.mark.parametrize("placement", ("preamble", "late"))
@pytest.mark.parametrize(
    "path",
    sorted(ZERO_TO_CAD_FIXTURES.glob("*.py")),
    ids=lambda path: path.stem,
)
def test_zero_to_cad_fixtures_round_trip(path: Path, placement: str) -> None:
    evaluation = _evaluate(
        path, run_perturbations=False, parameter_placement=placement
    )
    for gate in evaluation.gates:
        if gate.skipped or gate.name == "idempotent":
            continue
        assert gate.passed, (
            f"{path.stem} [{placement}]: {gate.name} failed: "
            f"{gate.error or gate.detail}"
        )


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


def test_late_layout_gate_flags_a_parameter_that_could_sink() -> None:
    """Guard the CC-step gate itself: it has to be able to fail.

    This is CC-step output with one parameter dragged back above the step before
    the one that reads it, which is exactly the defect the gate exists to catch.
    """

    canonical = """
import cadquery as cq
plate = 40.0
hole = 4.0
wp1 = cq.Workplane('XY')
wp2 = wp1.box(plate, plate, 6.0)
depth = 2.0
wp3 = wp2.faces('>Z')
wp4 = wp3.hole(hole, depth)
result = wp4
"""
    layout = code_metrics.late_parameter_layout(canonical)
    assert not layout.grouped
    assert layout.early_parameters == ("hole",)

    moved = canonical.replace("hole = 4.0\n", "").replace(
        "depth = 2.0", "hole = 4.0\ndepth = 2.0"
    )
    fixed = code_metrics.late_parameter_layout(moved)
    assert fixed.grouped, fixed.early_parameters
    assert fixed.group_count == 2


def test_loop_binding_gate_flags_an_accumulator_that_stopped_being_written() -> None:
    """Guard the new gate: it has to be able to fail, and not on ordinary output.

    Deleting the write-back is exactly the defect it exists to catch, and it is
    invisible to every other structural check -- the program still parses,
    compiles, keeps one terminal ``result`` and lowers every chain.
    """

    source = (CASES / "while_carried_union_accumulator.py").read_text(
        encoding="utf-8"
    )
    canonical = canonicalize_code(source).code
    assert code_metrics.dropped_loop_bindings(source, canonical) == []
    assert validate_structure(canonical) == []

    broken = "\n".join(
        line
        for line in canonical.splitlines()
        if not line.strip().startswith("pegs = wp")
    )
    assert validate_structure(broken) == []
    assert code_metrics.dropped_loop_bindings(source, broken) == ["pegs"]


def test_action_reassembly_gate_flags_a_lost_separator() -> None:
    """Guard the other new gate: a plain join is what it has to reject."""

    source = """
import cadquery as cq
wall = 3.0

def rib(wp, width):
    return wp.faces('>Z').workplane().rect(width, wall).extrude(4.0)
plate = cq.Workplane('XY').box(40.0, 40.0, 4.0)
result = rib(plate, 8.0)
"""
    canonical = canonicalize_code(source).code
    actions = decompose_actions(canonical)
    assert join_actions(actions) == canonical
    assert "\n".join(action.code for action in actions) != canonical


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
