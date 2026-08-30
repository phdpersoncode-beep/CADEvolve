"""Pytest wrapper around the independent CC-for evaluation suite.

The gates themselves live in ``evals/cc_for``; this module turns them into
assertions over the hand-written edge cases and the checked-in fixtures so a CI
run fails on a canonicalization regression. Idempotency remains a diagnostic
measurement rather than part of the one-pass conversion contract.
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

# Idempotency is diagnostic only: the converter accepts source programs and is
# intentionally not required to recognize or preserve its own lowered output.
NON_CONTRACT_GATES = {"idempotent"}

# One case still fails a contract gate.  Recorded explicitly, so a fix flips it
# to a failure here rather than passing unnoticed.
KNOWN_FAILURES: dict[str, dict[str, str]] = {
    "loop_over_geometry_iterable": {
        "structure": "a fluent chain in a `for` header is not lowered; the loop "
        "body and the header's names are, the iterable expression is not",
    },
}


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
        if gate.skipped or gate.name in NON_CONTRACT_GATES:
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
        if gate.skipped or gate.name in NON_CONTRACT_GATES or gate.name in expected:
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
        if gate.skipped or gate.name in NON_CONTRACT_GATES:
            continue
        assert gate.passed, (
            f"{path.stem} [{placement}]: {gate.name} failed: "
            f"{gate.error or gate.detail}"
        )


def test_opaque_builder_call_invalidates_attribute_aliases() -> None:
    """A builder mutation must be read back from the object, not a stale alias."""

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
    assert "self.model = workplane" not in canonical.code
    assert "self.model = self.wp" in canonical.code

    if not HAS_CADQUERY:
        return
    namespace: dict = {}
    exec(compile(canonical.code, "<canonical>", "exec"), namespace, namespace)
    assert namespace["result"].val().Volume() == pytest.approx(1000.0)


def test_custom_workplane_extension_keeps_receiver_alias() -> None:
    """A modeled Workplane extension must not invalidate its receiver name."""

    source = (
        "import cadquery as cq\n"
        "def firstSolid(self):\n"
        "    return self.newObject([self.findSolid()])\n"
        "cq.Workplane.firstSolid = firstSolid\n"
        "result = cq.Workplane('XY').box(10.0, 10.0, 10.0)\n"
        "result = result.firstSolid().edges('|Z').fillet(1.0)\n"
        "result = result.firstSolid().faces('>Z').workplane().hole(2.0)\n"
    )
    canonical = canonicalize_code(source)
    assert canonical.report.structural_errors == []
    assert canonical.code.count("result =") == 1

    if not HAS_CADQUERY:
        return
    source_namespace: dict = {}
    canonical_namespace: dict = {}
    exec(compile(source, "<source>", "exec"), source_namespace, source_namespace)
    exec(
        compile(canonical.code, "<canonical>", "exec"),
        canonical_namespace,
        canonical_namespace,
    )
    assert canonical_namespace["result"].val().Volume() == pytest.approx(
        source_namespace["result"].val().Volume()
    )


def test_augmented_assignment_updates_its_reaching_definition() -> None:
    """AugAssign must read and update the currently versioned scalar."""

    source = (
        "import cadquery as cq\n"
        "total = 1.0\n"
        "total += 2.0\n"
        "result = cq.Workplane('XY').box(total, 2, 2)\n"
    )
    canonical = canonicalize_code(source).code
    assert "total_1 = 1.0" in canonical
    assert "total_1 += 2.0" in canonical
    if not HAS_CADQUERY:
        return
    namespace: dict = {}
    exec(compile(canonical, "<canonical>", "exec"), namespace, namespace)
    assert namespace["total_1"] == 3.0
    assert namespace["result"].val().Volume() == pytest.approx(12.0)


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
