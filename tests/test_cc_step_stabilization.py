"""Execution regressions found in the astra-and-beyond audit."""

import ast
from pathlib import Path

import pytest

from utils.canonicalization.cc_for import CCForConfig, canonicalize_code
from utils.canonicalization.cc_for_validation import (
    binarize_numeric_literals,
    validate_round_trip,
    validate_quantized_geometry,
)


@pytest.mark.parametrize("placement", ["preamble", "late"])
@pytest.mark.parametrize("count", [0, 2])
def test_scalar_binding_survives_zero_iteration_loop(placement, count):
    source = f"""import cadquery as cq
height = 3
for i in range({count}):
    height = 5
result = cq.Workplane('XY').box(10, 10, height)
"""
    converted = canonicalize_code(source, CCForConfig(parameter_placement=placement))
    assert validate_round_trip(source, converted.code).success


@pytest.mark.parametrize("placement", ["preamble", "late"])
def test_helper_reads_reassigned_global_at_call_time(placement):
    source = """import cadquery as cq
height = 3
def build():
    return cq.Workplane('XY').box(10, 10, height)
first = build()
height = 5
result = first.union(build().translate((20, 0, 0)))
"""
    converted = canonicalize_code(source, CCForConfig(parameter_placement=placement))
    assert validate_round_trip(source, converted.code).success


def test_round_trip_detects_dropped_second_stack_body():
    source = """import cadquery as cq
left = cq.Workplane('XY').box(2, 3, 4)
right = cq.Workplane('XY').box(5, 6, 7).translate((20, 0, 0))
result = left.add(right)
"""
    broken = source.replace("result = left.add(right)", "result = left")
    assert validate_round_trip(source, source).success
    assert not validate_round_trip(source, broken).success


@pytest.mark.parametrize("method", ["shell", "extrude"])
@pytest.mark.parametrize("value", [-1, -0.4, 0.4])
def test_binarizer_preserves_direction_of_signed_lengths(method, value):
    code = f"result = wp.{method}({value})"
    quantized = binarize_numeric_literals(code)
    argument = ast.parse(quantized).body[0].value.args[0]
    assert ast.literal_eval(argument) == (-1 if value < 0 else 1)


def test_binarizer_preserves_inward_shell_geometry():
    source = """import cadquery as cq
result = cq.Workplane('XY').box(10, 10, 10).faces('>Z').shell(-1)
"""
    assert validate_round_trip(source, binarize_numeric_literals(source)).success


@pytest.mark.parametrize("tail", [
    "result = cq.Workplane('XY', origin=(10, 0, 0)).box(2, 3, 4)",
    "result = cq.Workplane('XY', (10, 0, 0)).box(2, 3, 4)",
    "result = cq.Workplane('XY').box(2, 3, 4)\nunused = cq.Workplane('XY').box(8, 9, 10)",
    "result = cq.Workplane('XY').box(2, 3, 4).val().translate((10, 0, 0))",
    "result = cq.Workplane('XY').box(2, 3, 4)\nprint('built')",
])
def test_tracer_preserves_constructor_and_actual_result(tail):
    from evals.cc_for.representations import build_cadevolve_c

    source = "import cadquery as cq\n" + tail + "\n"
    assert validate_round_trip(source, build_cadevolve_c(source)).success


def test_boolean_comparison_detects_different_holes_with_equal_signatures():
    source = """import cadquery as cq
result = (cq.Workplane('XY').box(20, 20, 4).faces('>Z').workplane()
          .pushPoints([(4, 4), (-4, -4)]).hole(2))
"""
    different = source.replace("[(4, 4), (-4, -4)]", "[(4, -4), (-4, 4)]")
    assert validate_round_trip(source, different).success  # signature collision
    assert not validate_round_trip(source, different, check_boolean=True).success
    assert validate_round_trip(source, source, check_boolean=True).success


def test_eval_executes_dataclass_with_postponed_annotations():
    from evals.cc_for.harness import execute

    source = """from __future__ import annotations
from dataclasses import dataclass
@dataclass
class Measures:
    height: float = 3
result = Measures().height
"""
    assert execute(source)["result"] == 3


def test_quantization_gate_rejects_documented_derived_parameter_damage():
    path = Path(__file__).parents[1] / "evals/cc_for/cases/derived_parameter_chain.py"
    source = path.read_text()
    code = canonicalize_code(source, CCForConfig(parameter_placement="late")).code
    assert not validate_quantized_geometry(source, code).success


def test_quantization_gate_attributes_source_failure():
    source = "import cadquery as cq\nresult = cq.Workplane('XY').box(10 * 0.4, 2, 2)\n"
    checked = validate_quantized_geometry(source, source)
    assert not checked.success
    assert checked.error.startswith("binarized source failed")


def test_tracer_comparison_cannot_hide_a_translation(monkeypatch):
    from evals.cc_for import representations as reps

    source = "import cadquery as cq\nresult = cq.Workplane('XY').box(2, 3, 4)\n"
    translated = source + "result = result.translate((10, 0, 0))\n"
    monkeypatch.setattr(reps, "build_cadevolve_c", lambda *args, **kwargs: translated)
    checked = reps.compare_representations(
        source, include=(reps.SOURCE, reps.CADEVOLVE_C),
        pairs=((reps.CADEVOLVE_C, reps.SOURCE),), surface_points=256,
    )
    assert not checked.passed


@pytest.mark.parametrize("placement", ["preamble", "late"])
@pytest.mark.parametrize("body", [
    "try:\n    result: object = cq.Workplane('XY').box(10, 10, 5)\nexcept ValueError:\n    result = cq.Workplane('XY').box(5, 5, 5)",
    "result = cq.Workplane('XY').box(10, 10, 5)\nfor i in range(2):\n    result: object = result.faces('>Z').workplane().circle(2).extrude(1)",
])
def test_annotated_geometry_writes_runtime_state(placement, body):
    source = "import cadquery as cq\n" + body + "\n"
    converted = canonicalize_code(source, CCForConfig(parameter_placement=placement))
    assert not converted.report.structural_errors
    assert validate_round_trip(source, converted.code, check_boolean=True).success


def test_annotated_attribute_evaluates_its_target_only_once():
    source = """import cadquery as cq
class Holder:
    pass
holder = Holder()
calls = []
def target():
    calls.append(1)
    return holder
target().shape: object = cq.Workplane('XY').box(2, 3, 4)
result = holder.shape.translate((len(calls) * 10, 0, 0))
"""
    converted = canonicalize_code(source, CCForConfig(parameter_placement="late"))
    assert validate_round_trip(source, converted.code, check_boolean=True).success
