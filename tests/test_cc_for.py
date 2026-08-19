from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path

from dataset_utils.utils.canonicalization.cc_for import (
    CCForConfig,
    canonicalize_code,
    decompose_actions,
    validate_structure,
)
from dataset_utils.utils.canonicalization.cc_for_validation import (
    validate_parameter_perturbations,
    validate_prefixes,
    validate_round_trip,
)


FIXTURES = Path(__file__).parent / "fixtures" / "zero_to_cad"
CADEVOLVE_FIXTURES = Path(__file__).parent / "fixtures" / "cadevolve_p"
HAS_CADQUERY = importlib.util.find_spec("cadquery") is not None
ZERO_TO_CAD_FIXTURE_COUNT = 9


def assignment_counts(code: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in ast.walk(ast.parse(code)):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.For):
            targets = [node.target]
        for target in targets:
            for child in ast.walk(target):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    counts[child.id] = counts.get(child.id, 0) + 1
    return counts


class CCForStructuralTests(unittest.TestCase):
    def test_reassignments_receive_reaching_definition_names(self) -> None:
        converted = canonicalize_code(
            """
import cadquery as cq
x = 1
y = x + 1
x = 3
y = x + y
result = cq.Workplane('XY').box(x, y, 2)
"""
        )
        self.assertIn("x_1 = 1", converted.code)
        self.assertIn("y_1 = x_1 + 1", converted.code)
        self.assertIn("x_2 = 3", converted.code)
        self.assertIn("y_2 = x_2 + y_1", converted.code)
        self.assertFalse(converted.report.structural_errors)

    def test_dead_none_initializer_is_removed_only_before_overwrite(self) -> None:
        converted = canonicalize_code(
            """
import cadquery as cq
x = None
x = 4
y = None
z = y
result = cq.Workplane('XY').box(x, 2, 2)
"""
        )
        self.assertNotIn("x_1 = None", converted.code)
        self.assertIn("y = None", converted.code)
        self.assertIn("x", converted.report.removed_none_initializers)

    def test_namespace_parameters_are_flattened(self) -> None:
        converted = canonicalize_code(
            """
import cadquery as cq
from types import SimpleNamespace as Measures
params = Measures(length=10.0, width=4.0)
p = params
result = cq.Workplane('XY').box(p.length, p.width, 2)
"""
        )
        self.assertIn("length = 10.0", converted.code)
        self.assertIn("width = 4.0", converted.code)
        self.assertNotIn("p.length", converted.code)
        self.assertIn(
            "params = Measures(length=length, width=width)", converted.code
        )

    def test_nested_namespaces_become_individual_parameters(self) -> None:
        converted = canonicalize_code(
            """
import cadquery as cq
from types import SimpleNamespace as Measures
measures = Measures(
    panel=Measures(width=80.0, depth=60.0),
    hole=Measures(diameter=5.0),
)
m = measures
result = cq.Workplane('XY').box(m.panel.width, m.panel.depth, m.hole.diameter)
"""
        )
        self.assertIn("panel_width = 80.0", converted.code)
        self.assertIn("panel_depth = 60.0", converted.code)
        self.assertIn("hole_diameter = 5.0", converted.code)
        self.assertIn(
            "panel = Measures(width=panel_width, depth=panel_depth)",
            converted.code,
        )
        self.assertIn("m = measures", converted.code)
        self.assertNotIn("m.panel.width", converted.code)
        self.assertFalse(converted.report.structural_errors)

    def test_class_wrapped_model_keeps_namespace_and_lowers_method(self) -> None:
        source = """
import cadquery as cq
from types import SimpleNamespace as Measures
m = Measures(length=10.0, width=5.0, height=2.0)

class Part:
    def __init__(self, measures):
        self.m = measures
        self.model = None
        self.build()

    def build(self):
        m = self.m
        base = cq.Workplane('XY').box(m.length, m.width, m.height)
        self.model = base.edges('|Z').fillet(0.2)

result = Part(m).model
"""
        converted = canonicalize_code(source)
        self.assertIn(
            "m = Measures(length=length, width=width, height=height)",
            converted.code,
        )
        self.assertIn("def build(self):", converted.code)
        self.assertRegex(converted.code, r"wp\d+ = cq\.Workplane\('XY'\)")
        self.assertRegex(converted.code, r"wp\d+ = Part\(m\)\.model")
        self.assertFalse(converted.report.structural_errors)

    def test_geometry_returning_helper_is_lowered_at_call_site(self) -> None:
        converted = canonicalize_code(
            """
import cadquery as cq

def profile(width, height):
    wire = cq.Workplane('XY').rect(width, height)
    return wire

sketch = profile(10, 5)
result = sketch.extrude(2).edges('|Z').fillet(0.2)
"""
        )
        self.assertRegex(converted.code, r"wp\d+ = profile\(10, 5\)")
        self.assertNotIn("sketch.extrude", converted.code)
        self.assertFalse(converted.report.structural_errors)

    def test_conditional_geometry_expression_preserves_lazy_branches(self) -> None:
        converted = canonicalize_code(
            """
import cadquery as cq
base = cq.Workplane('XY').box(10, 5, 2)
optional = None
combined = base if optional is None else base.union(optional)
result = combined.edges('|Z').fillet(0.2)
"""
        )
        self.assertIn("if optional is None:", converted.code)
        self.assertNotIn("combined = base if", converted.code)
        self.assertFalse(converted.report.structural_errors)

    def test_global_result_from_class_method_gets_terminal_alias(self) -> None:
        converted = canonicalize_code(
            """
import cadquery as cq

class Part:
    def build(self):
        global result
        result = cq.Workplane('XY').box(10, 5, 2)

Part().build()
"""
        )
        self.assertIn("global result", converted.code)
        self.assertRegex(converted.code, r"result_state\w* = wp\d+")
        self.assertRegex(converted.code, r"result = wp\d+\n$")
        self.assertFalse(converted.report.structural_errors)

    def test_factory_receiver_and_lambda_calls_are_split(self) -> None:
        converted = canonicalize_code(
            """
import cadquery as cq

class Part:
    def solid(self):
        return cq.Workplane('XY').box(2, 2, 2)

seed = Part().solid()
result = cq.Workplane('XY').pushPoints([(0, 0)]).eachpoint(
    lambda loc: seed.val().located(loc)
)
"""
        )
        self.assertNotIn("Part().solid()", converted.code)
        self.assertIn("def _cc_for_lambda_1(loc):", converted.code)
        self.assertNotIn("lambda loc:", converted.code)
        self.assertFalse(converted.report.structural_errors)

    def test_preserved_loop_has_one_terminal_result(self) -> None:
        converted = canonicalize_code(
            """
import cadquery as cq
length = 20
result = cq.Workplane('XY').box(length, 10, 4)
for selector in ('>X', '<X'):
    result = result.faces(selector).workplane().hole(1)
"""
        )
        tree = ast.parse(converted.code)
        top_level_result = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "result"
        ]
        self.assertEqual(len(top_level_result), 1)
        self.assertIs(top_level_result[0], tree.body[-1])
        self.assertIn("result_state = wp2", converted.code)
        self.assertIn("for selector in", converted.code)
        self.assertFalse(converted.report.structural_errors)

    def test_static_nested_loops_unroll_to_strict_definitions(self) -> None:
        converted = canonicalize_code(
            """
import cadquery as cq
import math
dy = 2
cell_size = 3
radius = 20
points = []
for i in range(2):
    y = i * dy
    for j in range(2):
        x = j * cell_size
        z = math.sqrt(radius * radius - x * x - y * y)
        points.append((x, y, z))
result = cq.Workplane('XY').pushPoints(points).circle(1).extrude(2)
""",
            CCForConfig(loop_mode="unroll"),
        )
        self.assertFalse(any(isinstance(node, ast.For) for node in ast.walk(ast.parse(converted.code))))
        self.assertIn("i_1 = 0", converted.code)
        self.assertIn("i_2 = 1", converted.code)
        counts = assignment_counts(converted.code)
        repeated = {name: count for name, count in counts.items() if count > 1}
        self.assertEqual(repeated, {})
        self.assertFalse(converted.report.structural_errors)

    def test_plane_orientation_remains_symbolic(self) -> None:
        converted = canonicalize_code(
            """
import cadquery as cq
import math
angle_deg = 45
angle_rad = math.radians(angle_deg)
dx = math.cos(angle_rad)
dy = math.sin(angle_rad)
normal = (dx, dy, 0)
xdir = (-normal[1], normal[0], 0)
plane = cq.Plane(origin=(0, 0, 0), xDir=xdir, normal=normal)
result = cq.Workplane(plane).box(10, 5, 2)
"""
        )
        self.assertIn("math.cos(angle_rad)", converted.code)
        self.assertIn("xDir=xdir", converted.code)
        self.assertIn("normal=normal", converted.code)
        self.assertNotIn("cq.Vector(-", converted.code)

    def test_action_decomposition_groups_preamble_and_terminal(self) -> None:
        converted = canonicalize_code(
            """
import cadquery as cq
length = 10
width = 5
result = cq.Workplane('XY').box(length, width, 2)
"""
        )
        actions = decompose_actions(converted.code)
        self.assertEqual(actions[0].kind, "preamble")
        self.assertIn("length = 10", actions[0].code)
        self.assertTrue(actions[-1].code.endswith("result = wp2"))
        self.assertEqual(len(actions), 3)

    def test_all_zero_to_cad_examples_convert_in_both_loop_modes(self) -> None:
        fixtures = sorted(FIXTURES.glob("*.py"))
        self.assertEqual(len(fixtures), ZERO_TO_CAD_FIXTURE_COUNT)
        for fixture in fixtures:
            source = fixture.read_text(encoding="utf-8")
            for mode in ("preserve", "unroll"):
                with self.subTest(fixture=fixture.name, mode=mode):
                    converted = canonicalize_code(
                        source, CCForConfig(loop_mode=mode)
                    )
                    self.assertEqual(validate_structure(converted.code), [])
                    compile(converted.code, str(fixture), "exec")

    def test_cadevolve_p_and_adam_cases_convert_in_both_loop_modes(self) -> None:
        fixtures = sorted(CADEVOLVE_FIXTURES.glob("*.py"))
        self.assertEqual(len(fixtures), 3)
        for fixture in fixtures:
            source = fixture.read_text(encoding="utf-8")
            for mode in ("preserve", "unroll"):
                with self.subTest(fixture=fixture.name, mode=mode):
                    converted = canonicalize_code(
                        source,
                        CCForConfig(
                            loop_mode=mode,
                            max_unroll_iterations=64,
                        ),
                    )
                    self.assertEqual(validate_structure(converted.code), [])
                    compile(converted.code, str(fixture), "exec")

        adam = (CADEVOLVE_FIXTURES / "adam_reassignment_case.py").read_text(
            encoding="utf-8"
        )
        unrolled = canonicalize_code(adam, CCForConfig(loop_mode="unroll"))
        self.assertNotIn("for i in", unrolled.code)
        self.assertIn("i_1 = 0", unrolled.code)
        self.assertIn("i_4 = 1", unrolled.code)
        self.assertIn("x_1 =", unrolled.code)
        self.assertIn("x_8 =", unrolled.code)


@unittest.skipUnless(HAS_CADQUERY, "CadQuery is required for geometry validation")
class CCForCadQueryTests(unittest.TestCase):
    def test_round_trip_and_prefixes_on_zero_to_cad_examples(self) -> None:
        for fixture in sorted(FIXTURES.glob("*.py")):
            source = fixture.read_text(encoding="utf-8")
            with self.subTest(fixture=fixture.name, mode="preserve"):
                converted = canonicalize_code(source)
                round_trip = validate_round_trip(source, converted.code)
                self.assertTrue(round_trip.success, round_trip.to_dict())
                prefixes = validate_prefixes(converted.code)
                self.assertTrue(prefixes.success, prefixes.to_dict())

    def test_cadevolve_p_round_trip_in_both_modes(self) -> None:
        for name in ("adam_reassignment_case.py", "loft.py"):
            fixture = CADEVOLVE_FIXTURES / name
            source = fixture.read_text(encoding="utf-8")
            for mode in ("preserve", "unroll"):
                with self.subTest(fixture=name, mode=mode):
                    converted = canonicalize_code(
                        source, CCForConfig(loop_mode=mode)
                    )
                    validation = validate_round_trip(source, converted.code)
                    self.assertTrue(validation.success, validation.to_dict())

        for name in ("l_bracket.py", "patterned_plate.py"):
            fixture = FIXTURES / name
            source = fixture.read_text(encoding="utf-8")
            with self.subTest(fixture=name, mode="unroll"):
                converted = canonicalize_code(
                    source, CCForConfig(loop_mode="unroll")
                )
                round_trip = validate_round_trip(source, converted.code)
                self.assertTrue(round_trip.success, round_trip.to_dict())
                prefixes = validate_prefixes(converted.code)
                self.assertTrue(prefixes.success, prefixes.to_dict())

    def test_angle_perturbation_changes_symbolic_plane_consistently(self) -> None:
        source = """
import cadquery as cq
import math
angle_deg = 45
angle_rad = math.radians(angle_deg)
normal = (math.cos(angle_rad), math.sin(angle_rad), 0)
xdir = (-normal[1], normal[0], 0)
plane = cq.Plane(origin=(0, 0, 0), xDir=xdir, normal=normal)
result = cq.Workplane(plane).box(10, 5, 2)
"""
        converted = canonicalize_code(source)
        validation = validate_parameter_perturbations(
            source, converted, max_parameters=1
        )
        self.assertTrue(validation.success, validation.to_dict())

    def test_class_wrapper_prefixes_execute(self) -> None:
        source = """
import cadquery as cq

class Part:
    def build(self):
        base = cq.Workplane('XY').box(10, 5, 2)
        return base.edges('|Z').fillet(0.2)

result = Part().build()
"""
        converted = canonicalize_code(source)
        prefixes = validate_prefixes(converted.code)
        self.assertTrue(prefixes.success, prefixes.to_dict())


if __name__ == "__main__":
    unittest.main()
