"""CC-step: the CC-for converter with each parameter group placed at its step.

These tests cover the placement itself -- which statements move, where they land,
and that moving them cannot change what the program builds.  The gates shared
with CC-for live in ``test_cc_for.py`` and ``test_cc_for_eval_suite.py``.
"""

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
    validate_prefixes,
    validate_round_trip,
)
from evals.cc_for.code_metrics import late_parameter_layout, parameter_groups

FIXTURES = Path(__file__).parent / "fixtures" / "zero_to_cad"
CADEVOLVE_FIXTURES = Path(__file__).parent / "fixtures" / "cadevolve_p"
CASES = Path(__file__).parents[1] / "evals" / "cc_for" / "cases"
HAS_CADQUERY = importlib.util.find_spec("cadquery") is not None

# Programs whose CC-for conversion is already known to be wrong; CC-step inherits
# the defect because it shares every stage but parameter placement.  See
# tests/test_cc_for_eval_suite.py for the analysis of each.
KNOWN_BAD_CONVERSIONS = {
    "augmented_assignment_offsets",
    "result_read_before_alias",
    "self_attribute_rebuild",
}


def convert(source: str, placement: str = "late"):
    return canonicalize_code(source, CCForConfig(parameter_placement=placement))


def top_level_names(code: str) -> list[str]:
    """Names assigned by top-level simple assignments, in emitted order."""

    names: list[str] = []
    for stmt in ast.parse(code).body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            names.append(stmt.targets[0].id)
    return names


class LatePlacementTests(unittest.TestCase):
    def test_each_parameter_group_precedes_the_step_that_uses_it(self) -> None:
        source = """
import cadquery as cq
length = 80
width = 30
height = 10
hole_diameter = 5
base = cq.Workplane('XY').box(length, width, height)
result = base.faces('>Z').workplane().hole(hole_diameter)
"""
        converted = convert(source)
        names = top_level_names(converted.code)
        # The box dimensions introduce the first step; the hole diameter is not
        # read until the second, so it must not appear before the first wp line.
        self.assertLess(names.index("height"), names.index("wp1"))
        self.assertGreater(names.index("hole_diameter"), names.index("wp1"))
        self.assertEqual(
            converted.report.parameter_groups,
            [["length", "width", "height"], ["hole_diameter"]],
        )

    def test_cc_for_keeps_one_preamble_for_the_same_program(self) -> None:
        source = """
import cadquery as cq
length = 80
width = 30
height = 10
hole_diameter = 5
base = cq.Workplane('XY').box(length, width, height)
result = base.faces('>Z').workplane().hole(hole_diameter)
"""
        names = top_level_names(convert(source, "preamble").code)
        self.assertLess(names.index("hole_diameter"), names.index("wp1"))

    def test_derived_parameters_travel_with_the_value_they_feed(self) -> None:
        source = """
import cadquery as cq
plate = 40
margin = 4
inset = plate / 2 - margin
body = cq.Workplane('XY').box(plate, plate, 5)
result = body.faces('>Z').workplane().rect(inset, inset).cutBlind(-2)
"""
        names = top_level_names(convert(source).code)
        # margin is only read by inset, so it belongs to inset's group.
        for name in ("margin", "inset"):
            self.assertGreater(names.index(name), names.index("wp1"), name)
        self.assertLess(names.index("margin"), names.index("inset"))

    def test_parameter_read_inside_a_helper_stays_above_its_definition(self) -> None:
        source = """
import cadquery as cq
hub_radius = 9.0

def blade(width):
    return cq.Workplane('XY').box(width, hub_radius, 2.0)
blade_width = 3.0
result = blade(blade_width)
"""
        converted = convert(source)
        body = ast.parse(converted.code).body
        names = [
            stmt.name if isinstance(stmt, ast.FunctionDef) else None for stmt in body
        ]
        definition = names.index("blade")
        assigned = top_level_names(converted.code)
        # A function body's reads are invisible to a scoped analysis; sinking
        # hub_radius past the def would leave it unbound when blade() runs.
        self.assertIn("hub_radius", assigned)
        self.assertLess(
            [
                index
                for index, stmt in enumerate(body)
                if isinstance(stmt, ast.Assign)
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "hub_radius"
            ][0],
            definition,
        )

    def test_loop_parameters_group_above_the_loop(self) -> None:
        source = """
import cadquery as cq
count = 4
pitch = 6.0
stud = 1.5
base = cq.Workplane('XY').box(40, 10, 4)
points = []
for i in range(count):
    points.append((i * pitch - 9.0, 0))
result = base.faces('>Z').workplane().pushPoints(points).hole(stud)
"""
        names = top_level_names(convert(source).code)
        self.assertGreater(names.index("count"), names.index("wp1"))
        self.assertGreater(names.index("stud"), names.index("points"))

    def test_a_redefined_name_is_never_repositioned(self) -> None:
        source = """
import cadquery as cq
size = 10
part = cq.Workplane('XY').box(size, size, size)
size = 4
result = part.faces('>Z').workplane().hole(size)
"""
        converted = canonicalize_code(
            source,
            CCForConfig(parameter_placement="late", version_reassignments=False),
        )
        names = top_level_names(converted.code)
        # Both definitions of `size` keep their source position relative to the
        # steps between them, or the second would reach the first step's box().
        self.assertEqual([name for name in names if name == "size"], ["size", "size"])
        self.assertLess(names.index("size"), names.index("wp1"))

    def test_bare_math_imports_are_parameter_algebra(self) -> None:
        """``radians(a)`` has to count as data just as ``math.radians(a)`` does.

        Otherwise the first derived angle looks like the start of modelling and
        every parameter behind it is stuck where the source happened to put it.
        """

        source = """
import cadquery as cq
from math import cos, radians
span = 40.0
angle_deg = 30.0
reach = span * cos(radians(angle_deg))
plate = cq.Workplane('XY').box(span, span, 4.0)
result = plate.faces('>Z').workplane().circle(reach / 8).cutThruAll()
"""
        converted = convert(source)
        names = top_level_names(converted.code)
        for name in ("angle_deg", "reach"):
            self.assertIn(name, converted.report.hoisted_parameters, name)
            self.assertGreater(names.index(name), names.index("wp1"), name)

    def test_source_position_fallback_stays_valid(self) -> None:
        """The escape hatch has to recover a program, not just a warning.

        Placement can only be wrong in one direction -- emitting a read before
        its binding -- so the converter checks for that and, if it ever sees it,
        sends every parameter to the step that follows it in the source instead.
        """

        from dataset_utils.utils.canonicalization import cc_for as module

        source = (FIXTURES / "shelled_pocket.py").read_text(encoding="utf-8")
        original = module._defines_before_use
        module._defines_before_use = lambda *args, **kwargs: False
        try:
            converted = convert(source)
        finally:
            module._defines_before_use = original

        assert any(
            "source position" in warning for warning in converted.report.warnings
        ), converted.report.warnings
        self.assertEqual(converted.report.structural_errors, [])
        names = top_level_names(converted.code)
        self.assertLess(names.index("outer_width"), names.index("wp1"))
        compile(converted.code, "<fallback>", "exec")

    def test_structure_contract_holds(self) -> None:
        for fixture in sorted(FIXTURES.glob("*.py")):
            with self.subTest(fixture=fixture.name):
                converted = convert(fixture.read_text(encoding="utf-8"))
                self.assertEqual(converted.report.structural_errors, [])
                self.assertEqual(validate_structure(converted.code), [])

    def test_no_parameter_could_sink_further(self) -> None:
        sources = sorted(FIXTURES.glob("*.py")) + sorted(CASES.glob("*.py"))
        for fixture in sources:
            with self.subTest(fixture=fixture.name):
                converted = convert(fixture.read_text(encoding="utf-8"))
                layout = late_parameter_layout(
                    converted.code,
                    exempt=frozenset(converted.report.loop_carried_names),
                )
                self.assertEqual(layout.early_parameters, ())

    def test_placement_actually_splits_the_preamble(self) -> None:
        """CC-step has to differ from CC-for, not merely be allowed to."""

        split = 0
        for fixture in sorted(FIXTURES.glob("*.py")):
            source = fixture.read_text(encoding="utf-8")
            late = late_parameter_layout(convert(source).code)
            if late.group_count > 1:
                split += 1
            self.assertGreaterEqual(late.group_count, 1, fixture.name)
        self.assertGreaterEqual(split, 8, "late placement barely moved anything")

    def test_both_placements_move_exactly_the_same_parameters(self) -> None:
        """One predicate classifies parameters; only the destination differs.

        A name a second statement also binds is not a parameter for either of
        them -- CC-for would hoist it above that binding and CC-step would sink
        it below -- so the two lists are equal rather than merely nested.
        """

        for fixture in sorted(FIXTURES.glob("*.py")) + sorted(CASES.glob("*.py")):
            with self.subTest(fixture=fixture.name):
                source = fixture.read_text(encoding="utf-8")
                hoisted = set(convert(source, "preamble").report.hoisted_parameters)
                sunk = set(convert(source, "late").report.hoisted_parameters)
                self.assertEqual(sunk, hoisted)

    def test_a_conditionally_rebound_accumulator_never_moves(self) -> None:
        """Neither placement may carry an initializer across its own loop.

        Nothing in the loop *reads* ``tallest``, so the loop does not look like a
        reader and sinking would put the initializer below it, overwriting what
        the loop computed.  Running the same program with the reset after the
        loop pins the other direction, which is the one that catches hoisting.
        """

        sunk_below = """
import cadquery as cq
plate = 40.0
thickness = 4.0
heights = [1.0, 6.0, 2.0]
threshold = 3.0
part = cq.Workplane('XY').box(plate, plate, thickness)
tallest = 0.5
for height in heights:
    if height > threshold:
        tallest = height
result = part.faces('>Z').workplane().circle(6.0).extrude(tallest)
"""
        hoisted_above = """
import cadquery as cq
plate = 40.0
thickness = 4.0
heights = [1.0, 6.0, 2.0]
threshold = 3.0
part = cq.Workplane('XY').box(plate, plate, thickness)
for height in heights:
    if height > threshold:
        tallest = height
tallest = 0.5
result = part.faces('>Z').workplane().circle(6.0).extrude(tallest)
"""
        for label, source in (("sunk", sunk_below), ("hoisted", hoisted_above)):
            for placement in ("preamble", "late"):
                with self.subTest(case=label, placement=placement):
                    converted = convert(source, placement)
                    self.assertNotIn(
                        "tallest", converted.report.hoisted_parameters
                    )
                    body = ast.parse(converted.code).body
                    loop = next(
                        index
                        for index, stmt in enumerate(body)
                        if isinstance(stmt, ast.For)
                    )
                    initializer = next(
                        index
                        for index, stmt in enumerate(body)
                        if isinstance(stmt, ast.Assign)
                        and isinstance(stmt.targets[0], ast.Name)
                        and stmt.targets[0].id == "tallest"
                        and isinstance(stmt.value, ast.Constant)
                    )
                    if label == "sunk":
                        self.assertLess(initializer, loop)
                    else:
                        self.assertGreater(initializer, loop)

    def test_a_parameter_sinks_past_a_loop_that_does_not_read_it(self) -> None:
        """The refusal above is about rebinding, not about loops in general."""

        source = """
import cadquery as cq
bar_len = 70.0
bar_wid = 18.0
bar_thk = 6.0
groove_count = 3
notch_count = 2
notch_depth = 1.5
bar = cq.Workplane('XY').box(bar_len, bar_wid, bar_thk)
for groove in range(groove_count):
    bar = bar.faces('>Z').workplane().center(groove * 14.0 - 14.0, 0).rect(3.0, bar_wid).cutBlind(-1.0)
for notch in range(notch_count):
    bar = bar.faces('>Z').workplane().center(notch * 20.0 - 10.0, 0).circle(2.0).cutBlind(-notch_depth)
result = bar
"""
        body = ast.parse(convert(source).code).body
        loops = [
            index for index, stmt in enumerate(body) if isinstance(stmt, ast.For)
        ]
        depth = next(
            index
            for index, stmt in enumerate(body)
            if isinstance(stmt, ast.Assign)
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == "notch_depth"
        )
        self.assertEqual(len(loops), 2)
        self.assertGreater(depth, loops[0])
        self.assertLess(depth, loops[1])

    def test_actions_carry_their_parameter_group(self) -> None:
        source = """
import cadquery as cq
length = 80
width = 30
height = 10
hole_diameter = 5
base = cq.Workplane('XY').box(length, width, height)
result = base.faces('>Z').workplane().hole(hole_diameter)
"""
        converted = convert(source)
        actions = decompose_actions(converted.code, parameter_placement="late")
        self.assertEqual(actions[0].kind, "preamble")
        self.assertEqual(actions[0].code, "import cadquery as cq")
        self.assertIn("length = 80", actions[1].code)
        self.assertIn("wp1 = cq.Workplane('XY')", actions[1].code)
        self.assertTrue(
            any("hole_diameter = 5" in action.code for action in actions[2:])
        )
        # Every action still runs as a prefix of the ones before it.
        self.assertEqual(
            "\n".join(action.code for action in actions).strip(),
            converted.code.strip(),
        )


@unittest.skipUnless(HAS_CADQUERY, "CadQuery is required for geometry validation")
class LatePlacementGeometryTests(unittest.TestCase):
    def test_round_trip_and_prefixes_on_zero_to_cad_examples(self) -> None:
        for fixture in sorted(FIXTURES.glob("*.py")):
            source = fixture.read_text(encoding="utf-8")
            with self.subTest(fixture=fixture.name):
                converted = convert(source)
                round_trip = validate_round_trip(source, converted.code)
                self.assertTrue(round_trip.success, round_trip.to_dict())
                prefixes = validate_prefixes(
                    converted.code, parameter_placement="late"
                )
                self.assertTrue(prefixes.success, prefixes.to_dict())

    def test_source_position_fallback_builds_the_same_solid(self) -> None:
        from dataset_utils.utils.canonicalization import cc_for as module

        source = (FIXTURES / "shelled_pocket.py").read_text(encoding="utf-8")
        original = module._defines_before_use
        module._defines_before_use = lambda *args, **kwargs: False
        try:
            converted = convert(source)
        finally:
            module._defines_before_use = original
        round_trip = validate_round_trip(source, converted.code)
        self.assertTrue(round_trip.success, round_trip.to_dict())

    def test_round_trip_on_cadevolve_p_examples(self) -> None:
        for fixture in sorted(CADEVOLVE_FIXTURES.glob("*.py")):
            source = fixture.read_text(encoding="utf-8")
            with self.subTest(fixture=fixture.name):
                converted = convert(source)
                round_trip = validate_round_trip(source, converted.code)
                self.assertTrue(round_trip.success, round_trip.to_dict())

    def test_edge_cases_build_the_source_solid(self) -> None:
        for fixture in sorted(CASES.glob("*.py")):
            if fixture.stem in KNOWN_BAD_CONVERSIONS:
                continue
            source = fixture.read_text(encoding="utf-8")
            with self.subTest(fixture=fixture.name):
                round_trip = validate_round_trip(source, convert(source).code)
                self.assertTrue(round_trip.success, round_trip.to_dict())

    def test_known_bad_conversions_are_no_worse_than_cc_for(self) -> None:
        """A CC-for defect must not be reported as a CC-step regression."""

        for name in sorted(KNOWN_BAD_CONVERSIONS):
            source = (CASES / f"{name}.py").read_text(encoding="utf-8")
            with self.subTest(fixture=name):
                self.assertFalse(
                    validate_round_trip(source, convert(source, "preamble").code).success,
                    f"{name} now converts under CC-for; drop it from "
                    "KNOWN_BAD_CONVERSIONS",
                )
                self.assertFalse(
                    validate_round_trip(source, convert(source, "late").code).success
                )


@unittest.skipUnless(HAS_CADQUERY, "CadQuery is required for geometry comparison")
class RepresentationAgreementTests(unittest.TestCase):
    """CADEvolve-C, CC-for and CC-step must describe the same solid."""

    def test_zero_to_cad_fixtures_agree_across_representations(self) -> None:
        from evals.cc_for.representations import compare_representations

        for fixture in sorted(FIXTURES.glob("*.py")):
            with self.subTest(fixture=fixture.name):
                evaluation = compare_representations(
                    fixture.read_text(encoding="utf-8"),
                    name=fixture.stem,
                    require_cadevolve_c=True,
                )
                for build in evaluation.builds:
                    self.assertTrue(build.built, f"{build.name}: {build.error}")
                for comparison in evaluation.comparisons:
                    self.assertTrue(
                        comparison.passed,
                        f"{comparison.left} vs {comparison.right}: "
                        f"{comparison.mismatches} {comparison.scores}",
                    )
                self.assertTrue(evaluation.passed)


class ParameterGroupTests(unittest.TestCase):
    def test_groups_partition_every_top_level_statement(self) -> None:
        code = convert(
            (FIXTURES / "patterned_plate.py").read_text(encoding="utf-8")
        ).code
        groups = parameter_groups(code)
        counted = sum(
            len(parameters) + len(statements) for parameters, statements in groups
        )
        body = ast.parse(code).body
        imports = sum(
            1
            for stmt in body
            if isinstance(stmt, (ast.Import, ast.ImportFrom))
        )
        self.assertEqual(counted, len(body) - imports)


if __name__ == "__main__":
    unittest.main()
