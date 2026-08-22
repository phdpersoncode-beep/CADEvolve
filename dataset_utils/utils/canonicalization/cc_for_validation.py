"""Execution-based validation helpers for CC-for conversions.

The structural converter has no CadQuery dependency.  This module imports no CAD
packages at module import time either; CadQuery is loaded only when a program is
executed.  That keeps syntax-only dataset conversion usable in lightweight
environments while making geometry and perturbation checks available when the CAD
stack is installed.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .cc_for import CanonicalizationResult, decompose_actions


@dataclass(frozen=True)
class ShapeSignature:
    bounds: tuple[float, float, float, float, float, float]
    volume: float
    area: float
    solids: int
    faces: int
    edges: int
    vertices: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SignatureComparison:
    equivalent: bool
    mismatches: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoundTripValidation:
    success: bool
    original: ShapeSignature | None = None
    canonical: ShapeSignature | None = None
    comparison: SignatureComparison | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QuantizedGeometryValidation:
    """Geometry fidelity after the legacy CADEvolve integer binarization."""

    success: bool
    quantized_pair: RoundTripValidation | None = None
    raw_to_quantized_chamfer: float | None = None
    chamfer_threshold: float = 0.15
    sample_points: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrefixValidation:
    success: bool
    checked_prefixes: int
    failure_index: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PerturbationCase:
    parameter: str
    original_value: float
    perturbed_value: float
    success: bool
    error: str | None = None
    mismatches: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PerturbationValidation:
    success: bool
    cases: list[PerturbationCase] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_shape(result: Any) -> Any:
    if result is None:
        raise ValueError("program did not define a non-None result")
    if hasattr(result, "val") and callable(result.val):
        result = result.val()
    if result is None:
        raise ValueError("result.val() returned None")
    return result


def execute_program(code: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    exec(compile(code, "<cad-program>", "exec"), namespace, namespace)
    return namespace


def _count(shape: Any, method: str) -> int:
    value = getattr(shape, method, None)
    if not callable(value):
        return 0
    try:
        return len(list(value()))
    except (TypeError, RuntimeError):
        return 0


def shape_signature(result: Any) -> ShapeSignature:
    shape = _as_shape(result)
    bbox = shape.BoundingBox()
    bounds = tuple(
        float(value)
        for value in (
            bbox.xmin,
            bbox.ymin,
            bbox.zmin,
            bbox.xmax,
            bbox.ymax,
            bbox.zmax,
        )
    )
    volume_fn = getattr(shape, "Volume", None)
    area_fn = getattr(shape, "Area", None)
    return ShapeSignature(
        bounds=bounds,  # type: ignore[arg-type]
        volume=float(volume_fn()) if callable(volume_fn) else 0.0,
        area=float(area_fn()) if callable(area_fn) else 0.0,
        solids=_count(shape, "Solids"),
        faces=_count(shape, "Faces"),
        edges=_count(shape, "Edges"),
        vertices=_count(shape, "Vertices"),
    )


def compare_signatures(
    left: ShapeSignature,
    right: ShapeSignature,
    *,
    relative_tolerance: float = 1e-7,
    absolute_tolerance: float = 1e-7,
) -> SignatureComparison:
    mismatches: list[str] = []
    for index, (a, b) in enumerate(zip(left.bounds, right.bounds)):
        if not math.isclose(
            a, b, rel_tol=relative_tolerance, abs_tol=absolute_tolerance
        ):
            mismatches.append(f"bound[{index}] differs: {a} != {b}")
    for name in ("volume", "area"):
        a = getattr(left, name)
        b = getattr(right, name)
        if not math.isclose(
            a, b, rel_tol=relative_tolerance, abs_tol=absolute_tolerance
        ):
            mismatches.append(f"{name} differs: {a} != {b}")
    for name in ("solids", "faces", "edges", "vertices"):
        a = getattr(left, name)
        b = getattr(right, name)
        if a != b:
            mismatches.append(f"{name} differs: {a} != {b}")
    return SignatureComparison(equivalent=not mismatches, mismatches=tuple(mismatches))


def validate_round_trip(
    original_code: str,
    canonical_code: str,
    *,
    result_name: str = "result",
    relative_tolerance: float = 1e-7,
    absolute_tolerance: float = 1e-7,
) -> RoundTripValidation:
    try:
        original_ns = execute_program(original_code)
        canonical_ns = execute_program(canonical_code)
        original = shape_signature(original_ns.get(result_name))
        canonical = shape_signature(canonical_ns.get(result_name))
        comparison = compare_signatures(
            original,
            canonical,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        return RoundTripValidation(
            success=comparison.equivalent,
            original=original,
            canonical=canonical,
            comparison=comparison,
        )
    except Exception as error:  # Validation must report per-file failures.
        return RoundTripValidation(success=False, error=f"{type(error).__name__}: {error}")


def binarize_numeric_literals(
    code: str, *, zero_tolerance: float = 1e-7, min_positive: int = 1
) -> str:
    """Apply CADEvolve's existing integer-literal binarizer to source code.

    This is a validation transform only.  CC-for output itself stays symbolic and
    retains the source units; the legacy binarizer is applied to both programs when
    checking that canonicalization commutes with the downstream quantization step.
    """

    # Imported lazily because syntax-only CC-for conversion has no NumPy/CadQuery
    # dependency, while the legacy binarization module does.
    from .binarization import Binarize

    tree = ast.parse(code)
    binarizer = Binarize(
        zero_tol=zero_tolerance, min_positive=min_positive
    )

    # The legacy transformer clamps literal arguments such as ``.chamfer(0.4)``
    # but misses the equivalent symbolic form ``radius = 0.4; .chamfer(radius)``.
    # CC-for deliberately introduces/retains those names, so collect parameters
    # feeding non-zero method slots and clamp their definitions after rounding.
    required_modes: dict[str, str] = {}
    required_attribute_modes: dict[str, str] = {}
    protected_predicate_literals: dict[tuple[int, int], int | float] = {}
    for comparison in (
        node for node in ast.walk(tree) if isinstance(node, ast.Compare)
    ):
        for comparator in comparison.comparators:
            for constant in ast.walk(comparator):
                value = constant.value if isinstance(constant, ast.Constant) else None
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and 0 < abs(value) <= 0.1
                    and hasattr(constant, "lineno")
                ):
                    protected_predicate_literals[
                        (constant.lineno, constant.col_offset)
                    ] = value
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(
            node.func, ast.Attribute
        ):
            continue
        spec = binarizer.nonzero_methods.get(node.func.attr.lower())
        if spec is None:
            continue
        mode = str(spec.get("mode", "gt0"))
        expressions: list[ast.AST] = []
        for index in spec.get("pos", []):
            if 0 <= index < len(node.args):
                expressions.append(node.args[index])
        keyword_names = set(spec.get("kw", []))
        expressions.extend(
            keyword.value
            for keyword in node.keywords
            if keyword.arg in keyword_names
        )
        for expression in expressions:
            for name_node in ast.walk(expression):
                if isinstance(name_node, ast.Name) and isinstance(
                    name_node.ctx, ast.Load
                ):
                    required_modes[name_node.id] = mode
                elif isinstance(name_node, ast.Attribute) and isinstance(
                    name_node.ctx, ast.Load
                ):
                    required_attribute_modes[name_node.attr] = mode

    tree = binarizer.visit(tree)

    class ClampSymbolicMethodParameters(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            protected = protected_predicate_literals.get(
                (getattr(node, "lineno", -1), getattr(node, "col_offset", -1))
            )
            if protected is not None:
                return ast.copy_location(ast.Constant(value=protected), node)
            return node

        def _clamp(self, name: str, value: ast.AST) -> ast.AST:
            mode = required_modes.get(name) or required_attribute_modes.get(name)
            if isinstance(value, ast.Constant) and mode is not None:
                return binarizer._clamp_const(
                    value, mode
                )
            return value

        def visit_Call(self, node: ast.Call) -> ast.AST:
            node = self.generic_visit(node)
            for keyword in node.keywords:
                if (
                    keyword.arg in required_attribute_modes
                    and isinstance(keyword.value, ast.Constant)
                ):
                    keyword.value = binarizer._clamp_const(
                        keyword.value,
                        required_attribute_modes[keyword.arg],
                    )
            return node

        def visit_Assign(self, node: ast.Assign) -> ast.AST:
            node = self.generic_visit(node)
            if len(node.targets) == 1 and isinstance(
                node.targets[0], ast.Name
            ):
                node.value = self._clamp(node.targets[0].id, node.value)
            return node

        def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
            node = self.generic_visit(node)
            if isinstance(node.target, ast.Name) and node.value is not None:
                node.value = self._clamp(node.target.id, node.value)
            return node

    tree = ClampSymbolicMethodParameters().visit(tree)
    ast.fix_missing_locations(tree)
    quantized = ast.unparse(tree).rstrip() + "\n"
    compile(quantized, "<binarized-cad-program>", "exec")
    return quantized


def _sample_normalized_surface(
    result: Any,
    *,
    sample_points: int,
    random_seed: int,
    tessellation_tolerance: float,
) -> Any:
    import numpy as np

    shape = _as_shape(result)
    vertices_raw, faces_raw = shape.tessellate(
        tessellation_tolerance, 0.1
    )
    vertices = np.asarray(
        [vertex.toTuple() for vertex in vertices_raw], dtype=np.float64
    )
    faces = np.asarray(faces_raw, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("shape tessellation produced no surface triangles")

    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = np.linalg.norm(cross, axis=1) * 0.5
    valid = areas > np.finfo(np.float64).eps
    triangles = triangles[valid]
    areas = areas[valid]
    if len(triangles) == 0:
        raise ValueError("shape tessellation contains only degenerate triangles")

    rng = np.random.default_rng(random_seed)
    choices = rng.choice(
        len(triangles),
        size=sample_points,
        replace=True,
        p=areas / areas.sum(),
    )
    chosen = triangles[choices]
    root_u = np.sqrt(rng.random(sample_points))
    v = rng.random(sample_points)
    points = (
        (1.0 - root_u)[:, None] * chosen[:, 0]
        + (root_u * (1.0 - v))[:, None] * chosen[:, 1]
        + (root_u * v)[:, None] * chosen[:, 2]
    )

    bbox = shape.BoundingBox()
    minimum = np.asarray((bbox.xmin, bbox.ymin, bbox.zmin))
    maximum = np.asarray((bbox.xmax, bbox.ymax, bbox.zmax))
    extent = float(np.max(maximum - minimum))
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError(f"shape has invalid maximum extent: {extent}")
    return (points - (minimum + maximum) * 0.5) / extent


def _symmetric_squared_chamfer(left: Any, right: Any) -> float:
    import numpy as np
    from scipy.spatial import cKDTree

    left_distance, _ = cKDTree(left).query(right, k=1)
    right_distance, _ = cKDTree(right).query(left, k=1)
    return float(
        np.mean(np.square(left_distance))
        + np.mean(np.square(right_distance))
    )


def validate_quantized_geometry(
    original_code: str,
    canonical_code: str,
    *,
    result_name: str = "result",
    sample_points: int = 4_096,
    random_seed: int = 0,
    chamfer_threshold: float = 0.15,
    tessellation_tolerance: float = 0.1,
) -> QuantizedGeometryValidation:
    """Validate CC-for before and after CADEvolve-style binarization.

    Two conditions must hold: independently binarizing the source and CC-for code
    produces equivalent solids, and the binarized CC-for solid remains close to the
    raw source solid under the normalized symmetric squared Chamfer metric already
    used by CADEvolve's geometry-normalization stages.
    """

    if sample_points < 1:
        raise ValueError("sample_points must be positive")
    try:
        quantized_original = binarize_numeric_literals(original_code)
        quantized_canonical = binarize_numeric_literals(canonical_code)
        quantized_pair = validate_round_trip(
            quantized_original,
            quantized_canonical,
            result_name=result_name,
            relative_tolerance=1e-6,
            absolute_tolerance=1e-6,
        )
        if not quantized_pair.success:
            return QuantizedGeometryValidation(
                success=False,
                quantized_pair=quantized_pair,
                chamfer_threshold=chamfer_threshold,
                sample_points=sample_points,
                error="source and CC-for diverged after binarization",
            )

        original_ns = execute_program(original_code)
        quantized_ns = execute_program(quantized_canonical)
        original_points = _sample_normalized_surface(
            original_ns.get(result_name),
            sample_points=sample_points,
            random_seed=random_seed,
            tessellation_tolerance=tessellation_tolerance,
        )
        quantized_points = _sample_normalized_surface(
            quantized_ns.get(result_name),
            sample_points=sample_points,
            random_seed=random_seed,
            tessellation_tolerance=tessellation_tolerance,
        )
        chamfer = _symmetric_squared_chamfer(
            original_points, quantized_points
        )
        return QuantizedGeometryValidation(
            success=chamfer <= chamfer_threshold,
            quantized_pair=quantized_pair,
            raw_to_quantized_chamfer=chamfer,
            chamfer_threshold=chamfer_threshold,
            sample_points=sample_points,
            error=(
                None
                if chamfer <= chamfer_threshold
                else f"normalized Chamfer {chamfer} exceeds {chamfer_threshold}"
            ),
        )
    except Exception as error:
        return QuantizedGeometryValidation(
            success=False,
            chamfer_threshold=chamfer_threshold,
            sample_points=sample_points,
            error=f"{type(error).__name__}: {error}",
        )


_WP_RE = re.compile(r"^wp(\d+)$")


def _latest_geometry(namespace: dict[str, Any], result_name: str) -> Any | None:
    candidates: list[tuple[int, Any]] = []
    for name, value in namespace.items():
        match = _WP_RE.match(name)
        if match:
            candidates.append((int(match.group(1)), value))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return namespace.get(result_name)


def _defines_geometry_state(code: str, result_name: str) -> bool:
    tree = ast.parse(code)

    class GeometryDefinitionFinder(ast.NodeVisitor):
        found = False

        def _visit_targets(self, targets: Iterable[ast.AST]) -> None:
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == result_name or _WP_RE.match(target.id):
                    self.found = True

        def visit_Assign(self, node: ast.Assign) -> None:
            self._visit_targets(node.targets)
            self.visit(node.value)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            self._visit_targets([node.target])
            if node.value is not None:
                self.visit(node.value)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

    finder = GeometryDefinitionFinder()
    finder.visit(tree)
    return finder.found


def validate_prefixes(
    code: str,
    result_name: str = "result",
    parameter_placement: str = "preamble",
) -> PrefixValidation:
    actions = decompose_actions(
        code, result_name=result_name, parameter_placement=parameter_placement
    )
    if not actions:
        return PrefixValidation(success=False, checked_prefixes=0, error="no actions")

    chunks: list[str] = []
    checked = 0
    geometry_started = False
    for index, action in enumerate(actions):
        chunks.append(action.code)
        if action.kind == "preamble":
            continue
        try:
            geometry_started = geometry_started or _defines_geometry_state(
                action.code, result_name
            )
            namespace = execute_program("\n".join(chunks))
            latest = _latest_geometry(namespace, result_name)
            if geometry_started and latest is None:
                raise ValueError("prefix did not expose a Workplane/Shape state")
            checked += 1
        except Exception as error:
            return PrefixValidation(
                success=False,
                checked_prefixes=checked,
                failure_index=index,
                error=f"{type(error).__name__}: {error}",
            )
    return PrefixValidation(success=True, checked_prefixes=checked)


def _numeric_parameter_assignments(code: str, names: Iterable[str]) -> dict[str, float]:
    wanted = set(names)
    values: dict[str, float] = {}
    tree = ast.parse(code)
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id in wanted
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, (int, float))
            and not isinstance(stmt.value.value, bool)
        ):
            values[stmt.targets[0].id] = float(stmt.value.value)
    return values


class _ParameterOverride(ast.NodeTransformer):
    def __init__(self, parameter: str, value: float) -> None:
        self.parameter = parameter
        self.value = value
        self.changed = False

    def _replacement(self, old: ast.AST) -> ast.Constant:
        return ast.copy_location(ast.Constant(value=self.value), old)

    def visit_Module(self, node: ast.Module) -> ast.Module:
        for stmt in node.body:
            if self.changed:
                break
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == self.parameter
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, (int, float))
                and not isinstance(stmt.value.value, bool)
            ):
                stmt.value = self._replacement(stmt.value)
                self.changed = True
                break
            value = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
            if isinstance(value, ast.Call):
                for keyword in value.keywords:
                    if (
                        keyword.arg == self.parameter
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, (int, float))
                        and not isinstance(keyword.value.value, bool)
                    ):
                        keyword.value = self._replacement(keyword.value)
                        self.changed = True
                        break
        return node


def override_numeric_parameter(code: str, parameter: str, value: float) -> str:
    tree = ast.parse(code)
    transformer = _ParameterOverride(parameter, value)
    tree = transformer.visit(tree)
    if not transformer.changed:
        raise KeyError(f"numeric parameter {parameter!r} not found")
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).rstrip() + "\n"


def validate_parameter_perturbations(
    original_code: str,
    canonical: CanonicalizationResult,
    *,
    result_name: str = "result",
    max_parameters: int = 16,
    relative_delta: float = 0.05,
) -> PerturbationValidation:
    numeric = _numeric_parameter_assignments(
        canonical.code, canonical.report.hoisted_parameters
    )
    cases: list[PerturbationCase] = []
    for parameter, value in list(numeric.items())[:max_parameters]:
        perturbed = value * (1.0 + relative_delta)
        if math.isclose(perturbed, value):
            perturbed = value + relative_delta
        try:
            modified_original = override_numeric_parameter(
                original_code, parameter, perturbed
            )
            modified_canonical = override_numeric_parameter(
                canonical.code, parameter, perturbed
            )
            validation = validate_round_trip(
                modified_original,
                modified_canonical,
                result_name=result_name,
            )
            mismatches = (
                validation.comparison.mismatches
                if validation.comparison is not None
                else ()
            )
            cases.append(
                PerturbationCase(
                    parameter=parameter,
                    original_value=value,
                    perturbed_value=perturbed,
                    success=validation.success,
                    error=validation.error,
                    mismatches=mismatches,
                )
            )
        except KeyError:
            # A derived canonical parameter may not be an independent source binding.
            continue
        except Exception as error:
            cases.append(
                PerturbationCase(
                    parameter=parameter,
                    original_value=value,
                    perturbed_value=perturbed,
                    success=False,
                    error=f"{type(error).__name__}: {error}",
                )
            )
    return PerturbationValidation(
        success=bool(cases) and all(case.success for case in cases), cases=cases
    )


__all__ = [
    "PerturbationCase",
    "PerturbationValidation",
    "PrefixValidation",
    "QuantizedGeometryValidation",
    "RoundTripValidation",
    "ShapeSignature",
    "SignatureComparison",
    "compare_signatures",
    "binarize_numeric_literals",
    "execute_program",
    "override_numeric_parameter",
    "shape_signature",
    "validate_parameter_perturbations",
    "validate_prefixes",
    "validate_quantized_geometry",
    "validate_round_trip",
]
