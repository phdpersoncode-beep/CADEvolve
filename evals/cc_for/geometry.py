"""Geometry metrics for comparing solids produced by two CadQuery programs.

This module is deliberately independent of
``dataset_utils.utils.canonicalization.cc_for_validation``: the point of the
evaluation suite is to check the canonicalizer with measurements that were not
written alongside it.  Everything here works on an executed CadQuery result and
imports CadQuery/OCP lazily so that the code-level gates stay usable in an
environment without the CAD stack.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

# Shapes are compared after independent normalization (centre on the bounding-box
# centre, divide by the longest extent).  That makes IoU/Chamfer invariant to the
# uniform rescaling that the downstream binarization stage introduces.
DEFAULT_VOXEL_RESOLUTION = 48
DEFAULT_SURFACE_POINTS = 8192
DEFAULT_TESSELLATION_TOLERANCE = 0.05


def _as_shape(result: Any) -> Any:
    """Unwrap a ``Workplane``/``Sketch``/``Shape`` into a single OCC shape.

    Unlike ``Workplane.val()``, every object on the stack is kept: a program whose
    result holds several disjoint bodies is wrapped into one compound.  Comparing
    only ``val()`` would silently ignore a canonicalization that dropped or added
    bodies, which is exactly one of the failures this suite has to catch.
    """

    if result is None:
        raise ValueError("program produced no result")
    if hasattr(result, "_faces"):  # cq.Sketch
        return result._faces
    if hasattr(result, "vals") and callable(result.vals):
        import cadquery as cq

        objects = [obj for obj in result.vals() if obj is not None]
        if not objects:
            raise ValueError("result exposes no geometry")
        if len(objects) == 1:
            return objects[0]
        return cq.Compound.makeCompound(objects)
    if hasattr(result, "val") and callable(result.val):
        result = result.val()
    if result is None:
        raise ValueError("result.val() returned None")
    return result


def _entities(shape: Any, accessor: str) -> list[Any]:
    method = getattr(shape, accessor, None)
    if not callable(method):
        return []
    try:
        return list(method())
    except (TypeError, RuntimeError, ValueError):
        return []


def _geom_type_histogram(entities: Sequence[Any]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for entity in entities:
        getter = getattr(entity, "geomType", None)
        try:
            kind = str(getter()) if callable(getter) else "UNKNOWN"
        except (RuntimeError, ValueError, TypeError):
            kind = "UNKNOWN"
        histogram[kind] = histogram.get(kind, 0) + 1
    return histogram


@dataclass(frozen=True)
class ShapeMetrics:
    """Topology and mass properties of one executed program's solid."""

    solids: int
    shells: int
    faces: int
    wires: int
    edges: int
    vertices: int
    compounds: int
    volume: float
    area: float
    bounds: tuple[float, float, float, float, float, float]
    extents: tuple[float, float, float]
    center_of_mass: tuple[float, float, float]
    face_types: dict[str, int] = field(default_factory=dict)
    edge_types: dict[str, int] = field(default_factory=dict)
    valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def max_extent(self) -> float:
        return max(self.extents)


def shape_metrics(result: Any) -> ShapeMetrics:
    shape = _as_shape(result)
    bbox = shape.BoundingBox()
    bounds = (
        float(bbox.xmin),
        float(bbox.ymin),
        float(bbox.zmin),
        float(bbox.xmax),
        float(bbox.ymax),
        float(bbox.zmax),
    )
    extents = (
        bounds[3] - bounds[0],
        bounds[4] - bounds[1],
        bounds[5] - bounds[2],
    )

    faces = _entities(shape, "Faces")
    edges = _entities(shape, "Edges")

    volume_fn = getattr(shape, "Volume", None)
    area_fn = getattr(shape, "Area", None)
    try:
        volume = float(volume_fn()) if callable(volume_fn) else 0.0
    except (RuntimeError, ValueError):
        volume = float("nan")
    try:
        area = float(area_fn()) if callable(area_fn) else 0.0
    except (RuntimeError, ValueError):
        area = float("nan")

    try:
        center = shape.Center()
        center_of_mass = (float(center.x), float(center.y), float(center.z))
    except (RuntimeError, ValueError, AttributeError):
        center_of_mass = (float("nan"),) * 3

    validity = getattr(shape, "isValid", None)
    try:
        valid = bool(validity()) if callable(validity) else True
    except (RuntimeError, ValueError):
        valid = False

    return ShapeMetrics(
        solids=len(_entities(shape, "Solids")),
        shells=len(_entities(shape, "Shells")),
        faces=len(faces),
        wires=len(_entities(shape, "Wires")),
        edges=len(edges),
        vertices=len(_entities(shape, "Vertices")),
        compounds=len(_entities(shape, "Compounds")),
        volume=volume,
        area=area,
        bounds=bounds,
        extents=extents,
        center_of_mass=center_of_mass,
        face_types=_geom_type_histogram(faces),
        edge_types=_geom_type_histogram(edges),
        valid=valid,
    )


TOPOLOGY_FIELDS = (
    "solids",
    "shells",
    "faces",
    "wires",
    "edges",
    "vertices",
    "compounds",
)


def compare_metrics(
    left: ShapeMetrics,
    right: ShapeMetrics,
    *,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-9,
    compare_absolute_size: bool = True,
) -> list[str]:
    """Return human-readable mismatches; an empty list means the solids agree.

    ``compare_absolute_size`` is disabled when comparing across a stage that is
    allowed to rescale the part (numeric binarization), where only topology and
    normalized shape are expected to survive.
    """

    mismatches: list[str] = []

    for name in TOPOLOGY_FIELDS:
        a = getattr(left, name)
        b = getattr(right, name)
        if a != b:
            mismatches.append(f"{name}: {a} != {b}")

    for name in ("face_types", "edge_types"):
        a = getattr(left, name)
        b = getattr(right, name)
        if a != b:
            mismatches.append(f"{name}: {sorted(a.items())} != {sorted(b.items())}")

    if left.valid != right.valid:
        mismatches.append(f"valid: {left.valid} != {right.valid}")

    if not compare_absolute_size:
        return mismatches

    for name in ("volume", "area"):
        a = getattr(left, name)
        b = getattr(right, name)
        if not math.isclose(a, b, rel_tol=relative_tolerance, abs_tol=absolute_tolerance):
            mismatches.append(f"{name}: {a!r} != {b!r}")

    for index, (a, b) in enumerate(zip(left.bounds, right.bounds)):
        if not math.isclose(a, b, rel_tol=relative_tolerance, abs_tol=absolute_tolerance):
            mismatches.append(f"bounds[{index}]: {a!r} != {b!r}")

    for index, (a, b) in enumerate(zip(left.center_of_mass, right.center_of_mass)):
        if not math.isclose(a, b, rel_tol=relative_tolerance, abs_tol=absolute_tolerance):
            mismatches.append(f"center_of_mass[{index}]: {a!r} != {b!r}")

    return mismatches
