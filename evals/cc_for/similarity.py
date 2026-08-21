"""Scale-invariant shape similarity: voxel IoU and surface Chamfer distance.

Both metrics normalize each solid independently -- translate by the bounding-box
centre, divide by the longest extent -- so a stage that uniformly rescales a part
(numeric binarization) is not penalised for the rescaling itself, only for the
shape change it causes.

The voxeliser is mesh-based and fully vectorised: triangles are rasterised onto
the (x, y) grid columns, the z of every ray/triangle crossing is recorded, and
each column is filled by parity.  That is far faster than calling OpenCascade's
solid classifier once per voxel, and needs no CAD calls beyond tessellation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .geometry import (
    DEFAULT_SURFACE_POINTS,
    DEFAULT_TESSELLATION_TOLERANCE,
    DEFAULT_VOXEL_RESOLUTION,
    _as_shape,
)

# Column centres are nudged by this fraction of a cell so that rays do not pass
# exactly through a shared triangle edge or vertex, where parity counting is
# ambiguous.  The value is irrational-ish on purpose.
_COLUMN_JITTER = 0.0137


@dataclass(frozen=True)
class SimilarityScores:
    voxel_iou: float
    voxel_resolution: int
    occupancy_left: int
    occupancy_right: int
    chamfer_l2: float
    chamfer_squared: float
    hausdorff_95: float
    surface_points: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def triangle_mesh(result: Any, tolerance: float = DEFAULT_TESSELLATION_TOLERANCE):
    """Tessellate a shape into ``(vertices, faces)`` float/int numpy arrays."""

    import numpy as np

    shape = _as_shape(result)
    vertices_raw, faces_raw = shape.tessellate(tolerance, 0.1)
    vertices = np.asarray(
        [vertex.toTuple() for vertex in vertices_raw], dtype=np.float64
    )
    faces = np.asarray(faces_raw, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError("tessellation produced no triangles")
    return vertices, faces


def normalize_mesh(vertices):
    """Centre on the bounding-box centre and divide by the longest extent."""

    import numpy as np

    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    extent = float(np.max(maximum - minimum))
    if not math.isfinite(extent) or extent <= 0.0:
        raise ValueError(f"degenerate shape extent: {extent}")
    return (vertices - (minimum + maximum) * 0.5) / extent


def voxelize(vertices, faces, resolution: int = DEFAULT_VOXEL_RESOLUTION):
    """Return a boolean ``(resolution,) * 3`` occupancy grid over ``[-0.5, 0.5]``.

    Vertices are expected to be normalized already.  A voxel is occupied when its
    centre lies inside the closed surface, decided by even/odd ray crossings
    along +Z.
    """

    import numpy as np

    if resolution < 2:
        raise ValueError("resolution must be at least 2")

    cell = 1.0 / resolution
    axis = -0.5 + (np.arange(resolution) + 0.5 + _COLUMN_JITTER) * cell
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    columns = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (C, 2)

    triangles = vertices[faces]  # (T, 3, 3)
    column_indices: list[Any] = []
    crossing_z: list[Any] = []

    # Chunk over triangles so the (chunk, columns) intermediate stays small.
    chunk = max(1, int(2_000_000 // max(1, len(columns))))
    for start in range(0, len(triangles), chunk):
        block = triangles[start : start + chunk]
        a, b, c = block[:, 0, :], block[:, 1, :], block[:, 2, :]
        edge1 = b[:, :2] - a[:, :2]
        edge2 = c[:, :2] - a[:, :2]
        denominator = edge1[:, 0] * edge2[:, 1] - edge2[:, 0] * edge1[:, 1]
        usable = np.abs(denominator) > 1e-18
        if not np.any(usable):
            continue
        a, b, c = a[usable], b[usable], c[usable]
        edge1, edge2 = edge1[usable], edge2[usable]
        denominator = denominator[usable]

        offset = columns[None, :, :] - a[:, None, :2]  # (t, C, 2)
        weight1 = (
            offset[..., 0] * edge2[:, None, 1] - edge2[:, None, 0] * offset[..., 1]
        ) / denominator[:, None]
        weight2 = (
            edge1[:, None, 0] * offset[..., 1] - offset[..., 0] * edge1[:, None, 1]
        ) / denominator[:, None]
        inside = (weight1 >= 0.0) & (weight2 >= 0.0) & (weight1 + weight2 <= 1.0)
        if not np.any(inside):
            continue
        triangle_row, column_row = np.nonzero(inside)
        z = (
            a[triangle_row, 2]
            + weight1[triangle_row, column_row] * (b[triangle_row, 2] - a[triangle_row, 2])
            + weight2[triangle_row, column_row] * (c[triangle_row, 2] - a[triangle_row, 2])
        )
        column_indices.append(column_row)
        crossing_z.append(z)

    grid = np.zeros((len(columns), resolution), dtype=np.int32)
    if not column_indices:
        return grid.reshape(resolution, resolution, resolution).astype(bool)

    all_columns = np.concatenate(column_indices)
    all_z = np.concatenate(crossing_z)
    order = np.lexsort((all_z, all_columns))
    all_columns = all_columns[order]
    all_z = all_z[order]

    # Within each column the sorted crossings pair up into inside intervals.
    boundaries = np.flatnonzero(np.diff(all_columns)) + 1
    starts = np.concatenate([[0], boundaries])
    lengths = np.diff(np.concatenate([starts, [len(all_columns)]]))
    rank = np.arange(len(all_columns)) - np.repeat(starts, lengths)
    # A column whose crossings are odd in number is a tessellation artefact (a ray
    # grazing a shared edge); drop its trailing crossing rather than filling to
    # infinity, keeping entry/exit arrays aligned.
    paired = rank < np.repeat(lengths // 2 * 2, lengths)
    all_columns, all_z, rank = all_columns[paired], all_z[paired], rank[paired]
    enters = rank % 2 == 0
    entry_columns = all_columns[enters]
    entry_z = all_z[enters]
    exit_z = all_z[~enters]

    lower = np.ceil((entry_z + 0.5) / cell - 0.5 - _COLUMN_JITTER).astype(np.int64)
    upper = np.floor((exit_z + 0.5) / cell - 0.5 - _COLUMN_JITTER).astype(np.int64)
    np.clip(lower, 0, resolution, out=lower)
    np.clip(upper, -1, resolution - 1, out=upper)
    keep = upper >= lower
    entry_columns, lower, upper = entry_columns[keep], lower[keep], upper[keep]

    difference = np.zeros((len(columns), resolution + 1), dtype=np.int32)
    np.add.at(difference, (entry_columns, lower), 1)
    np.add.at(difference, (entry_columns, upper + 1), -1)
    grid = np.cumsum(difference[:, :-1], axis=1)
    return (grid > 0).reshape(resolution, resolution, resolution)


def sample_surface(
    vertices,
    faces,
    *,
    sample_points: int = DEFAULT_SURFACE_POINTS,
    random_seed: int = 0,
):
    """Area-weighted uniform samples on an (already normalized) triangle mesh."""

    import numpy as np

    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    areas = np.linalg.norm(cross, axis=1) * 0.5
    keep = areas > np.finfo(np.float64).eps
    triangles, areas = triangles[keep], areas[keep]
    if len(triangles) == 0:
        raise ValueError("mesh contains only degenerate triangles")

    rng = np.random.default_rng(random_seed)
    choice = rng.choice(
        len(triangles), size=sample_points, replace=True, p=areas / areas.sum()
    )
    chosen = triangles[choice]
    root_u = np.sqrt(rng.random(sample_points))
    v = rng.random(sample_points)
    return (
        (1.0 - root_u)[:, None] * chosen[:, 0]
        + (root_u * (1.0 - v))[:, None] * chosen[:, 1]
        + (root_u * v)[:, None] * chosen[:, 2]
    )


def chamfer(left_points, right_points) -> tuple[float, float, float]:
    """Return ``(mean L2, mean squared, 95th-percentile Hausdorff)``."""

    import numpy as np
    from scipy.spatial import cKDTree

    forward, _ = cKDTree(left_points).query(right_points, k=1)
    backward, _ = cKDTree(right_points).query(left_points, k=1)
    both = np.concatenate([forward, backward])
    return (
        float(np.mean(forward) + np.mean(backward)) * 0.5,
        float(np.mean(np.square(forward)) + np.mean(np.square(backward))),
        float(np.percentile(both, 95)),
    )


def iou(left_grid, right_grid) -> float:
    import numpy as np

    intersection = int(np.count_nonzero(left_grid & right_grid))
    union = int(np.count_nonzero(left_grid | right_grid))
    if union == 0:
        return 1.0
    return intersection / union


def compare_shapes(
    left_result: Any,
    right_result: Any,
    *,
    voxel_resolution: int = DEFAULT_VOXEL_RESOLUTION,
    sample_points: int = DEFAULT_SURFACE_POINTS,
    random_seed: int = 0,
    tessellation_tolerance: float = DEFAULT_TESSELLATION_TOLERANCE,
) -> SimilarityScores:
    """Normalize both solids independently, then score IoU and Chamfer."""

    import numpy as np

    left_vertices, left_faces = triangle_mesh(left_result, tessellation_tolerance)
    right_vertices, right_faces = triangle_mesh(right_result, tessellation_tolerance)
    left_vertices = normalize_mesh(left_vertices)
    right_vertices = normalize_mesh(right_vertices)

    left_grid = voxelize(left_vertices, left_faces, voxel_resolution)
    right_grid = voxelize(right_vertices, right_faces, voxel_resolution)

    left_points = sample_surface(
        left_vertices, left_faces, sample_points=sample_points, random_seed=random_seed
    )
    right_points = sample_surface(
        right_vertices,
        right_faces,
        sample_points=sample_points,
        random_seed=random_seed,
    )
    chamfer_l2, chamfer_squared, hausdorff = chamfer(left_points, right_points)

    return SimilarityScores(
        voxel_iou=iou(left_grid, right_grid),
        voxel_resolution=voxel_resolution,
        occupancy_left=int(np.count_nonzero(left_grid)),
        occupancy_right=int(np.count_nonzero(right_grid)),
        chamfer_l2=chamfer_l2,
        chamfer_squared=chamfer_squared,
        hausdorff_95=hausdorff,
        surface_points=sample_points,
    )
