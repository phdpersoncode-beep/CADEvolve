"""Render matched isometric views of source and CC-for CadQuery programs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

from utils.canonicalization.cc_for import canonicalize_code
from utils.canonicalization.cc_for_validation import (
    execute_program,
    shape_signature,
    validate_round_trip,
)


def _as_shape(result: Any) -> Any:
    if hasattr(result, "val") and callable(result.val):
        return result.val()
    return result


def _mesh(result: Any) -> tuple[np.ndarray, np.ndarray]:
    shape = _as_shape(result)
    vertices, faces = shape.tessellate(0.08, 0.08)
    return (
        np.asarray([vertex.toTuple() for vertex in vertices], dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
    )


def _camera_projection(vertices: np.ndarray, center: np.ndarray) -> np.ndarray:
    forward = np.asarray((1.35, -1.55, 1.1), dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(np.asarray((0.0, 0.0, 1.0)), forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    relative = vertices - center
    return np.column_stack(
        (relative @ right, relative @ up, relative @ forward)
    )


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _render_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    color: str,
    center: np.ndarray,
    projection_bounds: tuple[float, float, float, float],
    size: tuple[int, int] = (820, 600),
    supersample: int = 2,
) -> Image.Image:
    """Render triangles with a CPU z-buffer so concave CAD parts sort correctly."""

    width, height = (value * supersample for value in size)
    projected = _camera_projection(vertices, center)
    xmin, xmax, ymin, ymax = projection_bounds
    margin = 42 * supersample
    scale = min(
        (width - 2 * margin) / max(xmax - xmin, 1e-9),
        (height - 2 * margin) / max(ymax - ymin, 1e-9),
    )
    xmid = (xmin + xmax) * 0.5
    ymid = (ymin + ymax) * 0.5
    screen = np.empty_like(projected)
    screen[:, 0] = (projected[:, 0] - xmid) * scale + width * 0.5
    screen[:, 1] = height * 0.5 - (projected[:, 1] - ymid) * scale
    screen[:, 2] = projected[:, 2]

    background = np.asarray(ImageColor.getrgb("#f4f6f8"), dtype=np.uint8)
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[:] = background
    z_buffer = np.full((height, width), -np.inf, dtype=np.float64)

    base = np.asarray(ImageColor.getrgb(color), dtype=np.float64)
    light = np.asarray((-0.3, -0.45, 0.84), dtype=np.float64)
    light /= np.linalg.norm(light)

    for face in faces:
        world_triangle = vertices[face]
        triangle = screen[face]
        normal = np.cross(
            world_triangle[1] - world_triangle[0],
            world_triangle[2] - world_triangle[0],
        )
        length = float(np.linalg.norm(normal))
        if length <= np.finfo(np.float64).eps:
            continue
        normal /= length
        intensity = 0.34 + 0.66 * abs(float(normal @ light))
        face_color = np.clip(base * intensity + 12.0, 0, 255).astype(np.uint8)

        x0, y0, z0 = triangle[0]
        x1, y1, z1 = triangle[1]
        x2, y2, z2 = triangle[2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denominator) <= 1e-12:
            continue
        left = max(0, int(np.floor(min(x0, x1, x2))))
        right = min(width - 1, int(np.ceil(max(x0, x1, x2))))
        top = max(0, int(np.floor(min(y0, y1, y2))))
        bottom = min(height - 1, int(np.ceil(max(y0, y1, y2))))
        if left > right or top > bottom:
            continue

        yy, xx = np.mgrid[top : bottom + 1, left : right + 1]
        sample_x = xx + 0.5
        sample_y = yy + 0.5
        weight0 = (
            (y1 - y2) * (sample_x - x2)
            + (x2 - x1) * (sample_y - y2)
        ) / denominator
        weight1 = (
            (y2 - y0) * (sample_x - x2)
            + (x0 - x2) * (sample_y - y2)
        ) / denominator
        weight2 = 1.0 - weight0 - weight1
        inside = (
            (weight0 >= -1e-9)
            & (weight1 >= -1e-9)
            & (weight2 >= -1e-9)
        )
        depth = weight0 * z0 + weight1 * z1 + weight2 * z2
        region_z = z_buffer[top : bottom + 1, left : right + 1]
        update = inside & (depth > region_z)
        region_z[update] = depth[update]
        pixels[top : bottom + 1, left : right + 1][update] = face_color

    rendered = Image.fromarray(pixels, mode="RGB")
    mask = Image.fromarray(
        np.where(np.isfinite(z_buffer), 255, 0).astype(np.uint8), mode="L"
    )
    outline = mask.filter(ImageFilter.MaxFilter(7))
    outline_array = np.asarray(outline) > np.asarray(mask)
    outlined = np.asarray(rendered).copy()
    outlined[outline_array] = np.asarray((43, 53, 67), dtype=np.uint8)
    rendered = Image.fromarray(outlined, mode="RGB")
    return rendered.resize(size, Image.Resampling.LANCZOS)


def render_pair(source_path: Path, output_dir: Path) -> dict[str, Any]:
    source = source_path.read_text(encoding="utf-8")
    converted = canonicalize_code(source)
    if converted.report.structural_errors:
        raise ValueError(converted.report.structural_errors)

    validation = validate_round_trip(source, converted.code)
    if not validation.success:
        raise ValueError(validation.to_dict())

    original_namespace = execute_program(source)
    canonical_namespace = execute_program(converted.code)
    original_result = original_namespace["result"]
    canonical_result = canonical_namespace["result"]
    original_vertices, original_faces = _mesh(original_result)
    canonical_vertices, canonical_faces = _mesh(canonical_result)

    all_vertices = np.vstack((original_vertices, canonical_vertices))
    center = (all_vertices.min(axis=0) + all_vertices.max(axis=0)) * 0.5
    all_projected = _camera_projection(all_vertices, center)
    projection_bounds = (
        float(all_projected[:, 0].min()),
        float(all_projected[:, 0].max()),
        float(all_projected[:, 1].min()),
        float(all_projected[:, 1].max()),
    )
    original_image = _render_mesh(
        original_vertices,
        original_faces,
        color="#4093c7",
        center=center,
        projection_bounds=projection_bounds,
    )
    canonical_image = _render_mesh(
        canonical_vertices,
        canonical_faces,
        color="#e79b38",
        center=center,
        projection_bounds=projection_bounds,
    )

    signature = shape_signature(original_result)
    canvas = Image.new("RGB", (1800, 900), ImageColor.getrgb("#f4f6f8"))
    canvas.paste(original_image, (65, 145))
    canvas.paste(canonical_image, (915, 145))
    draw = ImageDraw.Draw(canvas)
    title = source_path.stem.replace("_", " ").title()
    draw.text(
        (900, 45), title, fill="#111820", font=_font(42, bold=True), anchor="ma"
    )
    draw.text(
        (475, 112),
        "Original Zero-to-CAD",
        fill="#1f2a36",
        font=_font(27, bold=True),
        anchor="ma",
    )
    draw.text(
        (1325, 112),
        "Custom CC-for canonicalization",
        fill="#1f2a36",
        font=_font(27, bold=True),
        anchor="ma",
    )
    draw.text(
        (900, 850),
        (
            "Exact signature match  •  "
            f"volume {signature.volume:,.3f}  •  "
            f"faces {signature.faces}  •  solids {signature.solids}"
        ),
        fill="#354052",
        font=_font(22),
        anchor="mm",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{source_path.stem}_comparison.png"
    canonical_path = output_dir / f"{source_path.stem}.cc_for.py"
    canvas.save(image_path, optimize=True)
    canonical_path.write_text(converted.code, encoding="utf-8")

    return {
        "source": str(source_path),
        "canonical": str(canonical_path),
        "image": str(image_path),
        "signature": asdict(signature),
        "workplane_steps": converted.report.workplane_steps,
        "preserved_loops": converted.report.preserved_loops,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = [render_pair(path, args.output_dir) for path in args.sources]
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
