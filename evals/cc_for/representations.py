"""Build one program in every canonical representation and compare the solids.

The gates in :mod:`evals.cc_for.harness` ask whether *one* representation still
builds the source solid.  This module asks the question the dataset actually
depends on: do CADEvolve-C, CC-for and CC-step all describe the same part?

The three representations are produced by very different machinery.  CADEvolve-C
executes the program and records the concrete CadQuery calls it observes; CC-for
and CC-step rewrite the syntax without ever running it.  A canonicalizer that
quietly dropped a feature would still satisfy its own round-trip check as long as
it dropped the feature consistently, so comparing the representations against
each other -- and all of them against the source -- is what closes that gap.

CADEvolve-C is built in a subprocess.  Its tracer installs monkeypatches over
CadQuery's ``Workplane`` and ``Shape`` classes for the duration of the recording,
and this module executes other programs in the same interpreter; a separate
process is the only way to guarantee the patch cannot leak into them.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (REPO_ROOT, REPO_ROOT / "dataset_utils"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from evals.cc_for import similarity  # noqa: E402
from evals.cc_for.geometry import ShapeMetrics, compare_metrics, shape_metrics  # noqa: E402
from evals.cc_for.harness import (  # noqa: E402
    EXACT_ABSOLUTE_TOLERANCE,
    EXACT_CHAMFER_NOISE_MULTIPLE,
    EXACT_MAX_CHAMFER,
    EXACT_MIN_IOU,
    EXACT_RELATIVE_TOLERANCE,
    execute,
)
from utils.canonicalization.cc_for import CCForConfig, canonicalize_code  # noqa: E402
from utils.canonicalization.cc_for_validation import compare_solid_geometry  # noqa: E402

SOURCE = "source"
CADEVOLVE_C = "cadevolve_c"
CC_FOR = "cc_for"
CC_STEP = "cc_step"

REPRESENTATIONS: tuple[str, ...] = (SOURCE, CADEVOLVE_C, CC_FOR, CC_STEP)

# Pairs worth checking.  Every representation is compared against the source it
# came from, and the two symbolic ones against each other: they share every stage
# except parameter placement, so any disagreement between them is a placement bug.
COMPARISON_PAIRS: tuple[tuple[str, str], ...] = (
    (CC_FOR, SOURCE),
    (CC_STEP, SOURCE),
    (CC_STEP, CC_FOR),
    (CADEVOLVE_C, SOURCE),
)

# CADEvolve-C replays recorded calls rather than re-evaluating the source
# expressions, so it may resolve a selector to a different-but-equivalent entity
# and it discretizes parametric curves.  Its solids are held to shape agreement
# rather than to bit-identical mass properties.  The symbolic representations
# copy every argument expression verbatim and are held to exact equality.
TRACED_MIN_IOU = 0.98
TRACED_MAX_CHAMFER = 5e-3
TRACED_RELATIVE_TOLERANCE = 1e-6

DEFAULT_TRACER_TIMEOUT = 180.0

# Driving the legacy tracer: record the program, then emit the standardized form.
_TRACER_DRIVER = """
import contextlib, json, sys
sys.path[:0] = [%(root)r, %(dataset_utils)r]
from utils.canonicalization import standardizing
from utils.canonicalization.execution import execute_program

source = open(%(source_path)r, encoding="utf-8").read()
if "parametricCurve" in source:
    source = standardizing.replace_parametric_curve(
        standardizing.del_lambda(source) if "lambda" in source else source, 100
    )
elif "lambda" in source:
    source = standardizing.del_lambda(source)

standardizing.LOG.clear()
standardizing.patch()
try:
    with contextlib.redirect_stdout(sys.stderr):
        namespace = execute_program(source, "<source>")
finally:
    standardizing.unpatch()

code = standardizing.standardize_code(
    standardizing.LOG, final_var=%(result_name)r,
    final_ref=standardizing._ser(namespace[%(result_name)r]),
)
json.dump({"code": code, "calls": len(standardizing.LOG)}, sys.stdout)
"""


@dataclass(frozen=True)
class RepresentationBuild:
    """One representation's code, and the solid it builds."""

    name: str
    code: str | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None
    seconds: float = 0.0

    @property
    def built(self) -> bool:
        return self.metrics is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RepresentationComparison:
    """How one representation's solid relates to another's."""

    left: str
    right: str
    passed: bool
    exact: bool
    mismatches: tuple[str, ...] = ()
    scores: dict[str, Any] = field(default_factory=dict)
    skipped: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepresentationEvaluation:
    name: str
    passed: bool
    builds: list[RepresentationBuild] = field(default_factory=list)
    comparisons: list[RepresentationComparison] = field(default_factory=list)
    seconds: float = 0.0

    def build(self, name: str) -> RepresentationBuild | None:
        for entry in self.builds:
            if entry.name == name:
                return entry
        return None

    def comparison(self, left: str, right: str) -> RepresentationComparison | None:
        for entry in self.comparisons:
            if entry.left == left and entry.right == right:
                return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "seconds": self.seconds,
            "builds": [entry.to_dict() for entry in self.builds],
            "comparisons": [entry.to_dict() for entry in self.comparisons],
            "failed_comparisons": [
                f"{entry.left}_vs_{entry.right}"
                for entry in self.comparisons
                if not entry.passed and entry.skipped is None
            ],
        }


def build_cadevolve_c(
    source: str,
    *,
    result_name: str = "result",
    timeout: float = DEFAULT_TRACER_TIMEOUT,
    workdir: Path | None = None,
) -> str:
    """Return the legacy unscaled standardized trace of ``source``.

    The historical helper name does not imply that centering, extent scaling,
    or integer quantization is applied here. Those are separate lossy stages.

    Raises ``RuntimeError`` when the tracer cannot record the program; that is a
    property of the legacy stage, not of the representation being tested, so
    callers report it rather than treating it as a canonicalization failure.
    """

    import tempfile

    with tempfile.TemporaryDirectory(prefix="cadevolve-c-", dir=workdir) as scratch:
        source_path = Path(scratch) / "program.py"
        source_path.write_text(source, encoding="utf-8")
        driver = _TRACER_DRIVER % {
            "root": str(REPO_ROOT),
            "dataset_utils": str(REPO_ROOT / "dataset_utils"),
            "source_path": str(source_path),
            "result_name": result_name,
        }
        try:
            completed = subprocess.run(
                [sys.executable, "-c", driver],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=scratch,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"tracer timed out after {timeout:g}s") from None

    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        raise RuntimeError(detail[-1] if detail else "tracer exited non-zero")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        raise RuntimeError("tracer produced no standardized program") from None
    if not payload.get("calls"):
        raise RuntimeError("tracer recorded no CadQuery calls")
    return payload["code"]


def build_representations(
    source: str,
    *,
    name: str = "<program>",
    result_name: str = "result",
    loop_mode: str = "preserve",
    include: Sequence[str] = REPRESENTATIONS,
    execution_timeout: float = 90.0,
    tracer_timeout: float = DEFAULT_TRACER_TIMEOUT,
) -> list[RepresentationBuild]:
    """Produce each requested representation and execute it."""

    def _converted(placement: str) -> str:
        return canonicalize_code(
            source,
            CCForConfig(
                loop_mode=loop_mode,
                result_name=result_name,
                parameter_placement=placement,
            ),
        ).code

    producers = {
        SOURCE: lambda: source,
        CADEVOLVE_C: lambda: build_cadevolve_c(
            source, result_name=result_name, timeout=tracer_timeout
        ),
        CC_FOR: lambda: _converted("preamble"),
        CC_STEP: lambda: _converted("late"),
    }

    builds: list[RepresentationBuild] = []
    for representation in include:
        started = time.monotonic()
        try:
            code = producers[representation]()
        except Exception as error:
            builds.append(
                RepresentationBuild(
                    name=representation,
                    error=f"{type(error).__name__}: {error}",
                    seconds=time.monotonic() - started,
                )
            )
            continue
        try:
            namespace = execute(
                code, f"{name}:{representation}", timeout=execution_timeout
            )
            metrics = shape_metrics(namespace[result_name])
        except Exception as error:
            builds.append(
                RepresentationBuild(
                    name=representation,
                    code=code,
                    error=f"{type(error).__name__}: {error}",
                    seconds=time.monotonic() - started,
                )
            )
            continue
        builds.append(
            RepresentationBuild(
                name=representation,
                code=code,
                metrics=metrics.to_dict(),
                seconds=time.monotonic() - started,
            )
        )
    return builds


def _metrics_from(entry: RepresentationBuild) -> ShapeMetrics:
    assert entry.metrics is not None
    return ShapeMetrics(**entry.metrics)


def compare_representations(
    source: str,
    *,
    name: str = "<program>",
    result_name: str = "result",
    loop_mode: str = "preserve",
    include: Sequence[str] = REPRESENTATIONS,
    pairs: Iterable[tuple[str, str]] = COMPARISON_PAIRS,
    execution_timeout: float = 90.0,
    tracer_timeout: float = DEFAULT_TRACER_TIMEOUT,
    voxel_resolution: int | None = None,
    surface_points: int | None = None,
    require_cadevolve_c: bool = False,
) -> RepresentationEvaluation:
    """Build every representation of ``source`` and compare the solids pairwise."""

    started = time.monotonic()
    builds = build_representations(
        source,
        name=name,
        result_name=result_name,
        loop_mode=loop_mode,
        include=include,
        execution_timeout=execution_timeout,
        tracer_timeout=tracer_timeout,
    )
    by_name = {entry.name: entry for entry in builds}

    similarity_kwargs: dict[str, Any] = {}
    if voxel_resolution is not None:
        similarity_kwargs["voxel_resolution"] = voxel_resolution
    if surface_points is not None:
        similarity_kwargs["sample_points"] = surface_points

    # Re-executing per comparison would double the CAD cost, so each solid is
    # rebuilt once here and reused across every pair it takes part in.
    solids: dict[str, Any] = {}
    noise_floors: dict[str, float] = {}

    def _solid(representation: str) -> Any:
        if representation not in solids:
            entry = by_name[representation]
            assert entry.code is not None
            namespace = execute(
                entry.code,
                f"{name}:{representation}:compare",
                timeout=execution_timeout,
            )
            solids[representation] = namespace[result_name]
        return solids[representation]

    def _noise_floor(representation: str) -> float:
        """Chamfer between two samplings of this solid's own surface.

        Two builds of the same solid can tessellate in a different vertex order,
        so the area-weighted sampler draws different points and scores a non-zero
        distance.  Every Chamfer bound here is calibrated against that measured
        floor, exactly as the single-representation gate is.
        """

        if representation not in noise_floors:
            noise_floors[representation] = similarity.sampling_noise_floor(
                _solid(representation),
                sample_points=similarity_kwargs.get(
                    "sample_points", similarity.DEFAULT_SURFACE_POINTS
                ),
            )
        return noise_floors[representation]

    comparisons: list[RepresentationComparison] = []
    for left, right in pairs:
        exact = CADEVOLVE_C not in (left, right)
        if left not in by_name or right not in by_name:
            continue
        missing = [
            side
            for side in (left, right)
            if not by_name[side].built
        ]
        if missing:
            reason = "; ".join(
                f"{side}: {by_name[side].error}" for side in missing
            )
            optional = CADEVOLVE_C in missing and not require_cadevolve_c
            comparisons.append(
                RepresentationComparison(
                    left=left,
                    right=right,
                    passed=optional,
                    exact=exact,
                    skipped=reason if optional else None,
                    mismatches=() if optional else (reason,),
                )
            )
            continue

        mismatches = tuple(
            compare_metrics(
                _metrics_from(by_name[left]),
                _metrics_from(by_name[right]),
                relative_tolerance=(
                    EXACT_RELATIVE_TOLERANCE if exact else TRACED_RELATIVE_TOLERANCE
                ),
                absolute_tolerance=EXACT_ABSOLUTE_TOLERANCE,
                # The tracer may split or merge a face without changing the part,
                # so its topology counts are reported but do not decide the pair.
                compare_absolute_size=True,
            )
        )
        try:
            scores = similarity.compare_shapes(
                _solid(left), _solid(right), **similarity_kwargs
            ).to_dict()
            # Independent mesh normalization hides translations and uniform scale
            # errors. Verify occupied volume in the original frame as well.
            solid_comparison = compare_solid_geometry(
                _solid(left), _solid(right),
                relative_tolerance=(EXACT_RELATIVE_TOLERANCE if exact else TRACED_RELATIVE_TOLERANCE),
            )
        except Exception as error:
            comparisons.append(
                RepresentationComparison(
                    left=left,
                    right=right,
                    passed=False,
                    exact=exact,
                    mismatches=mismatches + (f"{type(error).__name__}: {error}",),
                )
            )
            continue

        min_iou = EXACT_MIN_IOU if exact else TRACED_MIN_IOU
        floor = max(_noise_floor(left), _noise_floor(right))
        max_chamfer = max(
            EXACT_MAX_CHAMFER if exact else TRACED_MAX_CHAMFER,
            EXACT_CHAMFER_NOISE_MULTIPLE * floor,
        )
        close = (
            scores["voxel_iou"] >= min_iou and scores["chamfer_l2"] <= max_chamfer
        )
        passed = close and solid_comparison["equivalent"] and (not exact or not mismatches)
        comparisons.append(
            RepresentationComparison(
                left=left,
                right=right,
                passed=passed,
                exact=exact,
                mismatches=mismatches,
                scores={
                    **scores,
                    "solid_comparison": solid_comparison,
                    "min_iou": min_iou,
                    "max_chamfer": max_chamfer,
                    "sampling_noise_floor": floor,
                },
            )
        )

    required = {SOURCE, CC_FOR, CC_STEP}
    if require_cadevolve_c:
        required.add(CADEVOLVE_C)
    passed = all(
        by_name[representation].built
        for representation in required
        if representation in by_name
    ) and all(entry.passed for entry in comparisons)

    return RepresentationEvaluation(
        name=name,
        passed=passed,
        builds=builds,
        comparisons=comparisons,
        seconds=time.monotonic() - started,
    )


__all__ = [
    "CADEVOLVE_C",
    "CC_FOR",
    "CC_STEP",
    "COMPARISON_PAIRS",
    "REPRESENTATIONS",
    "RepresentationBuild",
    "RepresentationComparison",
    "RepresentationEvaluation",
    "SOURCE",
    "build_cadevolve_c",
    "build_representations",
    "compare_representations",
]
