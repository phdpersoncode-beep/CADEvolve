"""Run every CC-for evaluation gate for a single CadQuery program.

A gate is a named boolean with a detail payload.  ``evaluate_program`` never
raises for a program-level problem: a crash inside a gate becomes a failed gate
so that a corpus run produces one comparable record per program.

Gate families
-------------
``structure``   the canonical program still satisfies the CC-for contract;
``symbols``     named parameters, their reuse, and ``for`` loops survived;
``exact``       original and canonical build the *same* solid (topology, mass,
                bounding box, face/edge type histograms, IoU, Chamfer);
``prefix``      every emitted action prefix executes on its own;
``quantized``   canonicalization commutes with the downstream binarization, and
                the binarized solid stays close to the original;
``robust``      perturbing a named parameter changes both programs identically,
                which is only possible if the symbolic relationships survived.
"""

from __future__ import annotations

import ast
import math
import signal
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from utils.canonicalization.cc_for import (
    CCForConfig,
    canonicalize_code,
    decompose_actions,
    validate_structure,
)

from . import code_metrics
from .geometry import DEFAULT_SURFACE_POINTS, compare_metrics, shape_metrics
from .similarity import compare_shapes, sampling_noise_floor

# The canonical program replays the identical CadQuery calls, so topology counts
# and face/edge type histograms are required to match *exactly*.  Mass properties
# get a small relative tolerance instead: OpenCascade's tolerance-driven surface
# algorithms are not bit-reproducible across two builds of the same solid, and
# were measured drifting by ~5e-8 relative on filleted and swept parts.
EXACT_RELATIVE_TOLERANCE = 1e-7
EXACT_ABSOLUTE_TOLERANCE = 1e-9
# Voxel IoU has a discretization floor of its own: a planar face that lands on a
# voxel boundary flips a partial slab either way, so two identical solids can
# score ~0.998 and the shortfall does not shrink with resolution (measured across
# 32-80 on parts whose exact metrics match with zero mismatches).  A real
# divergence -- a missing hole, a dropped body -- moves IoU far more than 1%.
EXACT_MIN_IOU = 0.99
# Surface Chamfer cannot be held to 1e-9: two builds of the same solid can
# tessellate in a different vertex order, so the sampler draws different points.
# The gate calibrates against that measured noise floor and keeps this only as a
# lower bound for the well-behaved case.
EXACT_MAX_CHAMFER = 1e-3
EXACT_CHAMFER_NOISE_MULTIPLE = 3.0

# The quantized path is allowed to move geometry: binarization rounds every
# numeric literal to an integer, so a small part changes shape noticeably.
QUANTIZED_MIN_IOU = 0.80
QUANTIZED_MAX_CHAMFER = 0.05
# Slack for the same discretization floor when attributing IoU loss.
VOXEL_IOU_SLACK = 1.0 - EXACT_MIN_IOU

PERTURBATION_FACTOR = 1.37
PERTURBATION_MIN_IOU = 0.999
DEFAULT_MAX_PERTURBATIONS = 3

# A CAD program can run forever: a corpus program with a data-dependent ``while``,
# or one of ours once a perturbation pushes a convergence ratio past 1.0.  Every
# execution therefore runs under a wall-clock alarm.
DEFAULT_EXECUTION_TIMEOUT = 90.0

# Replaying every action prefix costs O(n^2) CAD operations.  Long programs are
# subsampled -- always including the first and last prefix -- so a corpus run
# stays affordable without dropping the gate.
DEFAULT_MAX_PREFIX_CHECKS = 24


class ProgramTimeout(RuntimeError):
    """Raised when a program exceeds its execution budget."""


@contextmanager
def time_limit(seconds: float):
    """Interrupt the running program after ``seconds`` of wall clock.

    ``SIGALRM`` only fires between bytecodes, so a single OpenCascade call that
    never returns cannot be interrupted -- but a Python loop calling into
    OpenCascade repeatedly can, and that is the case that actually occurs.
    """

    if seconds <= 0:
        yield
        return

    def _fire(signum, frame):  # noqa: ANN001 - signal handler signature
        raise ProgramTimeout(f"program exceeded {seconds:g}s")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


@dataclass
class Gate:
    name: str
    passed: bool
    skipped: bool = False
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProgramEvaluation:
    name: str
    loop_mode: str
    passed: bool
    gates: list[Gate] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    code_comparison: dict[str, Any] = field(default_factory=dict)
    canonical_code: str | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failed_gates"] = [
            gate.name for gate in self.gates if not gate.passed and not gate.skipped
        ]
        return data

    def gate(self, name: str) -> Gate | None:
        for gate in self.gates:
            if gate.name == name:
                return gate
        return None


def execute(
    code: str,
    filename: str = "<program>",
    timeout: float = DEFAULT_EXECUTION_TIMEOUT,
) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "__cad_program__"}
    with time_limit(timeout):
        exec(compile(code, filename, "exec"), namespace, namespace)
    return namespace


def _result_of(namespace: dict[str, Any], result_name: str) -> Any:
    value = namespace.get(result_name)
    if value is None:
        raise ValueError(f"program defined no non-None {result_name!r}")
    return value


def _run_gate(
    name: str, action: Callable[[], tuple[bool, dict[str, Any]]]
) -> Gate:
    started = time.monotonic()
    try:
        passed, detail = action()
        return Gate(name=name, passed=passed, detail=detail, seconds=time.monotonic() - started)
    except Exception as error:  # a gate crash is a gate failure, never a run abort
        return Gate(
            name=name,
            passed=False,
            error=f"{type(error).__name__}: {error}",
            seconds=time.monotonic() - started,
        )


def _skipped(name: str, reason: str) -> Gate:
    return Gate(name=name, passed=True, skipped=True, detail={"reason": reason})


def _scale_numeric_parameter(code: str, parameter: str, factor: float) -> str | None:
    """Multiply one top-level numeric parameter's literal definition by ``factor``.

    Returns ``None`` when the name is not a plain top-level numeric assignment.
    """

    tree = ast.parse(code)
    changed = False
    for stmt in tree.body:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            continue
        target = stmt.targets[0]
        if not (isinstance(target, ast.Name) and target.id == parameter):
            continue
        value = stmt.value
        if not (
            isinstance(value, ast.Constant)
            and isinstance(value.value, (int, float))
            and not isinstance(value.value, bool)
        ):
            return None
        stmt.value = ast.copy_location(
            ast.Constant(value=float(value.value) * factor), value
        )
        changed = True
    if not changed:
        return None
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).rstrip() + "\n"


def _perturbable_parameters(
    source: str, canonical: str, comparison: code_metrics.CodeComparison, report: dict[str, Any]
) -> list[tuple[str, str]]:
    """Source/canonical name pairs that are plain non-zero numeric parameters."""

    canonical_names = code_metrics.bound_names(canonical)
    canonical_loads = code_metrics.load_counts(canonical)
    # Only single-definition names are comparable: a name the source rebinds is
    # split into x_1/x_2 by SSA, so scaling "x" on one side and "x_1" on the other
    # would perturb different things and the gate would compare unlike programs.
    definitions = Counter(
        target.id
        for stmt in ast.parse(source).body
        if isinstance(stmt, ast.Assign)
        for target in stmt.targets
        if isinstance(target, ast.Name)
    )
    pairs: list[tuple[str, str]] = []
    for stmt in ast.parse(source).body:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            continue
        target, value = stmt.targets[0], stmt.value
        if not (isinstance(target, ast.Name) and isinstance(value, ast.Constant)):
            continue
        if definitions[target.id] != 1:
            continue
        if not isinstance(value.value, float) or value.value == 0.0:
            # Integers are usually counts; scaling them changes topology on
            # purpose, which is not what this gate is measuring.
            continue
        canonical_name = code_metrics.resolve_parameter(
            target.id,
            canonical_names,
            flattened=report.get("flattened_namespaces"),
            versioned=report.get("versioned_names"),
        )
        if canonical_name and canonical_loads.get(canonical_name, 0) > 0:
            pairs.append((target.id, canonical_name))
    return pairs


def evaluate_program(
    source: str,
    *,
    name: str = "<program>",
    loop_mode: str = "preserve",
    result_name: str = "result",
    run_geometry: bool = True,
    run_prefixes: bool = True,
    run_quantization: bool = True,
    run_perturbations: bool = True,
    max_perturbations: int = DEFAULT_MAX_PERTURBATIONS,
    execution_timeout: float = DEFAULT_EXECUTION_TIMEOUT,
    max_prefix_checks: int = DEFAULT_MAX_PREFIX_CHECKS,
    voxel_resolution: int | None = None,
    surface_points: int | None = None,
    keep_canonical_code: bool = False,
) -> ProgramEvaluation:
    started = time.monotonic()
    gates: list[Gate] = []

    def run(code: str, label: str) -> dict[str, Any]:
        return execute(code, f"{name}:{label}", timeout=execution_timeout)

    similarity_kwargs: dict[str, Any] = {}
    if voxel_resolution is not None:
        similarity_kwargs["voxel_resolution"] = voxel_resolution
    if surface_points is not None:
        similarity_kwargs["sample_points"] = surface_points

    # --- conversion -------------------------------------------------------
    try:
        conversion = canonicalize_code(
            source, CCForConfig(loop_mode=loop_mode, result_name=result_name)
        )
    except Exception as error:
        gate = Gate(
            name="converts",
            passed=False,
            error=f"{type(error).__name__}: {error}",
            seconds=time.monotonic() - started,
        )
        return ProgramEvaluation(
            name=name,
            loop_mode=loop_mode,
            passed=False,
            gates=[gate],
            seconds=time.monotonic() - started,
        )

    canonical = conversion.code
    report = conversion.report.to_dict()
    gates.append(Gate(name="converts", passed=True))

    # --- structural contract ---------------------------------------------
    gates.append(
        _run_gate(
            "structure",
            lambda: (
                not report["structural_errors"] and not validate_structure(canonical, result_name),
                {
                    "structural_errors": report["structural_errors"],
                    "contract_errors": validate_structure(canonical, result_name),
                    "workplane_steps": report["workplane_steps"],
                },
            ),
        )
    )

    def _idempotent() -> tuple[bool, dict[str, Any]]:
        again = canonicalize_code(
            canonical, CCForConfig(loop_mode=loop_mode, result_name=result_name)
        ).code
        return again == canonical, {"stable": again == canonical}

    gates.append(_run_gate("idempotent", _idempotent))

    # --- symbolic representation -----------------------------------------
    comparison = code_metrics.compare_code(
        source, canonical, report=report, result_name=result_name
    )

    gates.append(
        _run_gate(
            "parameters_preserved",
            lambda: (
                not comparison.lost_parameters and not comparison.inlined_parameters,
                {
                    "source_parameters": comparison.source_parameters,
                    "retained": comparison.retained_parameters,
                    "retention": comparison.parameter_retention,
                    "lost": comparison.lost_parameters,
                    "inlined": comparison.inlined_parameters,
                },
            ),
        )
    )

    gates.append(
        _run_gate(
            "parameters_hoisted",
            lambda: (
                bool(comparison.preamble["contiguous"]),
                comparison.preamble,
            ),
        )
    )

    if loop_mode == "preserve":
        gates.append(
            _run_gate(
                "loops_preserved",
                lambda: (
                    comparison.canonical_loops == comparison.source_loops
                    and comparison.canonical_comprehensions
                    >= comparison.source_comprehensions,
                    {
                        "source_loops": comparison.source_loops,
                        "canonical_loops": comparison.canonical_loops,
                        "source_comprehensions": comparison.source_comprehensions,
                        "canonical_comprehensions": comparison.canonical_comprehensions,
                        "preserved_loops": report["preserved_loops"],
                    },
                ),
            )
        )
    else:
        gates.append(
            _run_gate(
                "loops_unrolled",
                lambda: (
                    comparison.canonical_loops == 0 or report["preserved_loops"] > 0,
                    {
                        "canonical_loops": comparison.canonical_loops,
                        "unrolled_loops": report["unrolled_loops"],
                        "preserved_loops": report["preserved_loops"],
                    },
                ),
            )
        )

    gates.append(
        _run_gate(
            "literals_stable",
            lambda: (
                not comparison.numeric_literals_added
                and not comparison.numeric_literals_removed
                and not comparison.string_literals_added
                and not comparison.string_literals_removed,
                {
                    "numeric_added": comparison.numeric_literals_added[:12],
                    "numeric_removed": comparison.numeric_literals_removed[:12],
                    "string_added": comparison.string_literals_added[:12],
                    "string_removed": comparison.string_literals_removed[:12],
                },
            ),
        )
    )

    gates.append(
        _run_gate(
            "chains_lowered",
            lambda: (
                comparison.canonical_chain_depth <= 1,
                {
                    "source_chain_depth": comparison.source_chain_depth,
                    "canonical_chain_depth": comparison.canonical_chain_depth,
                    "workplane_steps": comparison.workplane_steps,
                },
            ),
        )
    )

    evaluation = ProgramEvaluation(
        name=name,
        loop_mode=loop_mode,
        passed=False,
        gates=gates,
        report=report,
        code_comparison=comparison.to_dict(),
        canonical_code=canonical if keep_canonical_code else None,
    )

    if not run_geometry:
        for gate_name in (
            "source_executes",
            "canonical_executes",
            "topology_identical",
            "shape_identical",
            "prefixes_execute",
            "quantization_commutes",
            "quantized_shape_close",
            "parameter_perturbation",
        ):
            gates.append(_skipped(gate_name, "geometry disabled"))
        evaluation.passed = all(gate.passed for gate in gates)
        evaluation.seconds = time.monotonic() - started
        return evaluation

    # --- execution --------------------------------------------------------
    source_result: Any = None
    canonical_result: Any = None

    def _source_executes() -> tuple[bool, dict[str, Any]]:
        nonlocal source_result
        source_result = _result_of(run(source, "source"), result_name)
        metrics = shape_metrics(source_result)
        return metrics.valid, {"metrics": metrics.to_dict()}

    source_gate = _run_gate("source_executes", _source_executes)
    gates.append(source_gate)

    def _canonical_executes() -> tuple[bool, dict[str, Any]]:
        nonlocal canonical_result
        canonical_result = _result_of(run(canonical, "canonical"), result_name)
        metrics = shape_metrics(canonical_result)
        return metrics.valid, {"metrics": metrics.to_dict()}

    canonical_gate = _run_gate("canonical_executes", _canonical_executes)
    gates.append(canonical_gate)

    if not (source_gate.passed and canonical_gate.passed):
        for gate_name in (
            "topology_identical",
            "shape_identical",
            "prefixes_execute",
            "quantization_commutes",
            "quantized_shape_close",
            "parameter_perturbation",
        ):
            gates.append(_skipped(gate_name, "execution gate failed"))
        evaluation.passed = all(gate.passed for gate in gates)
        evaluation.seconds = time.monotonic() - started
        return evaluation

    def _topology_identical() -> tuple[bool, dict[str, Any]]:
        left = shape_metrics(source_result)
        right = shape_metrics(canonical_result)
        mismatches = compare_metrics(
            left,
            right,
            relative_tolerance=EXACT_RELATIVE_TOLERANCE,
            absolute_tolerance=EXACT_ABSOLUTE_TOLERANCE,
        )
        return not mismatches, {
            "mismatches": mismatches,
            "solids": left.solids,
            "faces": left.faces,
            "edges": left.edges,
            "vertices": left.vertices,
            "volume": left.volume,
        }

    gates.append(_run_gate("topology_identical", _topology_identical))

    def _shape_identical() -> tuple[bool, dict[str, Any]]:
        scores = compare_shapes(source_result, canonical_result, **similarity_kwargs)
        noise = sampling_noise_floor(
            source_result,
            sample_points=similarity_kwargs.get(
                "sample_points", DEFAULT_SURFACE_POINTS
            ),
        )
        limit = max(EXACT_MAX_CHAMFER, EXACT_CHAMFER_NOISE_MULTIPLE * noise)
        ok = scores.voxel_iou >= EXACT_MIN_IOU and scores.chamfer_l2 <= limit
        return ok, {
            "scores": scores.to_dict(),
            "min_iou": EXACT_MIN_IOU,
            "sampling_noise_floor": noise,
            "max_chamfer": limit,
        }

    gates.append(_run_gate("shape_identical", _shape_identical))

    # --- action prefixes --------------------------------------------------
    if run_prefixes:

        def _prefixes_execute() -> tuple[bool, dict[str, Any]]:
            actions = decompose_actions(canonical, result_name=result_name)
            if max_prefix_checks > 0 and len(actions) > max_prefix_checks:
                step = len(actions) / max_prefix_checks
                selected = sorted(
                    {int(index * step) for index in range(max_prefix_checks)}
                    | {len(actions) - 1}
                )
            else:
                selected = list(range(len(actions)))

            chunks: list[str] = []
            checked = 0
            for index, action in enumerate(actions):
                chunks.append(action.code)
                if index not in selected:
                    continue
                try:
                    run("\n".join(chunks), f"prefix{index}")
                    checked += 1
                except Exception as error:
                    return False, {
                        "actions": len(actions),
                        "checked": checked,
                        "failed_index": index,
                        "error": f"{type(error).__name__}: {error}",
                    }
            return True, {"actions": len(actions), "checked": checked}

        gates.append(_run_gate("prefixes_execute", _prefixes_execute))
    else:
        gates.append(_skipped("prefixes_execute", "disabled"))

    # --- downstream binarization -----------------------------------------
    # Binarization is a separate legacy stage.  It can mangle or even break a
    # program on its own, so damage is attributed: a gate fails only when the
    # *canonical* program fares worse than the source under the same transform.
    if run_quantization:
        from utils.canonicalization.cc_for_validation import binarize_numeric_literals

        binarized: dict[str, Any] = {}

        def _binarize_both() -> None:
            for label, code in (("source", source), ("canonical", canonical)):
                entry: dict[str, Any] = {}
                try:
                    entry["code"] = binarize_numeric_literals(code)
                except Exception as error:
                    entry["error"] = f"transform: {type(error).__name__}: {error}"
                    binarized[label] = entry
                    continue
                try:
                    entry["result"] = _result_of(
                        run(entry["code"], f"q{label}"), result_name
                    )
                    entry["metrics"] = shape_metrics(entry["result"])
                except Exception as error:
                    entry["error"] = f"build: {type(error).__name__}: {error}"
                binarized[label] = entry

        def _quantization_commutes() -> tuple[bool, dict[str, Any]]:
            _binarize_both()
            left, right = binarized["source"], binarized["canonical"]
            left_ok = "metrics" in left
            right_ok = "metrics" in right
            if not left_ok and not right_ok:
                # The legacy binarizer cannot handle this program at all.  That is
                # a property of the binarizer, not of canonicalization.
                return True, {
                    "not_applicable": "binarized source does not build either",
                    "source_error": left.get("error"),
                    "canonical_error": right.get("error"),
                }
            if left_ok != right_ok:
                return False, {
                    "source_builds": left_ok,
                    "canonical_builds": right_ok,
                    "source_error": left.get("error"),
                    "canonical_error": right.get("error"),
                }
            mismatches = compare_metrics(
                left["metrics"],
                right["metrics"],
                relative_tolerance=1e-6,
                absolute_tolerance=1e-6,
            )
            return not mismatches, {
                "mismatches": mismatches,
                "faces": left["metrics"].faces,
            }

        gates.append(_run_gate("quantization_commutes", _quantization_commutes))

        def _quantized_shape_close() -> tuple[bool, dict[str, Any]]:
            if not binarized:
                _binarize_both()
            canonical_entry = binarized.get("canonical", {})
            source_entry = binarized.get("source", {})
            if "metrics" not in canonical_entry:
                return True, {
                    "not_applicable": "binarized program does not build",
                    "error": canonical_entry.get("error"),
                }
            scores = compare_shapes(
                source_result, canonical_entry["result"], **similarity_kwargs
            )
            detail: dict[str, Any] = {
                "scores": scores.to_dict(),
                "min_iou": QUANTIZED_MIN_IOU,
                "max_chamfer": QUANTIZED_MAX_CHAMFER,
            }
            # How much of the divergence is the binarizer's own doing?
            if "metrics" in source_entry:
                baseline = compare_shapes(
                    source_result, source_entry["result"], **similarity_kwargs
                )
                detail["binarizer_baseline"] = baseline.to_dict()
                detail["iou_lost_to_canonicalization"] = (
                    baseline.voxel_iou - scores.voxel_iou
                )
                # CC-for is only at fault for the gap against that baseline, and
                # only beyond the voxel discretization floor described above.
                return scores.voxel_iou >= baseline.voxel_iou - VOXEL_IOU_SLACK, detail
            return (
                scores.voxel_iou >= QUANTIZED_MIN_IOU
                and scores.chamfer_l2 <= QUANTIZED_MAX_CHAMFER
            ), detail

        gates.append(_run_gate("quantized_shape_close", _quantized_shape_close))
    else:
        gates.append(_skipped("quantization_commutes", "disabled"))
        gates.append(_skipped("quantized_shape_close", "disabled"))

    # --- symbolic robustness ---------------------------------------------
    if run_perturbations:

        def _parameter_perturbation() -> tuple[bool, dict[str, Any]]:
            pairs = _perturbable_parameters(source, canonical, comparison, report)
            if not pairs:
                return True, {"checked": 0, "reason": "no plain float parameters"}
            cases: list[dict[str, Any]] = []
            for source_name, canonical_name in pairs[:max_perturbations]:
                perturbed_source = _scale_numeric_parameter(
                    source, source_name, PERTURBATION_FACTOR
                )
                perturbed_canonical = _scale_numeric_parameter(
                    canonical, canonical_name, PERTURBATION_FACTOR
                )
                if perturbed_source is None or perturbed_canonical is None:
                    continue
                case: dict[str, Any] = {"parameter": source_name}
                try:
                    left = shape_metrics(
                        _result_of(run(perturbed_source, "pert-source"), result_name)
                    )
                    right = shape_metrics(
                        _result_of(
                            run(perturbed_canonical, "pert-canonical"), result_name
                        )
                    )
                except Exception as error:
                    # Some parameters are not freely scalable (a boss larger than
                    # its plate).  A build failure in *both* programs is not a
                    # canonicalization defect, so only a one-sided failure counts.
                    left_ok = right_ok = False
                    try:
                        shape_metrics(
                            _result_of(
                                run(perturbed_source, "pert-source"), result_name
                            )
                        )
                        left_ok = True
                    except Exception:
                        pass
                    try:
                        shape_metrics(
                            _result_of(
                                run(perturbed_canonical, "pert-canonical"), result_name
                            )
                        )
                        right_ok = True
                    except Exception:
                        pass
                    case.update(
                        {
                            "passed": left_ok == right_ok,
                            "both_failed": not left_ok and not right_ok,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    cases.append(case)
                    continue

                mismatches = compare_metrics(
                    left,
                    right,
                    relative_tolerance=EXACT_RELATIVE_TOLERANCE,
                    absolute_tolerance=EXACT_ABSOLUTE_TOLERANCE,
                )
                # The perturbation must actually do something, otherwise the gate
                # would pass on a program that ignores the parameter entirely.
                baseline = shape_metrics(source_result)
                moved = not math.isclose(
                    baseline.volume, left.volume, rel_tol=1e-9, abs_tol=1e-9
                )
                case.update(
                    {
                        "passed": not mismatches,
                        "changed_geometry": moved,
                        "mismatches": mismatches[:4],
                    }
                )
                cases.append(case)
            checked = [case for case in cases if "passed" in case]
            return all(case["passed"] for case in checked), {
                "checked": len(checked),
                "effective": sum(1 for case in checked if case.get("changed_geometry")),
                "cases": cases,
            }

        gates.append(_run_gate("parameter_perturbation", _parameter_perturbation))
    else:
        gates.append(_skipped("parameter_perturbation", "disabled"))

    evaluation.passed = all(gate.passed for gate in gates)
    evaluation.seconds = time.monotonic() - started
    return evaluation
