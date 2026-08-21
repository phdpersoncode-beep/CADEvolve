"""CLI: run the CC-for evaluation gates over a corpus of CadQuery programs.

Each program is evaluated in a worker process with a wall-clock timeout, because
executing arbitrary OpenCascade programs can hang or abort the interpreter and one
bad program must not take the run down with it.

Examples
--------
Edge cases written for this suite, all gates::

    PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_eval \\
        --corpus cases --report /tmp/cases.json

A sample of the checked-in Zero-to-CAD 5K snapshot::

    PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_eval \\
        --corpus demo --limit 200 --workers 8 --report /tmp/demo.json

Structure-only scan of the whole snapshot (no CadQuery needed)::

    PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_eval \\
        --corpus demo --no-geometry --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tarfile
import tempfile
import time
import warnings
from concurrent.futures import (
    ProcessPoolExecutor,
    TimeoutError as FutureTimeout,
    as_completed,
)
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_ARCHIVE = REPO_ROOT / "demo_data" / "zero_to_cad_5k" / "raw_sources.tar.gz"
CASES_DIR = Path(__file__).resolve().parent / "cases"
ZERO_TO_CAD_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "zero_to_cad"
CADEVOLVE_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "cadevolve_p"

GATE_ORDER = (
    "converts",
    "structure",
    "idempotent",
    "parameters_preserved",
    "parameters_hoisted",
    "loops_preserved",
    "loops_unrolled",
    "literals_stable",
    "chains_lowered",
    "source_executes",
    "canonical_executes",
    "topology_identical",
    "shape_identical",
    "prefixes_execute",
    "quantization_commutes",
    "quantized_shape_close",
    "parameter_perturbation",
)


def _extract_demo_archive(destination: Path) -> Path:
    if not DEMO_ARCHIVE.exists():
        raise SystemExit(f"demo archive not found: {DEMO_ARCHIVE}")
    with tarfile.open(DEMO_ARCHIVE, "r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith(".py")
        ]
        for member in members:
            # Refuse absolute or parent-escaping member paths.
            target = (destination / member.name).resolve()
            if not str(target).startswith(str(destination.resolve())):
                raise SystemExit(f"unsafe archive member: {member.name}")
        archive.extractall(destination, members=members)
    return destination


def collect_sources(corpus: str, limit: int | None, offset: int, workdir: Path) -> list[Path]:
    if corpus == "cases":
        paths = sorted(CASES_DIR.glob("*.py"))
    elif corpus == "fixtures":
        paths = sorted(ZERO_TO_CAD_FIXTURES.glob("*.py")) + sorted(
            CADEVOLVE_FIXTURES.glob("*.py")
        )
    elif corpus == "demo":
        root = _extract_demo_archive(workdir)
        paths = sorted(root.rglob("*.py"))
    else:
        root = Path(corpus)
        if root.is_file():
            paths = [root]
        elif root.is_dir():
            paths = sorted(root.rglob("*.py"))
        else:
            raise SystemExit(f"corpus not found: {corpus}")
    paths = [path for path in paths if path.name != "__init__.py"]
    paths = paths[offset:]
    if limit is not None:
        paths = paths[:limit]
    return paths


def _failure_record(name: str, error: str) -> dict[str, Any]:
    """A record for a program that never produced gate results."""

    return {
        "name": name,
        "passed": False,
        "failed_gates": ["converts"],
        "gates": [
            {
                "name": "converts",
                "passed": False,
                "skipped": False,
                "detail": {},
                "error": error,
                "seconds": 0.0,
            }
        ],
        "report": {},
        "code_comparison": {},
    }


_WORKER_OPTIONS: dict[str, Any] = {}


def _init_worker(options: dict[str, Any]) -> None:
    warnings.filterwarnings("ignore")
    _WORKER_OPTIONS.update(options)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "dataset_utils"))


def _evaluate_path(path_string: str) -> dict[str, Any]:
    from evals.cc_for.harness import evaluate_program

    path = Path(path_string)
    source = path.read_text(encoding="utf-8")
    evaluation = evaluate_program(source, name=path.name, **_WORKER_OPTIONS)
    return evaluation.to_dict()


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    gates: dict[str, dict[str, int]] = {}
    for record in records:
        for gate in record.get("gates", []):
            bucket = gates.setdefault(
                gate["name"], {"passed": 0, "failed": 0, "skipped": 0}
            )
            if gate.get("skipped"):
                bucket["skipped"] += 1
            elif gate["passed"]:
                bucket["passed"] += 1
            else:
                bucket["failed"] += 1

    def _scores(gate_name: str, key: str, key_path: str = "scores") -> list[float]:
        values: list[float] = []
        for record in records:
            for gate in record.get("gates", []):
                if gate["name"] != gate_name or gate.get("skipped"):
                    continue
                scores = gate.get("detail", {}).get(key_path)
                if scores and scores.get(key) is not None:
                    values.append(float(scores[key]))
        return values

    def _distribution(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        ordered = sorted(values)
        return {
            "count": len(ordered),
            "min": ordered[0],
            "mean": statistics.fmean(ordered),
            "median": statistics.median(ordered),
            "p05": ordered[max(0, int(0.05 * (len(ordered) - 1)))],
            "p95": ordered[int(0.95 * (len(ordered) - 1))],
            "max": ordered[-1],
        }

    retention = [
        record["code_comparison"]["parameter_retention"]
        for record in records
        if record.get("code_comparison")
    ]

    return {
        "programs": len(records),
        "passed": sum(1 for record in records if record.get("passed")),
        "failed": sum(1 for record in records if not record.get("passed")),
        "gates": {name: gates[name] for name in GATE_ORDER if name in gates},
        "parameter_retention": _distribution(retention),
        "exact_voxel_iou": _distribution(_scores("shape_identical", "voxel_iou")),
        "exact_chamfer_l2": _distribution(_scores("shape_identical", "chamfer_l2")),
        "quantized_voxel_iou": _distribution(_scores("quantized_shape_close", "voxel_iou")),
        "quantized_chamfer_l2": _distribution(
            _scores("quantized_shape_close", "chamfer_l2")
        ),
        "binarizer_baseline_iou": _distribution(
            _scores("quantized_shape_close", "voxel_iou", key_path="binarizer_baseline")
        ),
        "total_loops_preserved": sum(
            record.get("report", {}).get("preserved_loops", 0) for record in records
        ),
        "total_workplane_steps": sum(
            record.get("report", {}).get("workplane_steps", 0) for record in records
        ),
    }


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"programs {summary['programs']}  passed {summary['passed']}  failed {summary['failed']}",
        "",
        f"{'gate':24s} {'pass':>6s} {'fail':>6s} {'skip':>6s}",
    ]
    for name, counts in summary["gates"].items():
        lines.append(
            f"{name:24s} {counts['passed']:6d} {counts['failed']:6d} {counts['skipped']:6d}"
        )
    for label in (
        "parameter_retention",
        "exact_voxel_iou",
        "exact_chamfer_l2",
        "quantized_voxel_iou",
        "quantized_chamfer_l2",
        "binarizer_baseline_iou",
    ):
        distribution = summary.get(label)
        if distribution:
            lines.append(
                f"\n{label}: n={distribution['count']} min={distribution['min']:.6g} "
                f"mean={distribution['mean']:.6g} median={distribution['median']:.6g} "
                f"p05={distribution['p05']:.6g} p95={distribution['p95']:.6g} "
                f"max={distribution['max']:.6g}"
            )
    lines.append(
        f"\nloops preserved: {summary['total_loops_preserved']}   "
        f"workplane steps: {summary['total_workplane_steps']}"
    )
    return "\n".join(lines)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        default="cases",
        help="'cases', 'fixtures', 'demo', or a path to a file/directory of programs",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds per program")
    parser.add_argument(
        "--execution-timeout",
        type=float,
        default=90.0,
        help="wall-clock budget for one program execution; a CAD program can loop "
        "forever, and a perturbed one more easily still",
    )
    parser.add_argument(
        "--tasks-per-child",
        type=int,
        default=8,
        help="programs per worker before the pool is recreated; OpenCascade "
        "holds memory across programs, so long runs need periodic recycling",
    )
    parser.add_argument("--loop-mode", choices=("preserve", "unroll"), default="preserve")
    parser.add_argument("--no-geometry", action="store_true", help="AST gates only")
    parser.add_argument("--no-prefixes", action="store_true")
    parser.add_argument("--no-quantization", action="store_true")
    parser.add_argument("--no-perturbations", action="store_true")
    parser.add_argument("--voxel-resolution", type=int, default=None)
    parser.add_argument("--surface-points", type=int, default=None)
    parser.add_argument("--report", type=Path, default=None, help="write a JSON report")
    parser.add_argument(
        "--fail-under",
        type=float,
        default=1.0,
        help="exit non-zero when the pass rate falls below this fraction",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    options = {
        "loop_mode": args.loop_mode,
        "run_geometry": not args.no_geometry,
        "run_prefixes": not args.no_prefixes,
        "run_quantization": not args.no_quantization,
        "run_perturbations": not args.no_perturbations,
        "voxel_resolution": args.voxel_resolution,
        "surface_points": args.surface_points,
        "execution_timeout": args.execution_timeout,
    }

    with tempfile.TemporaryDirectory(prefix="cc-for-eval-") as scratch:
        sources = collect_sources(args.corpus, args.limit, args.offset, Path(scratch))
        if not sources:
            raise SystemExit("no programs to evaluate")

        started = time.monotonic()
        records: list[dict[str, Any]] = []
        # Workers are recycled by running the corpus in batches with a fresh pool
        # each time, rather than via max_tasks_per_child: that option forces the
        # spawn start method, which deadlocks on worker replacement here.
        # OpenCascade holds memory across programs, so unbounded workers grow
        # until the machine runs out.
        batch_size = max(1, args.workers * max(1, args.tasks_per_child))
        completed = 0
        for batch_start in range(0, len(sources), batch_size):
            batch = sources[batch_start : batch_start + batch_size]
            budget = args.timeout * (len(batch) / max(1, args.workers) + 1.0)
            with ProcessPoolExecutor(
                max_workers=args.workers, initializer=_init_worker, initargs=(options,)
            ) as pool:
                futures = {pool.submit(_evaluate_path, str(path)): path for path in batch}
                try:
                    for future in as_completed(futures, timeout=budget):
                        path = futures[future]
                        try:
                            record = future.result()
                        except Exception as error:
                            record = _failure_record(
                                path.name, f"{type(error).__name__}: {error}"
                            )
                        records.append(record)
                        completed += 1
                        if not args.quiet:
                            failed = record.get("failed_gates") or []
                            status = "ok  " if record.get("passed") else "FAIL"
                            note = f"  [{', '.join(failed)}]" if failed else ""
                            print(
                                f"[{completed}/{len(sources)}] {status} "
                                f"{record['name']}{note}",
                                flush=True,
                            )
                except FutureTimeout:
                    for future, path in futures.items():
                        if future.done():
                            continue
                        future.cancel()
                        records.append(
                            _failure_record(
                                path.name, f"batch budget of {budget:.0f}s exhausted"
                            )
                        )
                        completed += 1
                        if not args.quiet:
                            print(
                                f"[{completed}/{len(sources)}] FAIL {path.name}  "
                                "[timeout]",
                                flush=True,
                            )
                    for process in list(pool._processes.values()):
                        process.kill()

    records.sort(key=lambda record: record["name"])
    summary = summarize(records)
    summary["seconds"] = time.monotonic() - started
    summary["corpus"] = args.corpus
    summary["loop_mode"] = args.loop_mode

    print()
    print(format_summary(summary))

    failures = [record for record in records if not record.get("passed")]
    if failures:
        print(f"\n{len(failures)} failing program(s):")
        for record in failures[:40]:
            gates = record.get("failed_gates") or []
            print(f"  {record['name']}: {', '.join(gates) or 'unknown'}")
            for gate in record.get("gates", []):
                if not gate["passed"] and not gate.get("skipped") and gate.get("error"):
                    print(f"      {gate['name']}: {gate['error'][:200]}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"summary": summary, "records": records}, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nreport written to {args.report}")

    pass_rate = summary["passed"] / max(1, summary["programs"])
    return 0 if pass_rate >= args.fail_under else 1


if __name__ == "__main__":
    raise SystemExit(main())
