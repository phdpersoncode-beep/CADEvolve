"""CLI: check that every canonical representation of a program builds one solid.

Where ``run_eval`` scores one representation against its source, this compares
the representations against each other: CADEvolve-C (the executing tracer),
CC-for (one parameter preamble) and CC-step (a parameter group per step).

    PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_representations \\
        --corpus fixtures --report /tmp/representations.json

    PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_representations \\
        --corpus demo --limit 200 --workers 8

CADEvolve-C is optional by default.  Its tracer predates several CadQuery APIs
and simply cannot record some programs; that is a property of the legacy stage,
not of the representations under test, so such a program reports a skipped
comparison rather than a failure.  Pass ``--require-cadevolve-c`` to treat the
tracer failing as a failure.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (REPO_ROOT, REPO_ROOT / "dataset_utils"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from evals.cc_for.representations import (  # noqa: E402
    CADEVOLVE_C,
    CC_FOR,
    CC_STEP,
    COMPARISON_PAIRS,
    REPRESENTATIONS,
    SOURCE,
)
from evals.cc_for.run_eval import collect_sources  # noqa: E402

_WORKER_OPTIONS: dict[str, Any] = {}


def _init_worker(options: dict[str, Any]) -> None:
    warnings.filterwarnings("ignore")
    _WORKER_OPTIONS.update(options)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "dataset_utils"))


def _evaluate_path(path_string: str) -> dict[str, Any]:
    from evals.cc_for.representations import compare_representations

    path = Path(path_string)
    evaluation = compare_representations(
        path.read_text(encoding="utf-8"), name=path.name, **_WORKER_OPTIONS
    )
    return evaluation.to_dict()


def _failure_record(name: str, error: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": False,
        "seconds": 0.0,
        "builds": [{"name": SOURCE, "error": error}],
        "comparisons": [],
        "failed_comparisons": ["harness"],
        "error": error,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    builds: dict[str, dict[str, int]] = {
        name: {"built": 0, "failed": 0} for name in REPRESENTATIONS
    }
    pairs: dict[str, dict[str, int]] = {
        f"{left}_vs_{right}": {"passed": 0, "failed": 0, "skipped": 0}
        for left, right in COMPARISON_PAIRS
    }
    scores: dict[str, list[float]] = {key: [] for key in pairs}

    for record in records:
        for build in record.get("builds", []):
            bucket = builds.setdefault(build["name"], {"built": 0, "failed": 0})
            bucket["built" if build.get("metrics") else "failed"] += 1
        for comparison in record.get("comparisons", []):
            key = f"{comparison['left']}_vs_{comparison['right']}"
            bucket = pairs.setdefault(key, {"passed": 0, "failed": 0, "skipped": 0})
            if comparison.get("skipped"):
                bucket["skipped"] += 1
            elif comparison["passed"]:
                bucket["passed"] += 1
            else:
                bucket["failed"] += 1
            iou = (comparison.get("scores") or {}).get("voxel_iou")
            if iou is not None:
                scores.setdefault(key, []).append(float(iou))

    def _distribution(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        ordered = sorted(values)
        return {
            "count": len(ordered),
            "min": ordered[0],
            "mean": statistics.fmean(ordered),
            "median": statistics.median(ordered),
            "max": ordered[-1],
        }

    return {
        "programs": len(records),
        "passed": sum(1 for record in records if record.get("passed")),
        "failed": sum(1 for record in records if not record.get("passed")),
        "builds": builds,
        "comparisons": pairs,
        "voxel_iou": {
            key: _distribution(values) for key, values in scores.items() if values
        },
    }


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"programs {summary['programs']}  passed {summary['passed']}  "
        f"failed {summary['failed']}",
        "",
        f"{'representation':16s} {'built':>6s} {'failed':>7s}",
    ]
    for name, counts in summary["builds"].items():
        lines.append(f"{name:16s} {counts['built']:6d} {counts['failed']:7d}")

    lines += ["", f"{'comparison':28s} {'pass':>6s} {'fail':>6s} {'skip':>6s} {'iou':>10s}"]
    for name, counts in summary["comparisons"].items():
        distribution = summary["voxel_iou"].get(name)
        iou = f"{distribution['min']:.4f}" if distribution else "-"
        lines.append(
            f"{name:28s} {counts['passed']:6d} {counts['failed']:6d} "
            f"{counts['skipped']:6d} {iou:>10s}"
        )
    return "\n".join(lines)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        default="fixtures",
        help="'cases', 'fixtures', 'demo', or a path to a file/directory",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per program")
    parser.add_argument("--execution-timeout", type=float, default=90.0)
    parser.add_argument("--tracer-timeout", type=float, default=180.0)
    parser.add_argument("--loop-mode", choices=("preserve", "unroll"), default="preserve")
    parser.add_argument("--voxel-resolution", type=int, default=None)
    parser.add_argument("--surface-points", type=int, default=None)
    parser.add_argument(
        "--require-cadevolve-c",
        action="store_true",
        help="fail a program whose legacy tracer cannot record it, instead of "
        "reporting a skipped comparison",
    )
    parser.add_argument(
        "--skip-cadevolve-c",
        action="store_true",
        help="compare only the symbolic representations; skips the tracer "
        "subprocess entirely, which is much faster",
    )
    parser.add_argument("--report", type=Path, default=None, help="write a JSON report")
    parser.add_argument("--fail-under", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    include = list(REPRESENTATIONS)
    pairs = list(COMPARISON_PAIRS)
    if args.skip_cadevolve_c:
        include = [SOURCE, CC_FOR, CC_STEP]
        pairs = [pair for pair in pairs if CADEVOLVE_C not in pair]

    options = {
        "loop_mode": args.loop_mode,
        "include": include,
        "pairs": pairs,
        "execution_timeout": args.execution_timeout,
        "tracer_timeout": args.tracer_timeout,
        "voxel_resolution": args.voxel_resolution,
        "surface_points": args.surface_points,
        "require_cadevolve_c": args.require_cadevolve_c,
    }

    with tempfile.TemporaryDirectory(prefix="cc-representations-") as scratch:
        sources = collect_sources(args.corpus, args.limit, args.offset, Path(scratch))
        if not sources:
            raise SystemExit("no programs to evaluate")

        started = time.monotonic()
        records: list[dict[str, Any]] = []
        # OpenCascade holds memory across programs, so the pool is rebuilt every
        # batch rather than relying on max_tasks_per_child, which forces the
        # spawn start method and deadlocks on worker replacement here.
        batch_size = max(1, args.workers * 4)
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
                        if not args.quiet:
                            status = "ok  " if record["passed"] else "FAIL"
                            print(
                                f"[{len(records):4d}/{len(sources)}] {status} {record['name']}",
                                flush=True,
                            )
                except TimeoutError:
                    for future, path in futures.items():
                        if not future.done():
                            records.append(
                                _failure_record(path.name, "batch timeout")
                            )
                    for future in futures:
                        future.cancel()

    records.sort(key=lambda record: record["name"])
    summary = summarize(records)
    summary["corpus"] = args.corpus
    summary["loop_mode"] = args.loop_mode
    summary["seconds"] = time.monotonic() - started

    print()
    print(format_summary(summary))

    failing = [record for record in records if not record["passed"]]
    if failing:
        print(f"\n{len(failing)} failing program(s):")
        for record in failing[:20]:
            detail = ", ".join(record.get("failed_comparisons", [])) or "build"
            print(f"  {record['name']}: {detail}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"summary": summary, "programs": records}, indent=2),
            encoding="utf-8",
        )
        print(f"\nreport written to {args.report}")

    rate = summary["passed"] / max(1, summary["programs"])
    return 0 if rate >= args.fail_under else 1


if __name__ == "__main__":
    raise SystemExit(main())
