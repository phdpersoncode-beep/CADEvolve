"""Reproducible, crash-isolated CC-step/source/CADEvolve geometry audit.

Each JSONL row records a completed input immediately. Statuses distinguish source
failures, conversion failures, unstable sources, and successful comparisons.
Signature agreement is a screening test; use --boolean for occupied-volume checks.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time

from utils.isolated import iter_isolated


def audit_one(path_string, boolean=False, trace=False):
    from utils.canonicalization.cc_for import CCForConfig, canonicalize_code
    from utils.canonicalization.cc_for_validation import compare_solid_geometry
    from evals.cc_for.geometry import compare_metrics, shape_metrics
    from evals.cc_for.harness import execute
    from evals.cc_for.representations import build_cadevolve_c

    path = Path(path_string)
    source = path.read_text(encoding="utf-8")
    record = {"name": path.name, "source_sha256": hashlib.sha256(source.encode()).hexdigest()}
    started = time.monotonic()
    try:
        original = execute(source)["result"]
        metrics = shape_metrics(original)
        record["source_metrics"] = metrics.to_dict()
        if not metrics.valid or metrics.solids < 1 or metrics.volume <= 0:
            record.update(status="source_invalid")
            return record
    except Exception as error:
        record.update(status="source_error", error=f"{type(error).__name__}: {error}")
        return record
    stage = "canonical"
    try:
        converted = canonicalize_code(source, CCForConfig(parameter_placement="late"))
        record["canonical_sha256"] = hashlib.sha256(converted.code.encode()).hexdigest()
        record["structural_errors"] = converted.report.structural_errors
        canonical = execute(converted.code)["result"]
        record["canonical_metrics"] = shape_metrics(canonical).to_dict()
        stage = "comparison"
        mismatch = compare_metrics(metrics, shape_metrics(canonical), relative_tolerance=1e-7, absolute_tolerance=1e-7)
        record["mismatches"] = mismatch
        record["status"] = "signature_mismatch" if mismatch else "match"
        if not mismatch and boolean:
            check = compare_solid_geometry(original, canonical)
            record["boolean"] = check
            if not check["equivalent"]:
                record["status"] = "geometry_mismatch"
    except Exception as error:
        record.update(status=f"{stage}_error", error=f"{type(error).__name__}: {error}")
    # Rebuild the source to separate observed nondeterminism from transform defects.
    try:
        repeated = shape_metrics(execute(source)["result"])
        differences = compare_metrics(metrics, repeated, relative_tolerance=1e-7, absolute_tolerance=1e-7)
        if differences:
            record.update(status="source_unstable", source_repeat_mismatches=differences)
    except Exception as error:
        record.update(status="source_unstable", source_repeat_error=f"{type(error).__name__}: {error}")
    if trace and record["status"] == "match":
        try:
            traced_code = build_cadevolve_c(source)
            traced = execute(traced_code)["result"]
            record["tracer_metrics"] = shape_metrics(traced).to_dict()
            check = compare_solid_geometry(original, traced)
            record["tracer_boolean"] = check
            record["tracer_status"] = "match" if check["equivalent"] else "geometry_mismatch"
        except Exception as error:
            record.update(tracer_status="error", tracer_error=f"{type(error).__name__}: {error}")
    record["seconds"] = time.monotonic() - started
    return record


def main():
    from evals.cc_for.run_eval import collect_sources

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="demo")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--boolean", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--repeats", type=int, default=1,
                        help="fresh-process repetitions of each selected source")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    counts, tracer_counts = Counter(), Counter()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="cc-step-audit-") as scratch:
        sources = collect_sources(args.corpus, None, 0, Path(scratch))
        if args.limit is not None:
            sources = sorted(random.Random(args.seed).sample(sources, min(args.limit, len(sources))))
        sources = [path for path in sources for _ in range(args.repeats)]
        jobs = [(str(path), args.boolean, args.trace) for path in sources]
        with args.report.open("w", encoding="utf-8") as stream:
            for finished in iter_isolated(audit_one, jobs, workers=args.workers, timeout=args.timeout):
                record = finished.value if finished.error is None else {
                    "name": sources[finished.index].name, "status": "worker_failure", "error": finished.error,
                }
                record["repeat"] = finished.index % args.repeats
                counts[record["status"]] += 1
                if "tracer_status" in record:
                    tracer_counts[record["tracer_status"]] += 1
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                total = sum(counts.values())
                if total % 50 == 0 or record["status"] != "match" or total == len(sources):
                    print(f"{total}/{len(sources)} {dict(counts)}", flush=True)
    summary = {"counts": dict(counts), "tracer_counts": dict(tracer_counts),
               "seconds": time.monotonic() - started,
               "options": {**vars(args), "report": str(args.report)}}
    args.report.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
