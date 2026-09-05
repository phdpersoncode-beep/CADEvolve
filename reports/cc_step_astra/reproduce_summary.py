"""Reproduce the conservative corpus decisions from the checked-in evidence.

Run from the repository root:
    PYTHONPATH=.:dataset_utils python reports/cc_step_astra/reproduce_summary.py
No CAD programs are executed. JSONL reports are gzip-compressed to limit size.
"""

from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from evals.cc_for.geometry import ShapeMetrics, compare_metrics


ROOT = Path(__file__).resolve().parent
REPORTS = (
    "audit-5k", "recheck-once", "recheck-repeated", "recheck-tail",
    "recheck-deep", "recheck-last",
)
# These three first-pass failures came from the evaluator's unregistered module,
# not from the source. Their ordinary-module execution was rechecked successfully.
NAMESPACE_FIXES = {
    "00919da8-625c-7c5c-687c-5e5e94c1b097.py",
    "00c9ed5f-e98e-d74b-5112-d83b0ebe7bf1.py",
    "0c621948-7dba-775e-7d46-331fbb9789cd.py",
}


def main():
    grouped = defaultdict(list)
    for report in REPORTS:
        with gzip.open(ROOT / f"{report}.jsonl.gz", "rt") as stream:
            for line in stream:
                row = json.loads(line)
                row["evidence_report"] = report
                grouped[row["name"]].append(row)
    with gzip.open(ROOT / "canonical_hashes.json.gz", "rt") as stream:
        hashes = {r["name"]: r for r in json.load(stream)}
    coverage = {r["name"]: r for r in json.loads((ROOT / "coverage_failures.json").read_text())}
    decisions = []
    for name, rows in sorted(grouped.items()):
        clean = [r for r in rows if not (
            name in NAMESPACE_FIXES and r["evidence_report"] == "audit-5k"
            and r.get("error") == "AttributeError: 'NoneType' object has no attribute '__dict__'"
        )]
        metrics = [ShapeMetrics(**r["source_metrics"]) for r in clean if "source_metrics" in r]
        unstable = any(r["status"] == "source_unstable" for r in clean)
        unstable |= any(compare_metrics(metrics[0], m, relative_tolerance=1e-7,
                                        absolute_tolerance=1e-7) for m in metrics[1:])
        unstable |= bool(metrics) and any(r["status"] == "source_error" for r in clean)
        if unstable:
            status = "source_unstable"
        elif any(r["status"] == "worker_failure" for r in clean):
            status = "worker_failure"
        elif all(r["status"] == "source_error" for r in clean):
            status = "source_error"
        elif any(r["status"] == "source_invalid" for r in clean):
            status = "source_invalid"
        elif all(r["status"] == "match" for r in clean):
            status = "signature_match"
        else:
            status = "unresolved"
        # The screen overlapped fixes. A final-code match must actually have
        # been executed; matching only an obsolete canonical output is insufficient.
        final_matches = [r for r in clean if r["status"] == "match"
                         and r.get("canonical_sha256") == hashes[name]["canonical_sha256"]]
        if status == "signature_match" and not final_matches:
            status = "unverified_final_code"
        decisions.append({
            **hashes[name], "status": status,
            "coverage_failures": coverage.get(name, {}).get("failed_gates", []),
            "observations": dict(Counter(r["status"] for r in rows)),
            "evidence_reports": sorted({r["evidence_report"] for r in rows}),
        })
    counts = dict(Counter(r["status"] for r in decisions))
    uniform = sum(r["status"] == "signature_match" and not r["coverage_failures"] for r in decisions)
    print(json.dumps({"counts": counts, "total": len(decisions),
                      "signature_and_coverage_candidates": uniform}, indent=2))
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in decisions).encode()
    (ROOT / "corpus_decisions.jsonl.gz").write_bytes(gzip.compress(payload, mtime=0))


if __name__ == "__main__":
    main()
