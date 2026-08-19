"""Column-projected CC-for validation for the Zero-to-CAD Hugging Face dataset.

The dataset's Parquet shards also contain rendered images, STL, and STEP payloads.
This runner asks DuckDB for only ``uuid`` and ``cadquery_file`` so Parquet projection
pushdown avoids materializing those large binary columns.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from utils.canonicalization.cc_for import CCForConfig, canonicalize_code
from utils.canonicalization.cc_for_validation import (
    validate_prefixes,
    validate_quantized_geometry,
    validate_round_trip,
)


DEFAULT_DATASET = "ADSKAILab/Zero-To-CAD-100k"
PARQUET_ENDPOINT = "https://datasets-server.huggingface.co/parquet"


@dataclass(frozen=True)
class ParquetShard:
    split: str
    filename: str
    url: str
    size: int


@dataclass
class CorpusReport:
    dataset: str
    revision: str | None
    loop_mode: str
    splits: list[str]
    started_at: str
    parquet_shards: int = 0
    skipped_shards: int = 0
    processed_shards: int = 0
    projected_columns: list[str] = field(
        default_factory=lambda: ["uuid", "cadquery_file"]
    )
    rows: int = 0
    structural_passed: int = 0
    structural_failed: int = 0
    source_bytes: int = 0
    canonical_bytes: int = 0
    flattened_namespaces: int = 0
    preserved_loops: int = 0
    unrolled_loops: int = 0
    workplane_steps: int = 0
    structural_errors: Counter[str] = field(default_factory=Counter)
    warnings: Counter[str] = field(default_factory=Counter)
    shard_errors: list[str] = field(default_factory=list)
    execution_sample_size: int = 0
    round_trip_passed: int = 0
    round_trip_failed: int = 0
    prefix_passed: int = 0
    prefix_failed: int = 0
    quantized_geometry_passed: int = 0
    quantized_geometry_failed: int = 0
    quantized_chamfer_distances: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["structural_errors"] = dict(
            sorted(self.structural_errors.items())
        )
        payload["warnings"] = dict(sorted(self.warnings.items()))
        total = self.structural_passed + self.structural_failed
        payload["structural_pass_rate"] = (
            self.structural_passed / total if total else 0.0
        )
        distances = sorted(self.quantized_chamfer_distances)
        if distances:
            percentile_index = max(0, (95 * len(distances) + 99) // 100 - 1)
            payload["quantized_chamfer_mean"] = sum(distances) / len(distances)
            payload["quantized_chamfer_p95"] = distances[percentile_index]
            payload["quantized_chamfer_max"] = distances[-1]
        else:
            payload["quantized_chamfer_mean"] = None
            payload["quantized_chamfer_p95"] = None
            payload["quantized_chamfer_max"] = None
        return payload


def fetch_parquet_manifest(
    dataset: str, *, timeout_seconds: float = 60.0
) -> tuple[str | None, list[ParquetShard]]:
    try:
        import requests
    except ImportError as error:  # pragma: no cover - environment guidance
        raise RuntimeError(
            "requests is required for the Hugging Face validation runner"
        ) from error

    response = requests.get(
        PARQUET_ENDPOINT,
        params={"dataset": dataset},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    failures = payload.get("failed") or []
    if failures:
        raise RuntimeError(f"dataset viewer reported manifest failures: {failures}")
    shards = [
        ParquetShard(
            split=str(item["split"]),
            filename=str(item["filename"]),
            url=str(item["url"]),
            size=int(item["size"]),
        )
        for item in payload.get("parquet_files", [])
    ]
    return response.headers.get("x-revision"), shards


def _batched(values: Sequence[ParquetShard], size: int) -> Iterator[list[ParquetShard]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _decode_code(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8")
    raise TypeError(f"unsupported cadquery_file value: {type(value).__name__}")


def _open_duckdb(extension_directory: Path, threads: int) -> Any:
    try:
        import duckdb
    except ImportError as error:  # pragma: no cover - environment guidance
        raise RuntimeError(
            "duckdb is required for projected remote Parquet reads"
        ) from error

    extension_directory.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    escaped = str(extension_directory).replace("'", "''")
    connection.execute(f"SET extension_directory='{escaped}'")
    connection.execute(f"SET threads={max(1, threads)}")
    connection.execute("SET enable_progress_bar=false")
    return connection


def _read_projected_rows(
    connection: Any, shards: Sequence[ParquetShard], fetch_size: int
) -> Iterator[tuple[str, Any]]:
    urls = [shard.url for shard in shards]
    cursor = connection.execute(
        "SELECT uuid, cadquery_file FROM read_parquet(?)", [urls]
    )
    while True:
        rows = cursor.fetchmany(fetch_size)
        if not rows:
            break
        yield from rows


def _record_failure(
    failures: list[dict[str, Any]],
    *,
    maximum: int,
    uuid: str,
    split: str,
    errors: Iterable[str],
    row_index: int,
    shard_batch: str,
) -> None:
    if len(failures) >= maximum:
        return
    failures.append(
        {
            "uuid": uuid,
            "split": split,
            "row_index": row_index,
            "shard_batch": shard_batch,
            "errors": list(errors),
        }
    )


def _write_failure_source(
    directory: Path | None, split: str, uuid: str, source: str
) -> None:
    if directory is None:
        return
    destination = directory / split / f"{uuid}.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _report_payload(
    report: CorpusReport,
    failures: Sequence[dict[str, Any]],
    execution_failures: Sequence[dict[str, Any]],
    *,
    complete: bool,
) -> dict[str, Any]:
    """Build a durable report for both partial and completed corpus scans."""

    return {
        "complete": complete,
        "next_shard_offset": report.skipped_shards + report.processed_shards,
        "summary": report.to_dict(),
        "failure_examples": list(failures),
        "execution_failures": list(execution_failures),
    }


def validate_corpus(
    *,
    dataset: str,
    splits: Sequence[str],
    loop_mode: str,
    report_path: Path,
    max_rows: int | None = None,
    max_shards: int | None = None,
    skip_shards: int = 0,
    shard_batch_size: int = 8,
    fetch_size: int = 512,
    threads: int = 8,
    max_unroll_iterations: int = 64,
    max_failure_records: int = 100,
    execution_samples: int = 0,
    validate_sample_prefixes: bool = False,
    validate_sample_quantization: bool = False,
    quantized_surface_points: int = 4_096,
    quantized_chamfer_threshold: float = 0.15,
    random_seed: int = 0,
    extension_directory: Path | None = None,
    shard_retries: int = 2,
    progress: bool = False,
    failure_source_dir: Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    revision, manifest = fetch_parquet_manifest(dataset)
    selected = [
        shard
        for split in splits
        for shard in sorted(
            (item for item in manifest if item.split == split),
            key=lambda item: item.filename,
        )
    ]
    selected = selected[max(0, skip_shards) :]
    if max_shards is not None:
        selected = selected[:max_shards]
    if not selected:
        raise RuntimeError(
            f"no Parquet shards found for requested splits: {list(splits)}"
        )

    report = CorpusReport(
        dataset=dataset,
        revision=revision,
        loop_mode=loop_mode,
        splits=list(splits),
        started_at=datetime.now(UTC).isoformat(),
        parquet_shards=len(selected),
        skipped_shards=max(0, skip_shards),
    )
    failures: list[dict[str, Any]] = []
    reservoir: list[tuple[str, str, str]] = []
    rng = random.Random(random_seed)
    seen_for_reservoir = 0
    extension_directory = extension_directory or (
        Path(tempfile.gettempdir()) / "cc-for-duckdb-extensions"
    )
    connection = _open_duckdb(extension_directory, threads)
    stop = False

    for split in splits:
        split_shards = [shard for shard in selected if shard.split == split]
        for batch in _batched(split_shards, max(1, shard_batch_size)):
            if stop:
                break
            batch_rows: list[tuple[str, Any]] | None = None
            last_error: Exception | None = None
            for _attempt in range(max(0, shard_retries) + 1):
                try:
                    # Buffer only the projected source columns. This makes retries
                    # idempotent if a remote range request fails mid-batch.
                    batch_rows = list(
                        _read_projected_rows(connection, batch, fetch_size)
                    )
                    break
                except Exception as error:
                    last_error = error
            if batch_rows is None:
                assert last_error is not None
                report.shard_errors.append(
                    f"{split}/{batch[0].filename}..{batch[-1].filename}: "
                    f"{type(last_error).__name__}: {last_error}"
                )
                report.processed_shards += len(batch)
                report.elapsed_seconds = round(time.monotonic() - started, 6)
                _atomic_json(
                    report_path,
                    _report_payload(
                        report, failures, [], complete=False
                    ),
                )
                continue

            for uuid, encoded_code in batch_rows:
                if max_rows is not None and report.rows >= max_rows:
                    stop = True
                    break
                report.rows += 1
                source: str | None = None
                shard_batch = f"{batch[0].filename}..{batch[-1].filename}"
                try:
                    source = _decode_code(encoded_code)
                    report.source_bytes += len(source.encode("utf-8"))
                    converted = canonicalize_code(
                        source,
                        CCForConfig(
                            loop_mode=loop_mode,  # type: ignore[arg-type]
                            max_unroll_iterations=max_unroll_iterations,
                        ),
                    )
                    report.canonical_bytes += len(
                        converted.code.encode("utf-8")
                    )
                    conversion_report = converted.report
                    report.flattened_namespaces += len(
                        conversion_report.flattened_namespaces
                    )
                    report.preserved_loops += conversion_report.preserved_loops
                    report.unrolled_loops += conversion_report.unrolled_loops
                    report.workplane_steps += conversion_report.workplane_steps
                    report.warnings.update(conversion_report.warnings)
                    if conversion_report.structural_errors:
                        report.structural_failed += 1
                        report.structural_errors.update(
                            conversion_report.structural_errors
                        )
                        _record_failure(
                            failures,
                            maximum=max_failure_records,
                            uuid=str(uuid),
                            split=split,
                            errors=conversion_report.structural_errors,
                            row_index=report.rows - 1,
                            shard_batch=shard_batch,
                        )
                        _write_failure_source(
                            failure_source_dir, split, str(uuid), source
                        )
                        continue
                    report.structural_passed += 1

                    if execution_samples > 0:
                        seen_for_reservoir += 1
                        item = (str(uuid), source, converted.code)
                        if len(reservoir) < execution_samples:
                            reservoir.append(item)
                        else:
                            replacement = rng.randrange(seen_for_reservoir)
                            if replacement < execution_samples:
                                reservoir[replacement] = item
                except Exception as error:
                    message = f"{type(error).__name__}: {error}"
                    report.structural_failed += 1
                    report.structural_errors[message] += 1
                    _record_failure(
                        failures,
                        maximum=max_failure_records,
                        uuid=str(uuid),
                        split=split,
                        errors=[message],
                        row_index=report.rows - 1,
                        shard_batch=shard_batch,
                    )
                    if source is not None:
                        _write_failure_source(
                            failure_source_dir, split, str(uuid), source
                        )
            if progress:
                print(
                    f"validated={report.rows} "
                    f"passed={report.structural_passed} "
                    f"failed={report.structural_failed}",
                    file=sys.stderr,
                    flush=True,
                )
            if not stop:
                report.processed_shards += len(batch)
            report.elapsed_seconds = round(time.monotonic() - started, 6)
            _atomic_json(
                report_path,
                _report_payload(report, failures, [], complete=False),
            )

    report.execution_sample_size = len(reservoir)
    execution_failures: list[dict[str, Any]] = []
    for uuid, source, canonical in reservoir:
        round_trip = validate_round_trip(source, canonical)
        if round_trip.success:
            report.round_trip_passed += 1
        else:
            report.round_trip_failed += 1
            execution_failures.append(
                {"uuid": uuid, "check": "round_trip", **round_trip.to_dict()}
            )
        if validate_sample_prefixes:
            prefixes = validate_prefixes(canonical)
            if prefixes.success:
                report.prefix_passed += 1
            else:
                report.prefix_failed += 1
                execution_failures.append(
                    {"uuid": uuid, "check": "prefix", **prefixes.to_dict()}
                )
        if validate_sample_quantization:
            quantized = validate_quantized_geometry(
                source,
                canonical,
                sample_points=quantized_surface_points,
                random_seed=random_seed,
                chamfer_threshold=quantized_chamfer_threshold,
            )
            if quantized.raw_to_quantized_chamfer is not None:
                report.quantized_chamfer_distances.append(
                    quantized.raw_to_quantized_chamfer
                )
            if quantized.success:
                report.quantized_geometry_passed += 1
            else:
                report.quantized_geometry_failed += 1
                execution_failures.append(
                    {
                        "uuid": uuid,
                        "check": "quantized_geometry",
                        **quantized.to_dict(),
                    }
                )

    report.elapsed_seconds = round(time.monotonic() - started, 6)
    payload = _report_payload(
        report,
        failures,
        execution_failures,
        complete=not report.shard_errors,
    )
    _atomic_json(report_path, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate CC-for against projected CadQuery source from Zero-to-CAD"
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--split",
        action="append",
        dest="splits",
        choices=("train", "validation", "test"),
        help="repeat to select splits; defaults to all three",
    )
    parser.add_argument("--loop-mode", choices=("preserve", "unroll"), default="preserve")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--skip-shards", type=int, default=0)
    parser.add_argument("--shard-batch-size", type=int, default=8)
    parser.add_argument("--fetch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-unroll-iterations", type=int, default=64)
    parser.add_argument("--max-failure-records", type=int, default=100)
    parser.add_argument("--execution-samples", type=int, default=0)
    parser.add_argument("--validate-sample-prefixes", action="store_true")
    parser.add_argument("--validate-sample-quantization", action="store_true")
    parser.add_argument("--quantized-surface-points", type=int, default=4_096)
    parser.add_argument("--quantized-chamfer-threshold", type=float, default=0.15)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--extension-directory", type=Path)
    parser.add_argument("--shard-retries", type=int, default=2)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--failure-source-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = validate_corpus(
        dataset=args.dataset,
        splits=args.splits or ["train", "validation", "test"],
        loop_mode=args.loop_mode,
        report_path=args.report,
        max_rows=args.max_rows,
        max_shards=args.max_shards,
        skip_shards=args.skip_shards,
        shard_batch_size=args.shard_batch_size,
        fetch_size=args.fetch_size,
        threads=args.threads,
        max_unroll_iterations=args.max_unroll_iterations,
        max_failure_records=args.max_failure_records,
        execution_samples=args.execution_samples,
        validate_sample_prefixes=args.validate_sample_prefixes,
        validate_sample_quantization=args.validate_sample_quantization,
        quantized_surface_points=args.quantized_surface_points,
        quantized_chamfer_threshold=args.quantized_chamfer_threshold,
        random_seed=args.random_seed,
        extension_directory=args.extension_directory,
        shard_retries=args.shard_retries,
        progress=args.progress,
        failure_source_dir=args.failure_source_dir,
    )
    summary = payload["summary"]
    console_summary = {
        key: summary[key]
        for key in (
            "dataset",
            "revision",
            "loop_mode",
            "splits",
            "parquet_shards",
            "rows",
            "structural_passed",
            "structural_failed",
            "structural_pass_rate",
            "execution_sample_size",
            "round_trip_passed",
            "round_trip_failed",
            "prefix_passed",
            "prefix_failed",
            "quantized_geometry_passed",
            "quantized_geometry_failed",
            "quantized_chamfer_mean",
            "quantized_chamfer_p95",
            "quantized_chamfer_max",
            "elapsed_seconds",
        )
    }
    console_summary["warning_messages"] = len(summary["warnings"])
    console_summary["warning_occurrences"] = sum(summary["warnings"].values())
    console_summary["report"] = str(args.report)
    print(json.dumps(console_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
