"""Batch entry point for symbol-preserving CC-for canonicalization."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from utils.canonicalization.cc_for import CCForConfig, canonicalize_code
from utils.canonicalization.cc_for_validation import (
    validate_parameter_perturbations,
    validate_prefixes,
    validate_round_trip,
)


@dataclass(frozen=True)
class PipelineConfig:
    root_dir: Path
    out_dir: Path
    report_path: Path
    flat: bool = False
    n_workers: int = 1
    loop_mode: str = "preserve"
    max_unroll_iterations: int = 64
    validate_execution: bool = True
    validate_prefix_execution: bool = True
    validate_perturbations: bool = False
    max_perturbed_parameters: int = 16
    keep_failed: bool = False


def load_config(path: Path) -> PipelineConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return PipelineConfig(
        root_dir=Path(raw["root_dir"]),
        out_dir=Path(raw["out_dir"]),
        report_path=Path(raw.get("report_path", "logs/cc_for.jsonl")),
        flat=bool(raw.get("flat", False)),
        n_workers=max(1, int(raw.get("n_workers", 1))),
        loop_mode=str(raw.get("loop_mode", "preserve")),
        max_unroll_iterations=int(raw.get("max_unroll_iterations", 64)),
        validate_execution=bool(raw.get("validate_execution", True)),
        validate_prefix_execution=bool(
            raw.get("validate_prefix_execution", True)
        ),
        validate_perturbations=bool(raw.get("validate_perturbations", False)),
        max_perturbed_parameters=int(raw.get("max_perturbed_parameters", 16)),
        keep_failed=bool(raw.get("keep_failed", False)),
    )


def _output_path(source: Path, config: PipelineConfig) -> Path:
    relative = source.relative_to(config.root_dir)
    if config.flat:
        flattened = "__".join(relative.parts)
        return config.out_dir / flattened
    return config.out_dir / relative


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _process_one(source_string: str, config_data: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(source_string)
    config = PipelineConfig(**config_data)
    destination = _output_path(source_path, config)
    started = time.monotonic()
    record: dict[str, Any] = {
        "source": str(source_path),
        "output": str(destination),
        "success": False,
        "written": False,
    }

    try:
        original = source_path.read_text(encoding="utf-8")
        conversion = canonicalize_code(
            original,
            CCForConfig(
                loop_mode=config.loop_mode,  # type: ignore[arg-type]
                max_unroll_iterations=config.max_unroll_iterations,
            ),
        )
        record["canonicalization"] = conversion.report.to_dict()
        success = not conversion.report.structural_errors

        if config.validate_execution:
            round_trip = validate_round_trip(original, conversion.code)
            record["round_trip"] = round_trip.to_dict()
            success = success and round_trip.success

        if config.validate_prefix_execution:
            prefixes = validate_prefixes(conversion.code)
            record["prefixes"] = prefixes.to_dict()
            success = success and prefixes.success

        if config.validate_perturbations:
            perturbations = validate_parameter_perturbations(
                original,
                conversion,
                max_parameters=config.max_perturbed_parameters,
            )
            record["perturbations"] = perturbations.to_dict()
            success = success and perturbations.success

        record["success"] = success
        if success or config.keep_failed:
            _atomic_write(destination, conversion.code)
            record["written"] = True
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"

    record["elapsed_seconds"] = round(time.monotonic() - started, 6)
    return record


def _discover_sources(config: PipelineConfig) -> list[Path]:
    if not config.root_dir.is_dir():
        raise FileNotFoundError(f"input root does not exist: {config.root_dir}")
    output_resolved = config.out_dir.resolve()
    files = []
    for path in config.root_dir.rglob("*.py"):
        try:
            path.resolve().relative_to(output_resolved)
            continue
        except ValueError:
            files.append(path)
    return sorted(files)


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    sources = _discover_sources(config)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config_data = asdict(config)

    records: dict[str, dict[str, Any]] = {}
    if config.n_workers == 1:
        for source in sources:
            records[str(source)] = _process_one(str(source), config_data)
    else:
        with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
            futures = {
                executor.submit(_process_one, str(source), config_data): str(source)
                for source in sources
            }
            for future in as_completed(futures):
                source = futures[future]
                try:
                    records[source] = future.result()
                except Exception as error:
                    records[source] = {
                        "source": source,
                        "success": False,
                        "written": False,
                        "error": f"worker {type(error).__name__}: {error}",
                    }

    ordered = [records[str(source)] for source in sources]
    with config.report_path.open("w", encoding="utf-8") as stream:
        for record in ordered:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    succeeded = sum(bool(record.get("success")) for record in ordered)
    summary = {
        "total": len(ordered),
        "succeeded": succeeded,
        "failed": len(ordered) - succeeded,
        "written": sum(bool(record.get("written")) for record in ordered),
        "report_path": str(config.report_path),
    }
    summary_path = config.report_path.with_suffix(".summary.json")
    _atomic_write(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert sampled CADEvolve-P or Zero-to-CAD scripts to CC-for"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    summary = run_pipeline(load_config(args.config))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

