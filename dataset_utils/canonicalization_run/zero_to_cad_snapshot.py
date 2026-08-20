"""Export and validate a compact snapshot of raw Zero-to-CAD programs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from canonicalization_run.zero_to_cad_hf_validation import (
    DEFAULT_DATASET,
    _decode_code,
    _open_duckdb,
    _read_projected_rows,
    fetch_parquet_manifest,
)
from utils.canonicalization.cc_for import canonicalize_code


FORMAT_VERSION = 1
DEFAULT_COUNT = 5_000
DEFAULT_SPLIT = "train"
DEFAULT_ARCHIVE = Path("demo_data/zero_to_cad_5k/raw_sources.tar.gz")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _safe_member_name(uuid: str) -> str:
    allowed = "-0123456789_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if not uuid or any(character not in allowed for character in uuid):
        raise ValueError(f"unsafe UUID for archive member: {uuid!r}")
    return f"raw/{uuid}.py"


def _add_source(
    archive: tarfile.TarFile,
    *,
    member_name: str,
    source: bytes,
) -> None:
    member = tarfile.TarInfo(member_name)
    member.size = len(source)
    member.mode = 0o644
    member.mtime = 0
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    archive.addfile(member, io.BytesIO(source))


def export_snapshot(
    *,
    output_path: Path,
    manifest_path: Path,
    dataset: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    count: int = DEFAULT_COUNT,
    offset: int = 0,
    fetch_size: int = 512,
    threads: int = 8,
    extension_directory: Path | None = None,
) -> dict[str, Any]:
    """Project raw CadQuery source and store it in a deterministic archive."""

    if count <= 0:
        raise ValueError("count must be positive")
    if offset < 0:
        raise ValueError("offset cannot be negative")

    revision, all_shards = fetch_parquet_manifest(dataset)
    shards = sorted(
        (shard for shard in all_shards if shard.split == split),
        key=lambda shard: shard.filename,
    )
    if not shards:
        raise RuntimeError(f"no Parquet shards found for split {split!r}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    extension_directory = extension_directory or (
        Path(tempfile.gettempdir()) / "cc-for-duckdb-extensions"
    )
    connection = _open_duckdb(extension_directory, threads)
    selected = 0
    visited = 0
    source_bytes = 0
    used_shards: list[str] = []
    seen_members: set[str] = set()
    content_digest = hashlib.sha256()

    try:
        with temporary.open("wb") as raw_stream:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_stream,
                compresslevel=9,
                mtime=0,
            ) as gzip_stream:
                with tarfile.open(
                    fileobj=gzip_stream,
                    mode="w",
                    format=tarfile.USTAR_FORMAT,
                ) as archive:
                    for shard in shards:
                        used = False
                        for uuid, encoded_source in _read_projected_rows(
                            connection, [shard], fetch_size
                        ):
                            if visited < offset:
                                visited += 1
                                continue
                            if selected >= count:
                                break
                            visited += 1
                            source = _decode_code(encoded_source).encode("utf-8")
                            member_name = _safe_member_name(str(uuid))
                            if member_name in seen_members:
                                raise RuntimeError(
                                    f"duplicate dataset UUID: {uuid}"
                                )
                            seen_members.add(member_name)
                            _add_source(
                                archive,
                                member_name=member_name,
                                source=source,
                            )
                            content_digest.update(member_name.encode("utf-8"))
                            content_digest.update(b"\0")
                            content_digest.update(source)
                            selected += 1
                            source_bytes += len(source)
                            used = True
                        if used:
                            used_shards.append(shard.filename)
                        if selected >= count:
                            break
        if selected != count:
            raise RuntimeError(
                f"requested {count} programs after offset {offset}, found {selected}"
            )
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        connection.close()

    payload = {
        "format_version": FORMAT_VERSION,
        "dataset": dataset,
        "dataset_url": f"https://huggingface.co/datasets/{dataset}",
        "dataset_revision": revision,
        "upstream_license": "Apache-2.0",
        "split": split,
        "selection": {
            "offset": offset,
            "count": count,
            "order": "Parquet filename ascending, then stored row order",
        },
        "archive": output_path.name,
        "archive_bytes": output_path.stat().st_size,
        "archive_sha256": _sha256_file(output_path),
        "source_bytes": source_bytes,
        "source_content_sha256": content_digest.hexdigest(),
        "members": selected,
        "parquet_shards": used_shards,
    }
    _write_json(manifest_path, payload)
    return payload


def iter_snapshot_sources(archive_path: Path) -> Iterator[tuple[str, str]]:
    """Yield UUID/source pairs while rejecting unsafe or duplicate members."""

    seen: set[str] = set()
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or len(path.parts) != 2
                or path.parts[0] != "raw"
                or path.suffix != ".py"
                or ".." in path.parts
            ):
                raise ValueError(f"unexpected archive member: {member.name!r}")
            uuid = path.stem
            if uuid in seen:
                raise ValueError(f"duplicate archive UUID: {uuid}")
            seen.add(uuid)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read archive member: {member.name!r}")
            yield uuid, stream.read().decode("utf-8")


def validate_snapshot(
    *,
    archive_path: Path,
    report_path: Path,
    manifest_path: Path | None = None,
    max_failure_records: int = 20,
) -> dict[str, Any]:
    """Run the CC-for structural contract over every archived program."""

    expected: dict[str, Any] | None = None
    if manifest_path is not None:
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_sha256 = _sha256_file(archive_path)
        if actual_sha256 != expected["archive_sha256"]:
            raise ValueError(
                "archive SHA-256 does not match manifest: "
                f"{actual_sha256} != {expected['archive_sha256']}"
            )

    passed = 0
    failed = 0
    workplane_steps = 0
    preserved_loops = 0
    warnings: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    for uuid, source in iter_snapshot_sources(archive_path):
        converted = canonicalize_code(source)
        workplane_steps += converted.report.workplane_steps
        preserved_loops += converted.report.preserved_loops
        warnings.update(converted.report.warnings)
        if converted.report.structural_errors:
            failed += 1
            if len(failures) < max_failure_records:
                failures.append(
                    {
                        "uuid": uuid,
                        "errors": converted.report.structural_errors,
                    }
                )
        else:
            passed += 1

    programs = passed + failed
    if expected is not None and programs != expected["members"]:
        raise ValueError(
            f"archive has {programs} programs; manifest expects {expected['members']}"
        )
    payload = {
        "archive": str(archive_path),
        "archive_sha256": _sha256_file(archive_path),
        "programs": programs,
        "structural_passed": passed,
        "structural_failed": failed,
        "structural_pass_rate": passed / programs if programs else 0.0,
        "workplane_steps": workplane_steps,
        "preserved_loops": preserved_loops,
        "warnings": dict(sorted(warnings.items())),
        "failure_examples": failures,
    }
    _write_json(report_path, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="create a raw-source snapshot")
    export.add_argument("--output", type=Path, default=DEFAULT_ARCHIVE)
    export.add_argument("--manifest", type=Path)
    export.add_argument("--dataset", default=DEFAULT_DATASET)
    export.add_argument("--split", default=DEFAULT_SPLIT)
    export.add_argument("--count", type=int, default=DEFAULT_COUNT)
    export.add_argument("--offset", type=int, default=0)
    export.add_argument("--fetch-size", type=int, default=512)
    export.add_argument("--threads", type=int, default=8)
    export.add_argument("--extension-directory", type=Path)

    validate = subparsers.add_parser(
        "validate", help="validate every archived source with CC-for"
    )
    validate.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    validate.add_argument("--manifest", type=Path)
    validate.add_argument("--report", type=Path, required=True)
    validate.add_argument("--max-failure-records", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "export":
        manifest = args.manifest or args.output.with_name("manifest.json")
        payload = export_snapshot(
            output_path=args.output,
            manifest_path=manifest,
            dataset=args.dataset,
            split=args.split,
            count=args.count,
            offset=args.offset,
            fetch_size=args.fetch_size,
            threads=args.threads,
            extension_directory=args.extension_directory,
        )
    else:
        payload = validate_snapshot(
            archive_path=args.archive,
            report_path=args.report,
            manifest_path=args.manifest,
            max_failure_records=args.max_failure_records,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
