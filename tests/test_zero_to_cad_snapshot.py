from __future__ import annotations

import gzip
import io
import json
import tarfile
import tempfile
import unittest
from itertools import islice
from pathlib import Path

from dataset_utils.canonicalization_run.zero_to_cad_snapshot import (
    _sha256_file,
    iter_snapshot_sources,
)
from dataset_utils.utils.canonicalization.cc_for import canonicalize_code


ROOT = Path(__file__).parents[1]
SNAPSHOT = ROOT / "demo_data" / "zero_to_cad_5k"


class ZeroToCADSnapshotTests(unittest.TestCase):
    def test_snapshot_matches_manifest_and_contains_5000_sources(self) -> None:
        manifest = json.loads(
            (SNAPSHOT / "manifest.json").read_text(encoding="utf-8")
        )
        archive = SNAPSHOT / manifest["archive"]

        self.assertEqual(_sha256_file(archive), manifest["archive_sha256"])
        sources = list(iter_snapshot_sources(archive))
        self.assertEqual(len(sources), 5_000)
        self.assertEqual(len({uuid for uuid, _ in sources}), 5_000)

        for uuid, source in islice(sources, 25):
            with self.subTest(uuid=uuid):
                self.assertFalse(canonicalize_code(source).report.structural_errors)

    def test_reader_rejects_unsafe_archive_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "unsafe.tar.gz"
            with archive_path.open("wb") as raw_stream:
                with gzip.GzipFile(fileobj=raw_stream, mode="wb", mtime=0) as gz:
                    with tarfile.open(fileobj=gz, mode="w") as archive:
                        payload = b"result = None\n"
                        member = tarfile.TarInfo("../escape.py")
                        member.size = len(payload)
                        archive.addfile(member, io.BytesIO(payload))

            with self.assertRaisesRegex(ValueError, "unexpected archive member"):
                list(iter_snapshot_sources(archive_path))


if __name__ == "__main__":
    unittest.main()
