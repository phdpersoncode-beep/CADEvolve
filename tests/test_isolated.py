"""A failed native CAD worker must not invalidate healthy neighbouring jobs."""

import os
import subprocess
import sys
import time

import pytest

from utils.isolated import iter_isolated


def _job(kind):
    if kind == "crash":
        os._exit(17)
    if kind == "hang":
        time.sleep(30)
    return kind


def test_crash_and_timeout_are_isolated():
    arguments = [(kind,) for kind in ("first", "crash", "hang", "last")]
    results = {r.index: r for r in iter_isolated(_job, arguments, workers=2, timeout=2)}
    assert len(results) == 4
    assert results[0].value == "first"
    assert results[3].value == "last"
    assert "exitcode=17" in results[1].error
    assert "timeout" in results[2].error


def _spawn_descendant(path):
    subprocess.Popen([sys.executable, "-c",
                      "import pathlib,sys,time; time.sleep(2); pathlib.Path(sys.argv[1]).touch()",
                      str(path)])
    time.sleep(30)


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup is POSIX-specific")
def test_timeout_also_stops_descendant_processes(tmp_path):
    marker = tmp_path / "orphan-finished"
    results = list(iter_isolated(_spawn_descendant, [(marker,)], timeout=1))
    assert "timeout" in results[0].error
    time.sleep(2.5)
    assert not marker.exists()
