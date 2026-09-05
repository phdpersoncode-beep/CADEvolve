"""Run CAD jobs in disposable processes with individual wall-clock deadlines.

A segfault must affect only its own input. A Python alarm cannot interrupt an
OpenCascade call stuck in native code, so the parent enforces the deadline.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
from dataclasses import dataclass
from multiprocessing.connection import wait
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class IsolatedResult:
    index: int
    value: Any = None
    error: str | None = None


def _child(connection, function, args):
    if os.name == "posix":
        os.setsid()
    try:
        connection.send((True, function(*args)))
    except Exception as error:
        connection.send((False, f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def _stop(process):
    # Tracing jobs can spawn a second interpreter. Kill the job's process group
    # as well as its leader so an outer deadline cannot leave an orphan CAD run.
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # The child may not have reached setsid() before its deadline.
    if process.is_alive():
        process.kill()
    process.join()


def iter_isolated(
    function: Callable,
    arguments: Iterable[tuple],
    *,
    workers: int = 1,
    timeout: float = 120,
):
    """Yield one result per input, even after a native crash or infinite loop.

    Functions and arguments must be pickleable. No CAD object crosses processes.
    Results arrive as jobs finish, enabling durable per-input checkpoints.
    """
    if workers < 1 or timeout <= 0:
        raise ValueError("workers and timeout must be positive")
    context = mp.get_context("spawn")
    pending = iter(enumerate(arguments))
    active = {}
    exhausted = False
    try:
        while active or not exhausted:
            while len(active) < workers and not exhausted:
                try:
                    index, args = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(target=_child, args=(sender, function, args))
                process.start()
                sender.close()
                active[receiver] = (index, process, time.monotonic() + timeout)

            if not active:
                break
            deadline = min(item[2] for item in active.values())
            ready = set(wait(active, timeout=max(0, min(0.1, deadline - time.monotonic()))))
            for receiver, (index, process, deadline) in list(active.items()):
                result = None
                if receiver in ready:
                    try:
                        ok, value = receiver.recv()
                        result = IsolatedResult(index, value=value) if ok else IsolatedResult(index, error=value)
                    except EOFError:
                        result = IsolatedResult(index, error=f"worker exited without a result (exitcode={process.exitcode})")
                elif time.monotonic() >= deadline:
                    result = IsolatedResult(index, error=f"worker exceeded {timeout:g}s wall-clock timeout")
                elif not process.is_alive():
                    result = IsolatedResult(index, error=f"worker exited without a result (exitcode={process.exitcode})")
                if result is not None:
                    _stop(process)
                    receiver.close()
                    del active[receiver]
                    yield result
    finally:
        for receiver, (_, process, _) in active.items():
            _stop(process)
            receiver.close()
