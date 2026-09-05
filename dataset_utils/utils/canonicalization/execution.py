"""Execute a CAD script with normal module lookup semantics."""

from contextlib import contextmanager
import sys
import types


@contextmanager
def program_namespace():
    """Supply a registered module for dataclasses and postponed annotations.

    This helper is used within a single CAD worker, never by concurrent threads.
    """
    name = "__cad_program__"
    previous = sys.modules.get(name)
    module = types.ModuleType(name)
    sys.modules[name] = module
    try:
        yield module.__dict__
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def execute_program(code, filename="<cad-program>"):
    with program_namespace() as namespace:
        exec(compile(code, filename, "exec"), namespace, namespace)
        return namespace
