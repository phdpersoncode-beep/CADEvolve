"""AST-level gates: do named parameters, their reuse, and loops actually survive?

The geometry gates prove the canonical program *builds the same solid*.  They say
nothing about whether it still does so *symbolically* -- a canonicalizer that
constant-folded every parameter into a literal would pass every geometry check.
These gates measure the representation itself, which is the property the CC-for
change is actually about.
"""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

_SSA_SUFFIX = re.compile(r"^(?P<stem>.+)_(?P<version>\d+)$")
_WORKPLANE_NAME = re.compile(r"^wp\d+$")

# Names the lowering step introduces on purpose; they are not source parameters.
_GENERATED_PREFIXES = ("wp", "_cc_for_", "helper")


def _is_numeric_constant(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    )


def numeric_literals(code: str) -> list[float]:
    """Every numeric literal in a program, as a sorted multiset."""

    return sorted(
        float(node.value)
        for node in ast.walk(ast.parse(code))
        if _is_numeric_constant(node)
    )


def string_literals(code: str) -> list[str]:
    """String literals, which carry CadQuery selectors such as ``'>Z'``."""

    return sorted(
        node.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )


def bound_names(code: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(code)):
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            targets = [node.target]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        for target in targets:
            for child in ast.walk(target):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    names.add(child.id)
    return names


def load_counts(code: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            counts[node.id] = counts.get(node.id, 0) + 1
    return counts


def attribute_parameters(code: str) -> set[str]:
    """Attribute names read off parameter containers, e.g. ``m.wall_thickness``."""

    return {
        node.attr
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
    }


def _is_pure_data(node: ast.AST) -> bool:
    """True for expressions built only from literals, names, math and containers."""

    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Attribute):
                root = func.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                if not (isinstance(root, ast.Name) and root.id in {"math", "np", "numpy"}):
                    return False
            elif isinstance(func, ast.Name):
                if func.id not in {
                    "float", "int", "abs", "min", "max", "round", "len",
                    "range", "list", "tuple", "sum", "sorted", "pow",
                }:
                    return False
            else:
                return False
    return True


def source_parameters(code: str, result_name: str = "result") -> set[str]:
    """Top-level named values a human would call parameters of the source program.

    A parameter is a module-level ``name = <pure data expression>`` binding, plus
    every keyword of a ``SimpleNamespace``-style measures container.  The terminal
    ``result`` alias is excluded: it names the model, not a parameter.
    """

    tree = ast.parse(code)
    parameters: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target, value = stmt.targets[0], stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            target, value = stmt.target, stmt.value
        else:
            continue
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            for keyword in value.keywords:
                if keyword.arg:
                    parameters.add(keyword.arg)
        if not isinstance(target, ast.Name) or not _is_pure_data(value):
            continue
        if isinstance(value, ast.Constant) and value.value is None:
            continue
        if target.id != result_name:
            parameters.add(target.id)

    # Nested measures containers: Measures(panel=Measures(width=...)).
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"SimpleNamespace", "Measures", "namedtuple"}
        ):
            for keyword in node.keywords:
                if keyword.arg:
                    parameters.add(keyword.arg)
    parameters.discard(result_name)
    return parameters


def resolve_parameter(
    parameter: str,
    canonical_names: set[str],
    *,
    flattened: Mapping[str, Mapping[str, str]] | None = None,
    versioned: Mapping[str, Iterable[str]] | None = None,
) -> str | None:
    """Map a source parameter name onto its canonical name, or ``None`` if lost.

    Canonicalization is allowed to rename in exactly two ways: SSA versioning
    (``x`` -> ``x_1``) and namespace flattening (``m.width`` -> ``width`` or
    ``panel_width``).  Anything else counts as a lost parameter.
    """

    if parameter in canonical_names:
        return parameter
    for source_name, versions in (versioned or {}).items():
        if source_name == parameter:
            for version in versions:
                if version in canonical_names:
                    return version
    for mapping in (flattened or {}).values():
        renamed = mapping.get(parameter)
        if renamed and renamed in canonical_names:
            return renamed
    versioned_matches = sorted(
        name
        for name in canonical_names
        if (match := _SSA_SUFFIX.match(name)) and match.group("stem") == parameter
    )
    if versioned_matches:
        return versioned_matches[0]
    # Flattened nested containers use a ``<root>_<key>`` naming scheme.
    suffix_matches = sorted(
        name for name in canonical_names if name.endswith(f"_{parameter}")
    )
    if suffix_matches:
        return suffix_matches[0]
    return None


@dataclass(frozen=True)
class PreambleLayout:
    """Where the parameter block sits relative to the first modelling statement."""

    import_count: int
    preamble_size: int
    first_geometry_index: int
    contiguous: bool
    late_parameters: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preamble_layout(code: str, result_name: str = "result") -> PreambleLayout:
    """Check that named parameters form one contiguous block after the imports."""

    tree = ast.parse(code)
    body = tree.body

    def is_geometry_statement(stmt: ast.stmt) -> bool:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name) and (
                _WORKPLANE_NAME.match(target.id) or target.id == result_name
            ):
                return True
            return not _is_pure_data(stmt.value)
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            return False
        return True  # loops, defs, classes, expressions: modelling territory

    imports = 0
    for stmt in body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            imports += 1
        else:
            break

    first_geometry = len(body)
    for index in range(imports, len(body)):
        if is_geometry_statement(body[index]):
            first_geometry = index
            break

    late: list[str] = []
    for stmt in body[first_geometry:]:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if (
                isinstance(target, ast.Name)
                and not _WORKPLANE_NAME.match(target.id)
                and target.id != result_name
                and not target.id.startswith(_GENERATED_PREFIXES)
                and _is_pure_data(stmt.value)
            ):
                late.append(target.id)

    return PreambleLayout(
        import_count=imports,
        preamble_size=first_geometry - imports,
        first_geometry_index=first_geometry,
        contiguous=not late,
        late_parameters=tuple(late),
    )


def loop_count(code: str) -> int:
    return sum(
        1 for node in ast.walk(ast.parse(code)) if isinstance(node, (ast.For, ast.AsyncFor))
    )


def comprehension_count(code: str) -> int:
    return sum(
        1
        for node in ast.walk(ast.parse(code))
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
    )


def workplane_steps(code: str) -> int:
    return sum(1 for name in bound_names(code) if _WORKPLANE_NAME.match(name))


def fluent_chain_depth(code: str) -> int:
    """Longest run of chained attribute calls, e.g. ``a.b().c().d()`` -> 3."""

    depth = 0

    def measure(node: ast.AST) -> int:
        count = 0
        while (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        ):
            count += 1
            node = node.func.value
        return count

    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Call):
            depth = max(depth, measure(node))
    return depth


@dataclass
class CodeComparison:
    """All AST-level gate results for one (source, canonical) pair."""

    source_parameters: int
    retained_parameters: int
    lost_parameters: list[str] = field(default_factory=list)
    unused_parameters: list[str] = field(default_factory=list)
    inlined_parameters: list[str] = field(default_factory=list)
    source_loops: int = 0
    canonical_loops: int = 0
    source_comprehensions: int = 0
    canonical_comprehensions: int = 0
    workplane_steps: int = 0
    source_chain_depth: int = 0
    canonical_chain_depth: int = 0
    numeric_literals_added: list[float] = field(default_factory=list)
    numeric_literals_removed: list[float] = field(default_factory=list)
    string_literals_added: list[str] = field(default_factory=list)
    string_literals_removed: list[str] = field(default_factory=list)
    preamble: dict[str, Any] = field(default_factory=dict)

    @property
    def parameter_retention(self) -> float:
        if self.source_parameters == 0:
            return 1.0
        return self.retained_parameters / self.source_parameters

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["parameter_retention"] = self.parameter_retention
        return data


def _multiset_difference(left: list[Any], right: list[Any]) -> list[Any]:
    from collections import Counter

    difference = Counter(left) - Counter(right)
    return sorted(difference.elements())


def compare_code(
    source: str,
    canonical: str,
    *,
    report: Mapping[str, Any] | None = None,
    result_name: str = "result",
) -> CodeComparison:
    report = report or {}
    parameters = source_parameters(source, result_name)
    canonical_names = bound_names(canonical)
    canonical_loads = load_counts(canonical)
    canonical_attributes = attribute_parameters(canonical)

    lost: list[str] = []
    unused: list[str] = []
    resolved: dict[str, str] = {}
    for parameter in sorted(parameters):
        canonical_name = resolve_parameter(
            parameter,
            canonical_names,
            flattened=report.get("flattened_namespaces"),
            versioned=report.get("versioned_names"),
        )
        if canonical_name is None:
            # A container attribute may survive as an attribute rather than a name.
            if parameter in canonical_attributes:
                resolved[parameter] = parameter
                continue
            lost.append(parameter)
            continue
        resolved[parameter] = canonical_name
        if canonical_loads.get(canonical_name, 0) == 0 and parameter not in canonical_attributes:
            unused.append(parameter)

    # A parameter that the source passed to a CadQuery call but the canonical
    # program passes as a bare literal has been constant-folded away.
    source_loads = load_counts(source)
    inlined = [
        parameter
        for parameter, canonical_name in resolved.items()
        if source_loads.get(parameter, 0) > 0
        and canonical_loads.get(canonical_name, 0) == 0
        and parameter not in canonical_attributes
    ]

    source_numbers = numeric_literals(source)
    canonical_numbers = numeric_literals(canonical)
    source_strings = string_literals(source)
    canonical_strings = string_literals(canonical)

    return CodeComparison(
        source_parameters=len(parameters),
        retained_parameters=len(resolved),
        lost_parameters=lost,
        unused_parameters=unused,
        inlined_parameters=sorted(inlined),
        source_loops=loop_count(source),
        canonical_loops=loop_count(canonical),
        source_comprehensions=comprehension_count(source),
        canonical_comprehensions=comprehension_count(canonical),
        workplane_steps=workplane_steps(canonical),
        source_chain_depth=fluent_chain_depth(source),
        canonical_chain_depth=fluent_chain_depth(canonical),
        numeric_literals_added=_multiset_difference(canonical_numbers, source_numbers),
        numeric_literals_removed=_multiset_difference(source_numbers, canonical_numbers),
        string_literals_added=_multiset_difference(canonical_strings, source_strings),
        string_literals_removed=_multiset_difference(source_strings, canonical_strings),
        preamble=preamble_layout(canonical, result_name).to_dict(),
    )
