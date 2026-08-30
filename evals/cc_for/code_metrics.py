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


_PURE_BUILTINS = frozenset(
    {
        "float", "int", "abs", "min", "max", "round", "len",
        "range", "list", "tuple", "sum", "sorted", "pow", "dict", "set",
    }
)


def math_imports(tree: ast.AST) -> set[str]:
    """Names bound by ``from math import cos, radians, ...``.

    Such a call is parameter algebra; without this the importing program looks
    like it starts modelling at its first derived angle.
    """

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {"math", "numpy"}:
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def data_container_classes(tree: ast.AST) -> set[str]:
    """Locally defined classes that are parameter containers in all but name.

    Zero-to-CAD programs often hand-roll a ``Measures`` class whose ``__init__``
    does nothing but store and derive numbers.  Semantically it is a
    ``SimpleNamespace``, so its constructor keywords are design parameters.
    """

    containers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        initializers = [
            child
            for child in node.body
            if isinstance(child, ast.FunctionDef) and child.name == "__init__"
        ]
        if len(initializers) != 1:
            continue
        body = [
            stmt
            for stmt in initializers[0].body
            if not (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            )
        ]
        if not body:
            continue
        if all(
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Attribute)
            and isinstance(stmt.targets[0].value, ast.Name)
            and stmt.targets[0].value.id == "self"
            and _is_pure_data(stmt.value, math_imports(tree))
            for stmt in body
        ):
            containers.add(node.name)
    return containers


def namespace_aliases(tree: ast.AST) -> set[str]:
    """Local names that refer to ``types.SimpleNamespace``.

    Zero-to-CAD programs almost always import it under an alias
    (``from types import SimpleNamespace as Measures``) and use it as a parameter
    container, so a call to one is parameter data rather than geometry.
    """

    aliases = {"SimpleNamespace", "namedtuple"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {"types", "collections"}:
            for name in node.names:
                if name.name in {"SimpleNamespace", "namedtuple"}:
                    aliases.add(name.asname or name.name)
    aliases |= data_container_classes(tree)
    # A container assigned from another container call is one too.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in aliases
        ):
            aliases.add(node.targets[0].id)
    return aliases


def _is_pure_data(node: ast.AST, containers: frozenset[str] | set[str] = frozenset()) -> bool:
    """True for expressions built only from literals, names, math and containers."""

    allowed = _PURE_BUILTINS | set(containers)
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
                if func.id not in allowed:
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
    containers = namespace_aliases(tree) | math_imports(tree)
    parameters: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target, value = stmt.targets[0], stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            target, value = stmt.target, stmt.value
        else:
            continue
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in namespace_aliases(tree)
        ):
            for keyword in value.keywords:
                # Only a literal keyword introduces a new parameter.  When the
                # value is an expression its own names are already counted, and
                # the keyword is just the container's label for them.
                if keyword.arg and isinstance(keyword.value, ast.Constant):
                    parameters.add(keyword.arg)
        if not isinstance(target, ast.Name) or not _is_pure_data(value, containers):
            continue
        if isinstance(value, ast.Constant) and value.value is None:
            continue
        # A bare alias (``solid = base``, ``m = self.measures``) renames an
        # existing binding; it is not a new parameter.
        if isinstance(value, (ast.Name, ast.Attribute)):
            continue
        if target.id != result_name:
            parameters.add(target.id)

    # Nested measures containers: Measures(panel=Measures(width=...)).
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in namespace_aliases(tree)
        ):
            for keyword in node.keywords:
                if keyword.arg and isinstance(keyword.value, ast.Constant):
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


def control_flow_bound_names(code: str) -> set[str]:
    """Names assigned anywhere inside a loop, branch, or ``try`` block.

    Such a name cannot be hoisted into the preamble without changing semantics, so
    it is not evidence of a badly placed parameter block.
    """

    names: set[str] = set()
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.If, ast.Try)):
            continue
        for child in ast.walk(node):
            targets: list[ast.AST] = []
            if isinstance(child, ast.Assign):
                targets = list(child.targets)
            elif isinstance(child, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor)):
                targets = [child.target]
            elif (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr in {"append", "extend", "insert", "update", "add"}
                and isinstance(child.func.value, ast.Name)
            ):
                # ``points.append(...)`` rebinds nothing but mutates in place, so
                # the container has to stay where the loop can reach it in order.
                names.add(child.func.value.id)
            for target in targets:
                for leaf in ast.walk(target):
                    if isinstance(leaf, ast.Name) and isinstance(leaf.ctx, ast.Store):
                        names.add(leaf.id)
    return names


def _is_header_statement(stmt: ast.stmt) -> bool:
    """Imports and the module docstring precede any parameter block."""

    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return True
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _is_geometry_statement(
    stmt: ast.stmt,
    containers: frozenset[str] | set[str],
    result_name: str,
) -> bool:
    if _is_header_statement(stmt):
        return False
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        if isinstance(target, ast.Name) and (
            _WORKPLANE_NAME.match(target.id) or target.id == result_name
        ):
            return True
        # ``sweep_path = wp3`` binds a Workplane step to the source's own name so
        # a helper can still resolve it; the value is geometry however the target
        # is spelled.
        if isinstance(stmt.value, ast.Name) and _WORKPLANE_NAME.match(stmt.value.id):
            return True
        return not _is_pure_data(stmt.value, containers)
    return True  # loops, defs, classes, expressions: modelling territory


def _deep_loads(node: ast.AST) -> set[str]:
    """Every ``Load`` name below ``node``, function and lambda bodies included."""

    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def preamble_layout(
    code: str,
    result_name: str = "result",
    exempt: frozenset[str] | set[str] = frozenset(),
) -> PreambleLayout:
    """Check that named parameters form one contiguous block after the imports.

    ``exempt`` holds names the contract deliberately leaves in place -- loop-carried
    state and anything rebound inside control flow.
    """

    tree = ast.parse(code)
    body = tree.body
    containers = namespace_aliases(tree) | math_imports(tree)
    exempt = set(exempt) | control_flow_bound_names(code) | container_roots(code)

    def is_header(stmt: ast.stmt) -> bool:
        return _is_header_statement(stmt)

    def is_geometry_statement(stmt: ast.stmt) -> bool:
        return _is_geometry_statement(stmt, containers, result_name)

    imports = 0
    for stmt in body:
        if is_header(stmt):
            imports += 1
        else:
            break

    first_geometry = len(body)
    for index in range(imports, len(body)):
        if is_geometry_statement(body[index]):
            first_geometry = index
            break

    # A pure-data assignment whose value reads a lowered geometry step is a
    # geometry alias, not a parameter.
    def aliases_geometry(node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Name) and _WORKPLANE_NAME.match(child.id)
            for child in ast.walk(node)
        )

    # Map each top-level name to the names its definition reads, so a candidate
    # can be tested for a transitive dependency on loop-built state.
    definitions: dict[str, set[str]] = {}
    for stmt in body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            if isinstance(stmt.targets[0], ast.Name):
                definitions[stmt.targets[0].id] = {
                    child.id
                    for child in ast.walk(stmt.value)
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
                }

    def pinned_by_dependency(node: ast.AST) -> bool:
        """True when the value cannot move ahead of the modelling it follows."""

        pending = {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        }
        seen: set[str] = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            if name in exempt or _WORKPLANE_NAME.match(name):
                return True
            pending |= definitions.get(name, set()) - seen
        return False

    late: list[str] = []
    for stmt in body[first_geometry:]:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if (
                isinstance(target, ast.Name)
                and not _WORKPLANE_NAME.match(target.id)
                and target.id != result_name
                and target.id not in exempt
                and not target.id.startswith(_GENERATED_PREFIXES)
                and not aliases_geometry(stmt.value)
                and not pinned_by_dependency(stmt.value)
                and _is_pure_data(stmt.value, containers)
            ):
                late.append(target.id)

    return PreambleLayout(
        import_count=imports,
        preamble_size=first_geometry - imports,
        first_geometry_index=first_geometry,
        contiguous=not late,
        late_parameters=tuple(late),
    )


@dataclass(frozen=True)
class ParameterGroup:
    """One run of parameter assignments and the modelling step it introduces."""

    parameters: tuple[str, ...]
    step_size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LateParameterLayout:
    """Whether every parameter sits directly above the step that consumes it."""

    parameter_count: int
    group_count: int
    first_group_size: int
    largest_group: int
    grouped: bool
    early_parameters: tuple[str, ...] = ()
    unread_parameters: tuple[str, ...] = ()
    groups: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parameter_groups(
    code: str, result_name: str = "result"
) -> list[tuple[list[ast.stmt], list[ast.stmt]]]:
    """Split top-level code into ``(parameters, modelling statements)`` groups.

    A group ends where the next parameter assignment begins, so the modelling
    statements of a group are exactly the ones its parameters were placed for.
    """

    tree = ast.parse(code)
    containers = namespace_aliases(tree) | math_imports(tree)
    groups: list[tuple[list[ast.stmt], list[ast.stmt]]] = []
    pending: list[ast.stmt] = []
    seen_modelling = False

    for stmt in tree.body:
        if _is_header_statement(stmt) and not seen_modelling and not pending:
            continue
        if _is_geometry_statement(stmt, containers, result_name):
            seen_modelling = True
            if pending or not groups:
                groups.append((pending, [stmt]))
                pending = []
            else:
                groups[-1][1].append(stmt)
        else:
            pending.append(stmt)

    if pending:
        if groups:
            groups[-1][0].extend(pending)
        else:
            groups.append((pending, []))
    return groups


def late_parameter_layout(
    code: str,
    result_name: str = "result",
    exempt: frozenset[str] | set[str] = frozenset(),
) -> LateParameterLayout:
    """Check that no parameter could have been pushed into a later group.

    A parameter is justified where it stands when the modelling statements of its
    own group read it, or when another justified parameter of the same group does.
    Anything else was defined earlier than it had to be.  A parameter that is
    pinned by control flow, or that nothing reads at all, has no later anchor to
    move to and is reported rather than counted against the layout.

    This is a necessary condition, not a sufficient one.  Lowering erases the
    source statement boundaries -- every step reads the step before it, so a
    fluent chain and two consecutive features look alike -- which means a
    program whose parameters were never split at all presents as one group in
    which everything is legitimately justified.  ``group_count`` is reported so
    that case is visible, and the corpus-level check that placement actually
    splits the preamble lives in ``tests/test_cc_step.py``.
    """

    tree = ast.parse(code)
    containers = namespace_aliases(tree) | math_imports(tree)
    pinned = set(exempt) | control_flow_bound_names(code) | container_roots(code)

    groups = parameter_groups(code, result_name)
    all_reads: dict[str, int] = {}
    for stmt in tree.body:
        for name in _deep_loads(stmt):
            all_reads[name] = all_reads.get(name, 0) + 1

    early: list[str] = []
    unread: list[str] = []
    records: list[ParameterGroup] = []
    sizes: list[int] = []

    for parameters, statements in groups:
        names = [
            name
            for stmt in parameters
            if (name := _assignment_target(stmt)) is not None
        ]
        sizes.append(len(names))
        records.append(ParameterGroup(tuple(names), len(statements)))

        # A parameter settles in this group when the group's own statements read
        # it, when control flow pins it here, or when nothing reads it at all and
        # so no later group could claim it.  A statement that binds no simple
        # name -- an attribute write such as ``measures.width = width`` -- has
        # nothing to reposition and is settled by construction.  Whatever a
        # settled statement reads is needed here too, which is how a chain of
        # derived values, a field write, and the retained namespace container all
        # keep their inputs in their own group.
        def _settled(stmt: ast.stmt) -> bool:
            name = _assignment_target(stmt)
            if name is None:
                return True
            return name in justified or name in pinned or not all_reads.get(name)

        justified: set[str] = set()
        for stmt in statements:
            justified |= _deep_loads(stmt)
        changed = True
        while changed:
            changed = False
            for stmt in parameters:
                if not _settled(stmt):
                    continue
                for read in _deep_loads(stmt):
                    if read not in justified:
                        justified.add(read)
                        changed = True

        for name in names:
            if name in justified or name in pinned:
                continue
            if not all_reads.get(name):
                unread.append(name)
                continue
            early.append(name)

    return LateParameterLayout(
        parameter_count=sum(sizes),
        group_count=sum(1 for size in sizes if size),
        first_group_size=sizes[0] if sizes else 0,
        largest_group=max(sizes, default=0),
        grouped=not early,
        early_parameters=tuple(early),
        unread_parameters=tuple(unread),
        groups=tuple(record.to_dict() for record in records),
    )


def _assignment_target(stmt: ast.stmt) -> str | None:
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return stmt.targets[0].id
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id
    return None


def container_roots(code: str) -> set[str]:
    """Names bound to a parameter container, plus aliases of one.

    Flattening lifts a container's fields into named parameters and keeps the
    container itself only as a runtime compatibility shim, so it is expected to
    end up unread.  That is not an inlined parameter.
    """

    tree = ast.parse(code)
    containers = namespace_aliases(tree)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            if value.func.id in containers:
                roots.add(node.targets[0].id)
        elif isinstance(value, ast.Name) and value.id in roots:
            roots.add(node.targets[0].id)
        elif isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
            if value.value.id == "self" or value.value.id in roots:
                roots.add(node.targets[0].id)
    return roots


def preamble_names(code: str, result_name: str = "result") -> set[str]:
    """Names assigned in the canonical parameter preamble (before modelling)."""

    layout = preamble_layout(code, result_name)
    tree = ast.parse(code)
    names: set[str] = set()
    for stmt in tree.body[: layout.first_geometry_index]:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                for leaf in ast.walk(target):
                    if isinstance(leaf, ast.Name):
                        names.add(leaf.id)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names.add(stmt.target.id)
    return names


def parameter_block_names(
    code: str,
    result_name: str = "result",
    parameter_placement: str = "preamble",
) -> set[str]:
    """Names bound in the canonical parameter blocks, wherever they were placed.

    Under CC-for that is the single preamble; under CC-step it is the union of
    the per-step groups.  Design-parameter coverage asks whether a dimension is
    still bound by name, which is the same question either way.
    """

    if parameter_placement != "late":
        return preamble_names(code, result_name)

    names: set[str] = set()
    for parameters, _statements in parameter_groups(code, result_name):
        for stmt in parameters:
            name = _assignment_target(stmt)
            if name is not None:
                names.add(name)
    return names


def design_parameters(code: str, result_name: str = "result") -> set[str]:
    """Source parameters whose value is a bare number.

    These are the dimensions a downstream model is expected to be able to edit,
    so they are the ones that have to end up as named entries in the preamble.
    """

    tree = ast.parse(code)
    containers = namespace_aliases(tree)
    numeric: set[str] = set()

    def is_number(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        )

    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target, value = stmt.targets[0], stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            target, value = stmt.target, stmt.value
        else:
            continue
        if isinstance(target, ast.Name) and is_number(value) and target.id != result_name:
            numeric.add(target.id)

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in containers
        ):
            for keyword in node.keywords:
                if keyword.arg and is_number(keyword.value):
                    numeric.add(keyword.arg)
    return numeric


def _ssa_stem(name: str) -> str:
    """``thickness_2`` -> ``thickness``; anything else unchanged."""

    match = _SSA_SUFFIX.match(name)
    return match.group("stem") if match else name


def loop_body_bindings(code: str) -> set[str]:
    """Names bound inside a loop -- its target or anywhere in its body.

    Reported per name rather than per loop because lowering redistributes a
    body's statements: what has to survive is that the loop still writes the
    name, not that it writes it in the same statement.
    """

    names: set[str] = set()
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            continue
        subtrees = list(node.body)
        target = getattr(node, "target", None)
        if target is not None:
            subtrees.append(target)
        for subtree in subtrees:
            for child in ast.walk(subtree):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    names.add(child.id)
    return names


def loop_body_reads(code: str) -> set[str]:
    """Names read inside a loop body."""

    names: set[str] = set()
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            continue
        for stmt in node.body:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    names.add(child.id)
    return names


def dropped_loop_bindings(
    source: str, canonical: str, result_name: str = "result"
) -> list[str]:
    """Names a loop wrote in the source and no longer writes in the canonical code.

    A loop-carried accumulator that stops being assigned is the shape of silent
    geometry loss: the canonical program runs, keeps the structural contract, and
    builds whatever the initializer held.  Only names the canonical program still
    *reads* inside a loop count, so a value the lowerer legitimately replaced with
    its own ``wpN`` -- or renamed, as ``result`` becomes ``result_state`` -- is not
    reported.  SSA suffixes are stripped before comparing.
    """

    written = {_ssa_stem(name) for name in loop_body_bindings(source)}
    rewritten = {_ssa_stem(name) for name in loop_body_bindings(canonical)}
    still_read = {_ssa_stem(name) for name in loop_body_reads(canonical)}
    return sorted(
        name
        for name in written - rewritten
        if name in still_read
        and name != result_name
        and not name.startswith(_GENERATED_PREFIXES)
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


# Chains rooted in these stay symbolic by design: they are parameter algebra,
# not modelling steps (the contract keeps cq.Plane/cq.Vector expressions intact).
_SYMBOLIC_ROOTS = frozenset(
    {"Vector", "Plane", "Location", "Matrix", "Color", "Vertex"}
)
# Terminal queries read a shape instead of building one.
_QUERY_METHODS = frozenset(
    {
        "size", "val", "vals", "toTuple", "Volume", "Area", "Center",
        "BoundingBox", "isValid", "normalized", "cross", "dot", "Length",
    }
)


def fluent_chain_depth(code: str) -> int:
    """Longest run of chained CadQuery *modelling* calls, e.g. ``a.b().c()`` -> 2.

    Vector/Plane algebra and shape queries are excluded: the contract keeps those
    symbolic, so counting them would report a violation where none exists.
    """

    def root_of(node: ast.AST) -> ast.AST:
        while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            node = node.func.value
        return node

    def is_symbolic_root(node: ast.AST) -> bool:
        root = root_of(node)
        if isinstance(root, ast.Call):
            func = root.func
            if isinstance(func, ast.Attribute) and func.attr in _SYMBOLIC_ROOTS:
                return True
            if isinstance(func, ast.Name) and func.id in _SYMBOLIC_ROOTS:
                return True
        return False

    def measure(node: ast.AST) -> int:
        count = 0
        while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr not in _QUERY_METHODS:
                count += 1
            node = node.func.value
        return count

    depth = 0
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Call) and not is_symbolic_root(node):
            depth = max(depth, measure(node))
    return depth


@dataclass
class CodeComparison:
    """All AST-level gate results for one (source, canonical) pair."""

    source_parameters: int
    retained_parameters: int
    design_parameters: int = 0
    preamble_parameters: int = 0
    unhoisted_parameters: list[str] = field(default_factory=list)
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
    parameter_placement: str = "preamble"
    preamble: dict[str, Any] = field(default_factory=dict)
    late_layout: dict[str, Any] = field(default_factory=dict)

    @property
    def parameter_retention(self) -> float:
        if self.source_parameters == 0:
            return 1.0
        return self.retained_parameters / self.source_parameters

    @property
    def preamble_coverage(self) -> float:
        """Share of numeric design parameters still bound by name in a block."""

        if self.design_parameters == 0:
            return 1.0
        return self.preamble_parameters / self.design_parameters

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["parameter_retention"] = self.parameter_retention
        data["preamble_coverage"] = self.preamble_coverage
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
    parameter_placement: str = "preamble",
) -> CodeComparison:
    report = report or {}
    parameters = source_parameters(source, result_name)
    canonical_names = bound_names(canonical)
    canonical_loads = load_counts(canonical)
    canonical_attributes = attribute_parameters(canonical)

    canonical_containers = container_roots(canonical)
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
        if (
            canonical_loads.get(canonical_name, 0) == 0
            and parameter not in canonical_attributes
            and parameter not in canonical_containers
        ):
            unused.append(parameter)

    # A parameter that the source passed to a CadQuery call but the canonical
    # program passes as a bare literal has been constant-folded away.
    source_loads = load_counts(source)
    containers = container_roots(source) | canonical_containers
    inlined = [
        parameter
        for parameter, canonical_name in resolved.items()
        if source_loads.get(parameter, 0) > 0
        and canonical_loads.get(canonical_name, 0) == 0
        and parameter not in canonical_attributes
        and parameter not in containers
        and canonical_name not in containers
    ]

    numeric_design = design_parameters(source, result_name)
    canonical_preamble = parameter_block_names(
        canonical, result_name, parameter_placement
    )
    hoisted: set[str] = set()
    unhoisted: list[str] = []
    for parameter in sorted(numeric_design):
        canonical_name = resolve_parameter(
            parameter,
            canonical_preamble,
            flattened=report.get("flattened_namespaces"),
            versioned=report.get("versioned_names"),
        )
        if canonical_name:
            hoisted.add(parameter)
        else:
            unhoisted.append(parameter)

    source_numbers = numeric_literals(source)
    canonical_numbers = numeric_literals(canonical)
    source_strings = string_literals(source)
    canonical_strings = string_literals(canonical)

    return CodeComparison(
        source_parameters=len(parameters),
        retained_parameters=len(resolved),
        design_parameters=len(numeric_design),
        preamble_parameters=len(hoisted),
        unhoisted_parameters=unhoisted,
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
        parameter_placement=parameter_placement,
        preamble=preamble_layout(
            canonical,
            result_name,
            exempt=frozenset(report.get("loop_carried_names", ()) or ()),
        ).to_dict(),
        late_layout=late_parameter_layout(
            canonical,
            result_name,
            exempt=frozenset(report.get("loop_carried_names", ()) or ()),
        ).to_dict()
        if parameter_placement == "late"
        else {},
    )
