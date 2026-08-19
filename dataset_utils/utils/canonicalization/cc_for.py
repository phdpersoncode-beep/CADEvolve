"""Symbol-preserving CC-for canonicalization for CadQuery programs.

The legacy CADEvolve standardizer executes a program and records concrete CadQuery
calls.  That is useful for CADEvolve-C, but it necessarily loses named parameter
relationships.  This module instead rewrites Python syntax without executing the
program.  The result is suitable for parameter-aware metrics and feature-block
decomposition.
"""

from __future__ import annotations

import ast
import copy
import math
import operator
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal, Mapping, MutableMapping, Sequence


LoopMode = Literal["preserve", "unroll"]


@dataclass(frozen=True)
class CCForConfig:
    """Options for :func:`canonicalize_code`."""

    loop_mode: LoopMode = "preserve"
    max_unroll_iterations: int = 64
    flatten_namespaces: bool = True
    remove_dead_none: bool = True
    version_reassignments: bool = True
    hoist_parameters: bool = True
    explicit_workplanes: bool = True
    result_name: str = "result"

    def __post_init__(self) -> None:
        if self.loop_mode not in {"preserve", "unroll"}:
            raise ValueError(f"Unsupported loop mode: {self.loop_mode!r}")
        if self.max_unroll_iterations < 1:
            raise ValueError("max_unroll_iterations must be positive")


@dataclass
class CCForReport:
    """Machine-readable record of transformations and deliberate exceptions."""

    loop_mode: LoopMode
    flattened_namespaces: dict[str, dict[str, str]] = field(default_factory=dict)
    unrolled_loops: int = 0
    preserved_loops: int = 0
    removed_none_initializers: list[str] = field(default_factory=list)
    versioned_names: dict[str, list[str]] = field(default_factory=dict)
    loop_carried_names: list[str] = field(default_factory=list)
    hoisted_parameters: list[str] = field(default_factory=list)
    workplane_steps: int = 0
    structural_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalizationResult:
    code: str
    report: CCForReport


@dataclass(frozen=True)
class CanonicalAction:
    kind: Literal["preamble", "statement"]
    code: str


def _target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for element in target.elts:
            names.extend(_target_names(element))
        return names
    return []


class _ScopedNameCollector(ast.NodeVisitor):
    """Collect names without descending into nested Python scopes."""

    def __init__(self) -> None:
        self.loaded: set[str] = set()
        self.assigned: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loaded.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.assigned.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.assigned.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.assigned.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.assigned.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _names_in(node: ast.AST) -> tuple[set[str], set[str]]:
    collector = _ScopedNameCollector()
    collector.visit(node)
    return collector.loaded, collector.assigned


def _loaded_names(node: ast.AST) -> set[str]:
    return _names_in(node)[0]


def _assigned_names(node: ast.AST) -> set[str]:
    return _names_in(node)[1]


def _simple_assignment_name(stmt: ast.stmt) -> str | None:
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return stmt.targets[0].id
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id
    return None


def _assignment_value(stmt: ast.stmt) -> ast.AST | None:
    if isinstance(stmt, ast.Assign):
        return stmt.value
    if isinstance(stmt, ast.AnnAssign):
        return stmt.value
    return None


def _all_bound_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".")[0])
    return names


class _LoadNameRewriter(ast.NodeTransformer):
    """Rewrite reaching definitions while respecting local expression scopes."""

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self.mapping = mapping
        self.shadowed: list[set[str]] = [set()]

    def _is_shadowed(self, name: str) -> bool:
        return any(name in scope for scope in reversed(self.shadowed))

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if (
            isinstance(node.ctx, ast.Load)
            and node.id in self.mapping
            and not self._is_shadowed(node.id)
        ):
            return ast.copy_location(
                ast.Name(id=self.mapping[node.id], ctx=ast.Load()), node
            )
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        bound = {argument.arg for argument in node.args.args}
        bound.update(argument.arg for argument in node.args.posonlyargs)
        bound.update(argument.arg for argument in node.args.kwonlyargs)
        if node.args.vararg:
            bound.add(node.args.vararg.arg)
        if node.args.kwarg:
            bound.add(node.args.kwarg.arg)
        self.shadowed.append(bound)
        node.body = self.visit(node.body)
        self.shadowed.pop()
        return node

    def _visit_comprehension_expression(
        self,
        node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp,
    ) -> ast.AST:
        bound: set[str] = set()
        for generator in node.generators:
            generator.iter = self.visit(generator.iter)
            bound.update(_target_names(generator.target))
            self.shadowed.append(set(bound))
            generator.ifs = [self.visit(condition) for condition in generator.ifs]
            self.shadowed.pop()

        self.shadowed.append(bound)
        if isinstance(node, ast.DictComp):
            node.key = self.visit(node.key)
            node.value = self.visit(node.value)
        else:
            node.elt = self.visit(node.elt)
        self.shadowed.pop()
        return node

    visit_ListComp = _visit_comprehension_expression
    visit_SetComp = _visit_comprehension_expression
    visit_GeneratorExp = _visit_comprehension_expression
    visit_DictComp = _visit_comprehension_expression

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return node


def _rewrite_loads(node: ast.AST, mapping: Mapping[str, str]) -> ast.AST:
    return _LoadNameRewriter(mapping).visit(node)


# ---------------------------------------------------------------------------
# SimpleNamespace parameter flattening
# ---------------------------------------------------------------------------


class _NamespaceAttributeRewriter(ast.NodeTransformer):
    def __init__(self, mappings: Mapping[str, Mapping[str, str]]) -> None:
        self.mappings = mappings

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and isinstance(node.ctx, ast.Load):
            replacement = self.mappings.get(node.value.id, {}).get(node.attr)
            if replacement is not None:
                return ast.copy_location(ast.Name(id=replacement, ctx=ast.Load()), node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return node


def _simple_namespace_aliases(tree: ast.Module) -> tuple[set[str], set[str]]:
    constructor_names = {"SimpleNamespace"}
    module_names = {"types"}
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "types":
            for alias in stmt.names:
                if alias.name == "SimpleNamespace":
                    constructor_names.add(alias.asname or alias.name)
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name == "types":
                    module_names.add(alias.asname or alias.name)
    return constructor_names, module_names


def _is_namespace_call(
    node: ast.AST, constructor_names: set[str], module_names: set[str]
) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id in constructor_names
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "SimpleNamespace"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in module_names
    )


def _flatten_simple_namespaces(tree: ast.Module, report: CCForReport) -> ast.Module:
    constructors, modules = _simple_namespace_aliases(tree)
    reserved = _all_bound_names(tree)
    mappings: dict[str, dict[str, str]] = {}
    canonical_roots: dict[str, str] = {}

    def allocate(root: str, key: str, *, force_prefix: bool = False) -> str:
        candidate = f"{root}_{key}" if force_prefix else key
        if candidate in reserved:
            candidate = f"{root}_{key}"
        base = candidate
        suffix = 2
        while candidate in reserved:
            candidate = f"{base}_{suffix}"
            suffix += 1
        reserved.add(candidate)
        return candidate

    def flatten_call(
        root: str,
        call: ast.Call,
        location: ast.AST,
        *,
        nested: bool,
    ) -> tuple[list[ast.stmt], ast.Assign] | None:
        if call.args or any(keyword.arg is None for keyword in call.keywords):
            return None

        key_map: dict[str, str] = {}
        emitted: list[ast.stmt] = []
        for keyword in call.keywords:
            assert keyword.arg is not None
            if _is_namespace_call(keyword.value, constructors, modules):
                nested_call = keyword.value
                assert isinstance(nested_call, ast.Call)
                nested_root = allocate(root, keyword.arg)
                flattened_nested = flatten_call(
                    nested_root,
                    nested_call,
                    location,
                    nested=True,
                )
                if flattened_nested is None:
                    # Preserve the nested constructor as a named value even when
                    # its positional/** form cannot be recursively flattened.
                    assignment = ast.Assign(
                        targets=[ast.Name(id=nested_root, ctx=ast.Store())],
                        value=keyword.value,
                    )
                    emitted.append(ast.copy_location(assignment, location))
                else:
                    nested_statements, nested_reconstruction = flattened_nested
                    emitted.extend(nested_statements)
                    emitted.append(nested_reconstruction)
                key_map[keyword.arg] = nested_root
                continue

            if (
                isinstance(keyword.value, ast.Name)
                and keyword.value.id in reserved
            ):
                flattened = keyword.value.id
            else:
                flattened = allocate(
                    root, keyword.arg, force_prefix=nested
                )
                assignment = ast.Assign(
                    targets=[ast.Name(id=flattened, ctx=ast.Store())],
                    value=keyword.value,
                )
                emitted.append(ast.copy_location(assignment, location))
            key_map[keyword.arg] = flattened

        mappings[root] = key_map
        report.flattened_namespaces[root] = dict(key_map)
        reconstruction = ast.Assign(
            targets=[ast.Name(id=root, ctx=ast.Store())],
            value=ast.Call(
                func=copy.deepcopy(call.func),
                args=[],
                keywords=[
                    ast.keyword(
                        arg=keyword.arg,
                        value=ast.Name(
                            id=key_map[keyword.arg], ctx=ast.Load()
                        ),
                    )
                    for keyword in call.keywords
                    if keyword.arg is not None
                ],
            ),
        )
        return emitted, ast.copy_location(reconstruction, location)

    new_body: list[ast.stmt] = []
    for stmt in tree.body:
        name = _simple_assignment_name(stmt)
        value = _assignment_value(stmt)

        if name is not None and value is not None and _is_namespace_call(
            value, constructors, modules
        ):
            call = value
            assert isinstance(call, ast.Call)
            flattened_call = flatten_call(
                name, call, stmt, nested=False
            )
            if flattened_call is None:
                report.warnings.append(
                    f"Kept namespace {name!r}: positional or ** arguments are not safe to flatten"
                )
                new_body.append(stmt)
                continue
            emitted, reconstruction = flattened_call
            canonical_roots[name] = name
            new_body.extend(emitted)
            # Keep a symbolic compatibility object after exposing the individual
            # parameters.  Dataset programs often pass the namespace into a class
            # and dereference it through ``self.m`` inside a method.  Removing the
            # object in that case changes runtime semantics even though direct
            # module-level ``m.width`` references can be flattened safely.
            new_body.append(reconstruction)
            continue

        if (
            name is not None
            and isinstance(value, ast.Name)
            and value.id in mappings
        ):
            mappings[name] = mappings[value.id]
            canonical_roots[name] = canonical_roots.get(value.id, value.id)
            # Retain the alias because it may be passed into a class/function even
            # when all direct ``alias.field`` reads were flattened.
            new_body.append(stmt)
            continue

        new_body.append(stmt)

    if not mappings:
        return tree

    tree.body = new_body
    tree = _NamespaceAttributeRewriter(mappings).visit(tree)
    for alias, root in canonical_roots.items():
        if alias != root:
            report.flattened_namespaces[alias] = dict(mappings[alias])
    return tree


# ---------------------------------------------------------------------------
# Safe, bounded loop unrolling
# ---------------------------------------------------------------------------


_UNRESOLVED = object()
_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}
_COMPARE_OPS: dict[type[ast.cmpop], Any] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_SAFE_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "reversed": reversed,
    "tuple": tuple,
    "list": list,
}
_SAFE_MATH_FUNCTIONS: dict[str, Any] = {
    "radians": math.radians,
    "degrees": math.degrees,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
}


def _safe_eval(node: ast.AST, env: Mapping[str, Any]) -> Any:
    try:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return env.get(node.id, _UNRESOLVED)
        if isinstance(node, ast.Tuple):
            values = [_safe_eval(element, env) for element in node.elts]
            return _UNRESOLVED if _UNRESOLVED in values else tuple(values)
        if isinstance(node, ast.List):
            values = [_safe_eval(element, env) for element in node.elts]
            return _UNRESOLVED if _UNRESOLVED in values else values
        if isinstance(node, ast.Set):
            values = [_safe_eval(element, env) for element in node.elts]
            return _UNRESOLVED if _UNRESOLVED in values else set(values)
        if isinstance(node, ast.Dict):
            keys = [_safe_eval(key, env) for key in node.keys]
            values = [_safe_eval(value, env) for value in node.values]
            if _UNRESOLVED in keys or _UNRESOLVED in values:
                return _UNRESOLVED
            return dict(zip(keys, values))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            operand = _safe_eval(node.operand, env)
            if operand is _UNRESOLVED:
                return _UNRESOLVED
            return _UNARY_OPS[type(node.op)](operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left = _safe_eval(node.left, env)
            right = _safe_eval(node.right, env)
            if left is _UNRESOLVED or right is _UNRESOLVED:
                return _UNRESOLVED
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.BoolOp):
            values = [_safe_eval(value, env) for value in node.values]
            if _UNRESOLVED in values:
                return _UNRESOLVED
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare):
            left = _safe_eval(node.left, env)
            if left is _UNRESOLVED:
                return _UNRESOLVED
            for op_node, comparator in zip(node.ops, node.comparators):
                right = _safe_eval(comparator, env)
                fn = _COMPARE_OPS.get(type(op_node))
                if right is _UNRESOLVED or fn is None or not fn(left, right):
                    return False if right is not _UNRESOLVED and fn is not None else _UNRESOLVED
                left = right
            return True
        if isinstance(node, ast.IfExp):
            condition = _safe_eval(node.test, env)
            if condition is _UNRESOLVED:
                return _UNRESOLVED
            return _safe_eval(node.body if condition else node.orelse, env)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "math" and node.attr in {"pi", "tau", "e"}:
                return getattr(math, node.attr)
        if isinstance(node, ast.Call):
            fn: Any = None
            if isinstance(node.func, ast.Name):
                fn = _SAFE_FUNCTIONS.get(node.func.id)
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "math"
            ):
                fn = _SAFE_MATH_FUNCTIONS.get(node.func.attr)
            if fn is None or any(keyword.arg is None for keyword in node.keywords):
                return _UNRESOLVED
            args = [_safe_eval(argument, env) for argument in node.args]
            kwargs = {
                keyword.arg: _safe_eval(keyword.value, env)
                for keyword in node.keywords
                if keyword.arg is not None
            }
            if _UNRESOLVED in args or _UNRESOLVED in kwargs.values():
                return _UNRESOLVED
            return fn(*args, **kwargs)
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        return _UNRESOLVED
    return _UNRESOLVED


def _literal_node(value: Any) -> ast.AST:
    return ast.parse(repr(value), mode="eval").body


def _bind_static_target(target: ast.AST, value: Any, env: MutableMapping[str, Any]) -> bool:
    if isinstance(target, ast.Name):
        env[target.id] = value
        return True
    if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (tuple, list)):
        if len(target.elts) != len(value):
            return False
        return all(
            _bind_static_target(element, item, env)
            for element, item in zip(target.elts, value)
        )
    return False


def _target_assignment(target: ast.AST, value: Any, location: ast.AST) -> ast.Assign:
    assignment = ast.Assign(targets=[copy.deepcopy(target)], value=_literal_node(value))
    return ast.copy_location(assignment, location)


class _StaticLoopUnroller:
    def __init__(self, config: CCForConfig, report: CCForReport) -> None:
        self.config = config
        self.report = report

    @staticmethod
    def _unsafe_control_flow(node: ast.For) -> bool:
        return any(
            isinstance(child, (ast.Break, ast.Continue, ast.Yield, ast.YieldFrom))
            for child in ast.walk(node)
        )

    def transform_module(self, tree: ast.Module) -> ast.Module:
        tree.body, _ = self._transform_block(tree.body, {})
        return tree

    def _transform_block(
        self, statements: Sequence[ast.stmt], initial_env: Mapping[str, Any]
    ) -> tuple[list[ast.stmt], dict[str, Any]]:
        env = dict(initial_env)
        output: list[ast.stmt] = []

        for original in statements:
            stmt = copy.deepcopy(original)
            name = _simple_assignment_name(stmt)
            value_node = _assignment_value(stmt)

            if isinstance(stmt, ast.For):
                iterable = _safe_eval(stmt.iter, env)
                if iterable is not _UNRESOLVED and not self._unsafe_control_flow(stmt):
                    try:
                        values = list(iterable)
                    except TypeError:
                        values = []
                        iterable = _UNRESOLVED

                    if (
                        iterable is not _UNRESOLVED
                        and len(values) <= self.config.max_unroll_iterations
                    ):
                        self.report.unrolled_loops += 1
                        last_env = dict(env)
                        for item in values:
                            iteration_env = dict(env)
                            if not _bind_static_target(stmt.target, item, iteration_env):
                                iterable = _UNRESOLVED
                                break
                            output.append(_target_assignment(stmt.target, item, stmt))
                            expanded, iteration_env = self._transform_block(
                                stmt.body, iteration_env
                            )
                            output.extend(expanded)
                            last_env = iteration_env
                        if iterable is not _UNRESOLVED:
                            if not values and stmt.orelse:
                                expanded_else, last_env = self._transform_block(
                                    stmt.orelse, env
                                )
                                output.extend(expanded_else)
                            env.update(last_env)
                            continue

                self.report.preserved_loops += 1
                self.report.warnings.append(
                    f"Preserved non-static or oversized loop at line {getattr(stmt, 'lineno', '?')}"
                )
                stmt.body, _ = self._transform_block(stmt.body, env)
                stmt.orelse, _ = self._transform_block(stmt.orelse, env)
                output.append(stmt)
                for assigned in _assigned_names(stmt):
                    env.pop(assigned, None)
                continue

            if isinstance(stmt, ast.If):
                stmt.body, _ = self._transform_block(stmt.body, env)
                stmt.orelse, _ = self._transform_block(stmt.orelse, env)
                output.append(stmt)
                for assigned in _assigned_names(stmt):
                    env.pop(assigned, None)
                continue

            output.append(stmt)
            if name is not None and value_node is not None:
                value = _safe_eval(value_node, env)
                if value is _UNRESOLVED:
                    env.pop(name, None)
                else:
                    env[name] = value
            else:
                for assigned in _assigned_names(stmt):
                    env.pop(assigned, None)

        return output, env


def _count_preserved_loops(tree: ast.Module, report: CCForReport) -> None:
    report.preserved_loops = sum(isinstance(node, ast.For) for node in ast.walk(tree))


# ---------------------------------------------------------------------------
# Dead ``None`` initializers
# ---------------------------------------------------------------------------


def _is_none_assignment(stmt: ast.stmt) -> str | None:
    name = _simple_assignment_name(stmt)
    value = _assignment_value(stmt)
    if name is not None and isinstance(value, ast.Constant) and value.value is None:
        return name
    return None


def _remove_dead_none_from_block(
    statements: list[ast.stmt], report: CCForReport
) -> list[ast.stmt]:
    result: list[ast.stmt] = []
    for index, stmt in enumerate(statements):
        for field_name in ("body", "orelse", "finalbody"):
            child = getattr(stmt, field_name, None)
            if isinstance(child, list):
                setattr(stmt, field_name, _remove_dead_none_from_block(child, report))
        if isinstance(stmt, ast.Try):
            for handler in stmt.handlers:
                handler.body = _remove_dead_none_from_block(handler.body, report)

        name = _is_none_assignment(stmt)
        if name is None:
            result.append(stmt)
            continue

        removable = False
        for later in statements[index + 1 :]:
            loads = _loaded_names(later)
            if name in loads:
                break
            if _simple_assignment_name(later) == name:
                removable = True
                break
            if name in _assigned_names(later):
                break
        if removable:
            report.removed_none_initializers.append(name)
        else:
            result.append(stmt)
    return result


# ---------------------------------------------------------------------------
# Loop-aware reaching-definition renaming
# ---------------------------------------------------------------------------


class _DefinitionCounter(ast.NodeVisitor):
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def _count_target(self, target: ast.AST) -> None:
        self.counts.update(_target_names(target))

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._count_target(target)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._count_target(node.target)
        if node.value:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._count_target(node.target)
        self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        self._count_target(node.target)
        self.visit(node.iter)
        for stmt in node.body + node.orelse:
            self.visit(stmt)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)  # type: ignore[arg-type]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


def _loads_before_definition(statements: Sequence[ast.stmt]) -> set[str]:
    seen: set[str] = set()
    before: set[str] = set()
    for stmt in statements:
        loads = _loaded_names(stmt)
        before.update(name for name in loads if name not in seen)
        if isinstance(stmt, ast.For):
            seen.update(_target_names(stmt.target))
        else:
            seen.update(_assigned_names(stmt))
    return before


class _ProtectedNameCollector(ast.NodeVisitor):
    """Find definitions that cannot be converted to plain lexical SSA."""

    def __init__(self) -> None:
        self.protected: set[str] = set()
        self.loop_carried: set[str] = set()

    def visit_For(self, node: ast.For) -> None:
        assigned = set().union(*(_assigned_names(stmt) for stmt in node.body)) if node.body else set()
        carried = assigned & _loads_before_definition(node.body)
        self.protected.update(carried)
        self.loop_carried.update(carried)
        for stmt in node.body + node.orelse:
            self.visit(stmt)

    def visit_While(self, node: ast.While) -> None:
        assigned = set().union(*(_assigned_names(stmt) for stmt in node.body)) if node.body else set()
        self.protected.update(assigned)
        self.loop_carried.update(assigned)
        for stmt in node.body + node.orelse:
            self.visit(stmt)

    def visit_If(self, node: ast.If) -> None:
        self.protected.update(_assigned_names(node))
        for stmt in node.body + node.orelse:
            self.visit(stmt)

    def visit_Try(self, node: ast.Try) -> None:
        self.protected.update(_assigned_names(node))
        for stmt in node.body + node.orelse + node.finalbody:
            self.visit(stmt)
        for handler in node.handlers:
            for stmt in handler.body:
                self.visit(stmt)

    def visit_Match(self, node: ast.Match) -> None:
        self.protected.update(_assigned_names(node))
        for case in node.cases:
            for stmt in case.body:
                self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


class _SSARenamer:
    def __init__(
        self, tree: ast.Module, report: CCForReport, result_name: str
    ) -> None:
        counter = _DefinitionCounter()
        counter.visit(tree)
        protected = _ProtectedNameCollector()
        protected.visit(tree)

        self.repeated = {name for name, count in counter.counts.items() if count > 1}
        self.protected = protected.protected | {result_name}
        self.report = report
        self.report.loop_carried_names = sorted(protected.loop_carried)
        self.next_index: defaultdict[str, int] = defaultdict(int)
        self.reserved = _all_bound_names(tree)
        self.versions: defaultdict[str, list[str]] = defaultdict(list)

    def _fresh(self, name: str) -> str:
        while True:
            self.next_index[name] += 1
            candidate = f"{name}_{self.next_index[name]}"
            if candidate not in self.reserved:
                self.reserved.add(candidate)
                self.versions[name].append(candidate)
                return candidate

    def _rename_target(
        self, target: ast.AST, env: MutableMapping[str, str]
    ) -> ast.AST:
        if isinstance(target, ast.Name):
            original = target.id
            if original in self.repeated and original not in self.protected:
                renamed = self._fresh(original)
                env[original] = renamed
                return ast.copy_location(ast.Name(id=renamed, ctx=target.ctx), target)
            env[original] = original
            return target
        if isinstance(target, (ast.Tuple, ast.List)):
            target.elts = [self._rename_target(element, env) for element in target.elts]
            return target
        if isinstance(target, (ast.Attribute, ast.Subscript)):
            return _rewrite_loads(target, env)
        return target

    def transform(self, tree: ast.Module) -> ast.Module:
        tree.body, _ = self._block(tree.body, {})
        self.report.versioned_names = {
            name: versions for name, versions in sorted(self.versions.items())
        }
        for name in sorted(self.repeated & self.protected):
            self.report.warnings.append(
                f"Kept control-flow name {name!r} stable to preserve runtime semantics"
            )
        return tree

    def _block(
        self, statements: Sequence[ast.stmt], initial_env: Mapping[str, str]
    ) -> tuple[list[ast.stmt], dict[str, str]]:
        env = dict(initial_env)
        output: list[ast.stmt] = []

        for stmt in statements:
            if isinstance(stmt, ast.Assign):
                stmt.value = _rewrite_loads(stmt.value, env)
                stmt.targets = [self._rename_target(target, env) for target in stmt.targets]
                output.append(stmt)
                continue

            if isinstance(stmt, ast.AnnAssign):
                if stmt.value is not None:
                    stmt.value = _rewrite_loads(stmt.value, env)
                stmt.target = self._rename_target(stmt.target, env)
                output.append(stmt)
                continue

            if isinstance(stmt, ast.AugAssign):
                stmt.target = _rewrite_loads(stmt.target, env)
                stmt.value = _rewrite_loads(stmt.value, env)
                output.append(stmt)
                continue

            if isinstance(stmt, (ast.For, ast.AsyncFor)):
                stmt.iter = _rewrite_loads(stmt.iter, env)
                body_env = dict(env)
                stmt.target = self._rename_target(stmt.target, body_env)
                assigned_original = _assigned_names(ast.Module(body=stmt.body, type_ignores=[]))
                stmt.body, body_env = self._block(stmt.body, body_env)
                stmt.orelse, _ = self._block(stmt.orelse, env)
                for name in assigned_original | set(_target_names(stmt.target)):
                    if name in body_env:
                        env[name] = body_env[name]
                output.append(stmt)
                continue

            if isinstance(stmt, ast.While):
                stmt.test = _rewrite_loads(stmt.test, env)
                stmt.body, _ = self._block(stmt.body, env)
                stmt.orelse, _ = self._block(stmt.orelse, env)
                for name in _assigned_names(stmt):
                    env[name] = name
                output.append(stmt)
                continue

            if isinstance(stmt, ast.If):
                stmt.test = _rewrite_loads(stmt.test, env)
                stmt.body, _ = self._block(stmt.body, env)
                stmt.orelse, _ = self._block(stmt.orelse, env)
                for name in _assigned_names(stmt):
                    env[name] = name
                output.append(stmt)
                continue

            if isinstance(stmt, ast.Try):
                stmt.body, _ = self._block(stmt.body, env)
                stmt.orelse, _ = self._block(stmt.orelse, env)
                stmt.finalbody, _ = self._block(stmt.finalbody, env)
                for handler in stmt.handlers:
                    handler.body, _ = self._block(handler.body, env)
                for name in _assigned_names(stmt):
                    env[name] = name
                output.append(stmt)
                continue

            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                output.append(stmt)
                continue

            output.append(_rewrite_loads(stmt, env))

        return output, env


# ---------------------------------------------------------------------------
# Parameter preamble extraction
# ---------------------------------------------------------------------------


_SAFE_DATA_CALLS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "map",
    "max",
    "min",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}
_SAFE_CQ_DATA_CALLS = {"Location", "Plane", "Vector"}


class _PureDataExpression(ast.NodeVisitor):
    def __init__(self, geometry_names: set[str]) -> None:
        self.geometry_names = geometry_names
        self.pure = True

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id in self.geometry_names:
            self.pure = False

    def visit_Call(self, node: ast.Call) -> None:
        allowed = False
        if isinstance(node.func, ast.Name):
            allowed = node.func.id in _SAFE_DATA_CALLS
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "math":
                allowed = True
            elif node.func.value.id == "cq" and node.func.attr in _SAFE_CQ_DATA_CALLS:
                allowed = True
        if not allowed:
            self.pure = False
            return
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _is_pure_data_expression(node: ast.AST, geometry_names: set[str]) -> bool:
    visitor = _PureDataExpression(geometry_names)
    visitor.visit(node)
    return visitor.pure


_DIRECT_GEOMETRY_CONSTRUCTORS = {"Sketch", "Workplane"}
_GEOMETRY_FACTORY_TYPES = {
    "Compound",
    "Edge",
    "Face",
    "Shape",
    "Shell",
    "Solid",
    "Vertex",
    "Wire",
}


def _is_workplane_constructor(node: ast.AST) -> bool:
    """Recognize CadQuery objects that start a fluent geometry chain.

    The historical helper name is retained for compatibility inside this module,
    but Zero-to-CAD also uses ``cq.Sketch()`` and factories such as
    ``cq.Solid.makeCone(...)`` as chain roots.
    """

    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id in _DIRECT_GEOMETRY_CONSTRUCTORS
    if (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "cq"
        and node.func.attr in _DIRECT_GEOMETRY_CONSTRUCTORS
    ):
        return True
    return (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "cq"
        and node.func.value.attr in _GEOMETRY_FACTORY_TYPES
    )


def _expression_uses_geometry(node: ast.AST, geometry_names: set[str]) -> bool:
    if _is_workplane_constructor(node):
        return True
    return bool(_loaded_names(node) & geometry_names)


def _infer_top_level_geometry_names(tree: ast.Module, result_name: str) -> set[str]:
    geometry: set[str] = set()
    changed = True
    while changed:
        changed = False
        for stmt in tree.body:
            name = _simple_assignment_name(stmt)
            value = _assignment_value(stmt)
            if name is None or value is None or name in geometry:
                continue
            if name == result_name or _expression_uses_geometry(value, geometry):
                geometry.add(name)
                changed = True
    return geometry


class _ControlFlowMutationCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.mutated: set[str] = set()
        self.depth = 0

    def _visit_control_body(self, bodies: Iterable[Sequence[ast.stmt]]) -> None:
        self.depth += 1
        for body in bodies:
            for stmt in body:
                self.visit(stmt)
        self.depth -= 1

    def visit_For(self, node: ast.For) -> None:
        self.mutated.update(_target_names(node.target))
        self._visit_control_body((node.body, node.orelse))

    def visit_While(self, node: ast.While) -> None:
        self._visit_control_body((node.body, node.orelse))

    def visit_If(self, node: ast.If) -> None:
        self._visit_control_body((node.body, node.orelse))

    def visit_Try(self, node: ast.Try) -> None:
        bodies = [node.body, node.orelse, node.finalbody]
        bodies.extend(handler.body for handler in node.handlers)
        self._visit_control_body(bodies)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self.depth:
            for target in node.targets:
                self.mutated.update(_target_names(target))
        self.generic_visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self.depth:
            self.mutated.update(_target_names(node.target))
        if node.value:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.mutated.update(_target_names(node.target))
        self.visit(node.value)

    def visit_Expr(self, node: ast.Expr) -> None:
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr in {"add", "append", "extend", "insert", "update"}
            and isinstance(node.value.func.value, ast.Name)
        ):
            self.mutated.add(node.value.func.value.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


def _hoist_parameter_assignments(
    tree: ast.Module, report: CCForReport, result_name: str
) -> ast.Module:
    geometry = _infer_top_level_geometry_names(tree, result_name)
    mutation_collector = _ControlFlowMutationCollector()
    mutation_collector.visit(tree)

    imports: list[ast.stmt] = []
    docstrings: list[ast.stmt] = []
    body: list[ast.stmt] = []
    for index, stmt in enumerate(tree.body):
        if (
            index == 0
            and isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            docstrings.append(stmt)
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            imports.append(stmt)
        else:
            body.append(stmt)

    local_defs = {
        name
        for stmt in body
        for name in ([ _simple_assignment_name(stmt) ] if _simple_assignment_name(stmt) else [])
    }
    hoisted_names: set[str] = set()
    parameters: list[ast.stmt] = []
    remainder: list[ast.stmt] = []

    for stmt in body:
        name = _simple_assignment_name(stmt)
        value = _assignment_value(stmt)
        if name is None or value is None or name == result_name or name in geometry:
            remainder.append(stmt)
            continue

        loads = _loaded_names(value)
        local_dependencies = loads & local_defs
        dependencies_ready = local_dependencies <= hoisted_names
        is_hoistable = (
            dependencies_ready
            and not (loads & mutation_collector.mutated)
            and _is_pure_data_expression(value, geometry)
        )
        if is_hoistable:
            parameters.append(stmt)
            hoisted_names.add(name)
            report.hoisted_parameters.append(name)
        else:
            remainder.append(stmt)

    tree.body = docstrings + imports + parameters + remainder
    return tree


# ---------------------------------------------------------------------------
# Explicit Workplane lowering
# ---------------------------------------------------------------------------


_NON_GEOMETRY_METHODS = {
    "Area",
    "BoundingBox",
    "Center",
    "Length",
    "Volume",
    "isValid",
    "size",
    "toTuple",
}

# Methods whose receiver/result participates in CadQuery's fluent modeling state.
# This also lets helper functions accept a Workplane under domain names such as
# ``part`` without requiring fragile interprocedural type inference.
_CADQUERY_GEOMETRY_METHODS = {
    "add",
    "all",
    "arc",
    "assemble",
    "bezier",
    "box",
    "cboreHole",
    "center",
    "chamfer",
    "circle",
    "clean",
    "close",
    "combine",
    "combineSolids",
    "cone",
    "copyWorkplane",
    "cskHole",
    "cut",
    "cutBlind",
    "cutEach",
    "cutThruAll",
    "each",
    "eachpoint",
    "edge",
    "edges",
    "ellipse",
    "end",
    "extrude",
    "face",
    "faces",
    "fillet",
    "findSolid",
    "first",
    "fuse",
    "hLine",
    "hLineTo",
    "hole",
    "hull",
    "interpPlate",
    "intersect",
    "line",
    "lineTo",
    "loft",
    "mirror",
    "mirrorX",
    "mirrorY",
    "move",
    "moved",
    "moveTo",
    "newObject",
    "offset2D",
    "parametricCurve",
    "placeSketch",
    "polarArray",
    "polygon",
    "polyline",
    "push",
    "pushPoints",
    "rarray",
    "rect",
    "revolve",
    "rotate",
    "rotateAboutCenter",
    "segment",
    "shell",
    "slot",
    "slot2D",
    "solid",
    "solids",
    "sphere",
    "spline",
    "split",
    "sweep",
    "tangentArcPoint",
    "text",
    "toPending",
    "translate",
    "transformed",
    "trapezoid",
    "twistExtrude",
    "union",
    "vLine",
    "vLineTo",
    "val",
    "vertex",
    "vertices",
    "wire",
    "wires",
    "workplane",
    "workplaneFromTagged",
}

_GEOMETRY_PARAMETER_HINTS = {
    "base",
    "body",
    "model",
    "part",
    "result",
    "shape",
    "solid",
    "workplane",
    "wp",
}
_GEOMETRY_ATTRIBUTE_HINTS = {
    "base",
    "body",
    "model",
    "panel",
    "part",
    "result",
    "shape",
    "solid",
    "workplane",
    "wp",
}


def _state_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _state_key(node.value)
        if parent is not None:
            return f"{parent}.{node.attr}"
    return None


def _geometry_factory_names(tree: ast.AST) -> set[str]:
    """Find helpers that return a value built in a CadQuery geometry scope."""

    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        has_value_return = any(
            isinstance(child, ast.Return) and child.value is not None
            for child in ast.walk(node)
        )
        starts_geometry = any(
            isinstance(child, ast.Call) and _is_workplane_constructor(child)
            for child in ast.walk(node)
        )
        arguments = {
            argument.arg
            for argument in (
                list(node.args.posonlyargs)
                + list(node.args.args)
                + list(node.args.kwonlyargs)
            )
        }
        transforms_geometry_argument = bool(
            arguments & _GEOMETRY_PARAMETER_HINTS
        ) and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            for child in ast.walk(node)
        )
        if has_value_return and (
            starts_geometry or transforms_geometry_argument
        ):
            names.add(node.name)
    return names


class _WorkplaneLowerer:
    def __init__(self, tree: ast.Module, report: CCForReport, result_name: str) -> None:
        self.report = report
        self.result_name = result_name
        self.reserved = _all_bound_names(tree)
        self.index = 0
        self.helper_index = 0
        self.saw_result = False
        self.geometry_factories = _geometry_factory_names(tree)
        self.global_result_state: str | None = None
        if any(
            isinstance(node, ast.Global) and result_name in node.names
            for node in ast.walk(tree)
        ):
            self.global_result_state = self._state_name(result_name)

    def _state_name(self, original: str) -> str:
        """Return a collision-free stable name for loop-carried geometry."""

        if original != self.result_name:
            return original
        base = f"{original}_state"
        candidate = base
        suffix = 2
        while candidate in self.reserved:
            candidate = f"{base}_{suffix}"
            suffix += 1
        self.reserved.add(candidate)
        return candidate

    def _new_wp(self) -> str:
        while True:
            self.index += 1
            name = f"wp{self.index}"
            if name not in self.reserved:
                self.reserved.add(name)
                self.report.workplane_steps += 1
                return name

    def _new_helper(self) -> str:
        while True:
            self.helper_index += 1
            name = f"_cc_for_lambda_{self.helper_index}"
            if name not in self.reserved:
                self.reserved.add(name)
                return name

    def _emit_geometry_call(self, call: ast.Call, location: ast.AST) -> tuple[ast.Assign, ast.Name]:
        name = self._new_wp()
        assignment = ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())], value=call
        )
        return ast.copy_location(assignment, location), ast.Name(id=name, ctx=ast.Load())

    def _lower_expr(
        self,
        node: ast.AST,
        aliases: Mapping[str, str],
        dynamic_geometry: Mapping[str, str],
    ) -> tuple[list[ast.stmt], ast.AST, bool]:
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and node.id in aliases:
                return [], ast.copy_location(
                    ast.Name(id=aliases[node.id], ctx=ast.Load()), node
                ), True
            if isinstance(node.ctx, ast.Load) and node.id in dynamic_geometry:
                return [], ast.copy_location(
                    ast.Name(id=dynamic_geometry[node.id], ctx=ast.Load()), node
                ), True
            return [], node, False

        if isinstance(node, ast.Call):
            prefix: list[ast.stmt] = []
            receiver_is_geometry = False
            if isinstance(node.func, ast.Attribute):
                before, receiver, receiver_is_geometry = self._lower_expr(
                    node.func.value, aliases, dynamic_geometry
                )
                prefix.extend(before)
                method_name = node.func.attr
                if (
                    not receiver_is_geometry
                    and method_name in _CADQUERY_GEOMETRY_METHODS
                    and isinstance(receiver, ast.Call)
                ):
                    assignment, receiver_name = self._emit_geometry_call(
                        receiver, node.func.value
                    )
                    prefix.append(assignment)
                    receiver = receiver_name
                    receiver_is_geometry = True
                function: ast.expr = ast.Attribute(
                    value=receiver, attr=node.func.attr, ctx=ast.Load()
                )
            else:
                function = _rewrite_loads(node.func, aliases)
                method_name = node.func.id if isinstance(node.func, ast.Name) else ""

            arguments: list[ast.expr] = []
            for argument in node.args:
                before, lowered, _ = self._lower_expr(
                    argument, aliases, dynamic_geometry
                )
                prefix.extend(before)
                arguments.append(lowered)  # type: ignore[arg-type]

            keywords: list[ast.keyword] = []
            for keyword in node.keywords:
                before, lowered, _ = self._lower_expr(
                    keyword.value, aliases, dynamic_geometry
                )
                prefix.extend(before)
                keywords.append(ast.keyword(arg=keyword.arg, value=lowered))

            call = ast.Call(func=function, args=arguments, keywords=keywords)
            ast.copy_location(call, node)
            constructor = _is_workplane_constructor(call) or (
                isinstance(node.func, ast.Name)
                and node.func.id in self.geometry_factories
            ) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in self.geometry_factories
            )
            returns_geometry = constructor or (
                (
                    receiver_is_geometry
                    or method_name in _CADQUERY_GEOMETRY_METHODS
                )
                and method_name not in _NON_GEOMETRY_METHODS
            )
            if returns_geometry:
                assignment, name = self._emit_geometry_call(call, node)
                prefix.append(assignment)
                return prefix, name, True
            return prefix, call, False

        if isinstance(node, ast.Attribute):
            key = _state_key(node)
            if isinstance(node.ctx, ast.Load) and key is not None:
                if key in aliases:
                    return [], ast.copy_location(
                        ast.Name(id=aliases[key], ctx=ast.Load()), node
                    ), True
                if key in dynamic_geometry:
                    return [], ast.copy_location(
                        ast.Name(id=dynamic_geometry[key], ctx=ast.Load()), node
                    ), True
            prefix, value, is_geometry = self._lower_expr(
                node.value, aliases, dynamic_geometry
            )
            if (
                key is not None
                and key.startswith("self.")
                and node.attr in _GEOMETRY_ATTRIBUTE_HINTS
            ):
                is_geometry = True
            return prefix, ast.copy_location(
                ast.Attribute(value=value, attr=node.attr, ctx=node.ctx), node
            ), is_geometry

        if isinstance(node, ast.Subscript):
            prefix, value, is_geometry = self._lower_expr(
                node.value, aliases, dynamic_geometry
            )
            _, slice_node, _ = self._lower_expr(node.slice, aliases, dynamic_geometry)
            return prefix, ast.copy_location(
                ast.Subscript(value=value, slice=slice_node, ctx=node.ctx), node
            ), is_geometry

        if isinstance(node, ast.BinOp):
            left_prefix, left, left_geometry = self._lower_expr(
                node.left, aliases, dynamic_geometry
            )
            right_prefix, right, right_geometry = self._lower_expr(
                node.right, aliases, dynamic_geometry
            )
            return left_prefix + right_prefix, ast.copy_location(
                ast.BinOp(left=left, op=node.op, right=right), node
            ), left_geometry or right_geometry

        if isinstance(node, ast.IfExp):
            test = _rewrite_loads(node.test, aliases)
            body_prefix, body, body_geometry = self._lower_expr(
                node.body, aliases, dynamic_geometry
            )
            else_prefix, orelse, else_geometry = self._lower_expr(
                node.orelse, aliases, dynamic_geometry
            )
            if body_geometry and else_geometry:
                wp_name = self._new_wp()
                body_statements = body_prefix + [
                    ast.Assign(
                        targets=[ast.Name(id=wp_name, ctx=ast.Store())],
                        value=body,
                    )
                ]
                else_statements = else_prefix + [
                    ast.Assign(
                        targets=[ast.Name(id=wp_name, ctx=ast.Store())],
                        value=orelse,
                    )
                ]
                conditional = ast.If(
                    test=test,
                    body=body_statements,
                    orelse=else_statements,
                )
                return [ast.copy_location(conditional, node)], ast.Name(
                    id=wp_name, ctx=ast.Load()
                ), True
            # Do not move branch prefixes outside a conditional expression: that
            # would evaluate an originally lazy branch. Unknown mixed-type cases
            # stay intact and can be recognized at their next CadQuery use.
            return [], ast.copy_location(
                ast.IfExp(
                    test=test,
                    body=_rewrite_loads(node.body, aliases),
                    orelse=_rewrite_loads(node.orelse, aliases),
                ),
                node,
            ), False

        if isinstance(node, ast.UnaryOp):
            prefix, operand, _ = self._lower_expr(node.operand, aliases, dynamic_geometry)
            return prefix, ast.copy_location(
                ast.UnaryOp(op=node.op, operand=operand), node
            ), False

        if isinstance(node, ast.Lambda):
            body_prefix, body, _ = self._lower_expr(
                node.body, aliases, dynamic_geometry
            )
            if not body_prefix:
                return [], node, False
            helper_name = self._new_helper()
            helper = ast.FunctionDef(
                name=helper_name,
                args=copy.deepcopy(node.args),
                body=body_prefix + [ast.Return(value=body)],
                decorator_list=[],
                returns=None,
                type_comment=None,
            )
            return [ast.copy_location(helper, node)], ast.copy_location(
                ast.Name(id=helper_name, ctx=ast.Load()), node
            ), False

        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            prefix: list[ast.stmt] = []
            elements: list[ast.expr] = []
            for element in node.elts:
                before, lowered, _ = self._lower_expr(
                    element, aliases, dynamic_geometry
                )
                prefix.extend(before)
                elements.append(lowered)  # type: ignore[arg-type]
            if isinstance(node, ast.Tuple):
                rebuilt: ast.AST = ast.Tuple(elts=elements, ctx=node.ctx)
            elif isinstance(node, ast.List):
                rebuilt = ast.List(elts=elements, ctx=node.ctx)
            else:
                rebuilt = ast.Set(elts=elements)
            return prefix, ast.copy_location(rebuilt, node), False

        if isinstance(node, ast.Dict):
            prefix: list[ast.stmt] = []
            keys: list[ast.expr | None] = []
            values: list[ast.expr] = []
            for key in node.keys:
                if key is None:
                    keys.append(None)
                else:
                    before, lowered, _ = self._lower_expr(key, aliases, dynamic_geometry)
                    prefix.extend(before)
                    keys.append(lowered)  # type: ignore[arg-type]
            for value in node.values:
                before, lowered, _ = self._lower_expr(value, aliases, dynamic_geometry)
                prefix.extend(before)
                values.append(lowered)  # type: ignore[arg-type]
            return prefix, ast.copy_location(ast.Dict(keys=keys, values=values), node), False

        return [], _rewrite_loads(node, aliases), False

    def _materialize_alias(
        self,
        name: str,
        aliases: MutableMapping[str, str],
        dynamic: MutableMapping[str, str],
    ) -> ast.Assign | None:
        current = aliases.pop(name, None)
        stable_name = dynamic.setdefault(name, self._state_name(name))
        if current is None or current == stable_name:
            return None
        return ast.Assign(
            targets=[ast.Name(id=stable_name, ctx=ast.Store())],
            value=ast.Name(id=current, ctx=ast.Load()),
        )

    def _lower_block(
        self,
        statements: Sequence[ast.stmt],
        initial_aliases: Mapping[str, str],
        initial_dynamic: Mapping[str, str],
        *,
        tracks_result: bool = True,
    ) -> tuple[list[ast.stmt], dict[str, str], dict[str, str]]:
        aliases = dict(initial_aliases)
        dynamic = dict(initial_dynamic)
        output: list[ast.stmt] = []

        for stmt in statements:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                prefix, value, is_geometry = self._lower_expr(
                    stmt.value, aliases, dynamic
                )
                output.extend(prefix)
                if isinstance(target, ast.Name):
                    name = target.id
                    if is_geometry:
                        if not isinstance(value, ast.Name):
                            wp_name = self._new_wp()
                            output.append(
                                ast.copy_location(
                                    ast.Assign(
                                        targets=[
                                            ast.Name(
                                                id=wp_name, ctx=ast.Store()
                                            )
                                        ],
                                        value=value,
                                    ),
                                    stmt,
                                )
                            )
                            value = ast.Name(id=wp_name, ctx=ast.Load())
                        assert isinstance(value, ast.Name)
                        if name in dynamic:
                            output.append(
                                ast.copy_location(
                                    ast.Assign(
                                        targets=[
                                            ast.Name(id=dynamic[name], ctx=ast.Store())
                                        ],
                                        value=value,
                                    ),
                                    stmt,
                                )
                            )
                        else:
                            aliases[name] = value.id
                        if tracks_result and name == self.result_name:
                            self.saw_result = True
                        continue
                    if tracks_result and name == self.result_name:
                        # A factory/class expression can return a Workplane without
                        # looking like a CadQuery call statically.  Give it an
                        # explicit state name and still suppress all non-terminal
                        # ``result`` definitions.
                        wp_name = self._new_wp()
                        output.append(
                            ast.copy_location(
                                ast.Assign(
                                    targets=[
                                        ast.Name(id=wp_name, ctx=ast.Store())
                                    ],
                                    value=value,
                                ),
                                stmt,
                            )
                        )
                        if name in dynamic:
                            output.append(
                                ast.copy_location(
                                    ast.Assign(
                                        targets=[
                                            ast.Name(
                                                id=dynamic[name], ctx=ast.Store()
                                            )
                                        ],
                                        value=ast.Name(
                                            id=wp_name, ctx=ast.Load()
                                        ),
                                    ),
                                    stmt,
                                )
                            )
                        else:
                            aliases[name] = wp_name
                        self.saw_result = True
                        continue
                    aliases.pop(name, None)
                    if name in dynamic:
                        dynamic.pop(name)
                    stmt.value = value
                    output.append(stmt)
                    continue

                stmt.value = value
                stmt.targets = [
                    _rewrite_loads(target, aliases) for target in stmt.targets
                ]
                output.append(stmt)
                key = _state_key(target)
                if key is not None:
                    if is_geometry and isinstance(value, ast.Name):
                        aliases[key] = value.id
                        dynamic.pop(key, None)
                    else:
                        aliases.pop(key, None)
                        dynamic.pop(key, None)
                continue

            if isinstance(stmt, ast.AnnAssign):
                if stmt.value is not None:
                    prefix, value, is_geometry = self._lower_expr(
                        stmt.value, aliases, dynamic
                    )
                    output.extend(prefix)
                    if isinstance(stmt.target, ast.Name) and is_geometry:
                        if not isinstance(value, ast.Name):
                            wp_name = self._new_wp()
                            output.append(
                                ast.copy_location(
                                    ast.Assign(
                                        targets=[
                                            ast.Name(
                                                id=wp_name, ctx=ast.Store()
                                            )
                                        ],
                                        value=value,
                                    ),
                                    stmt,
                                )
                            )
                            value = ast.Name(id=wp_name, ctx=ast.Load())
                        assert isinstance(value, ast.Name)
                        aliases[stmt.target.id] = value.id
                        continue
                    stmt.value = value
                output.append(_rewrite_loads(stmt, aliases))
                continue

            if isinstance(stmt, ast.For):
                assigned = _assigned_names(ast.Module(body=stmt.body, type_ignores=[]))
                carried = assigned & _loads_before_definition(stmt.body)
                for name in sorted(carried & (set(aliases) | set(dynamic))):
                    initializer = self._materialize_alias(name, aliases, dynamic)
                    if initializer is not None:
                        output.append(ast.copy_location(initializer, stmt))

                stmt.iter = _rewrite_loads(stmt.iter, aliases)
                stmt.body, body_aliases, body_dynamic = self._lower_block(
                    stmt.body,
                    aliases,
                    dynamic,
                    tracks_result=tracks_result,
                )
                stmt.orelse, _, _ = self._lower_block(
                    stmt.orelse,
                    aliases,
                    dynamic,
                    tracks_result=tracks_result,
                )
                if not stmt.body:
                    stmt.body = [ast.copy_location(ast.Pass(), stmt)]
                for name in assigned:
                    if name in body_dynamic:
                        dynamic[name] = body_dynamic[name]
                        aliases.pop(name, None)
                    elif name in body_aliases:
                        aliases[name] = body_aliases[name]
                output.append(stmt)
                continue

            if isinstance(stmt, ast.While):
                assigned = _assigned_names(stmt)
                for name in sorted(assigned & set(aliases)):
                    initializer = self._materialize_alias(name, aliases, dynamic)
                    if initializer is not None:
                        output.append(ast.copy_location(initializer, stmt))
                stmt.test = _rewrite_loads(stmt.test, aliases)
                stmt.body, _, _ = self._lower_block(
                    stmt.body,
                    aliases,
                    dynamic,
                    tracks_result=tracks_result,
                )
                stmt.orelse, _, _ = self._lower_block(
                    stmt.orelse,
                    aliases,
                    dynamic,
                    tracks_result=tracks_result,
                )
                if not stmt.body:
                    stmt.body = [ast.copy_location(ast.Pass(), stmt)]
                output.append(stmt)
                continue

            if isinstance(stmt, ast.If):
                assigned = _assigned_names(stmt)
                had_else = bool(stmt.orelse)
                for name in sorted(assigned & set(aliases)):
                    initializer = self._materialize_alias(name, aliases, dynamic)
                    if initializer is not None:
                        output.append(ast.copy_location(initializer, stmt))
                stmt.test = _rewrite_loads(stmt.test, aliases)
                stmt.body, body_aliases, body_dynamic = self._lower_block(
                    stmt.body,
                    aliases,
                    dynamic,
                    tracks_result=tracks_result,
                )
                stmt.orelse, else_aliases, else_dynamic = self._lower_block(
                    stmt.orelse,
                    aliases,
                    dynamic,
                    tracks_result=tracks_result,
                )
                if not stmt.body:
                    stmt.body = [ast.copy_location(ast.Pass(), stmt)]
                if had_else and not stmt.orelse:
                    stmt.orelse = [ast.copy_location(ast.Pass(), stmt)]
                if had_else:
                    for name in sorted(assigned):
                        body_name = body_dynamic.get(name) or body_aliases.get(name)
                        else_name = else_dynamic.get(name) or else_aliases.get(name)
                        if body_name is None or else_name is None:
                            continue
                        stable_name = dynamic.setdefault(
                            name, self._state_name(name)
                        )
                        aliases.pop(name, None)
                        if body_name != stable_name:
                            stmt.body.append(
                                ast.copy_location(
                                    ast.Assign(
                                        targets=[
                                            ast.Name(
                                                id=stable_name, ctx=ast.Store()
                                            )
                                        ],
                                        value=ast.Name(
                                            id=body_name, ctx=ast.Load()
                                        ),
                                    ),
                                    stmt,
                                )
                            )
                        if else_name != stable_name:
                            stmt.orelse.append(
                                ast.copy_location(
                                    ast.Assign(
                                        targets=[
                                            ast.Name(
                                                id=stable_name, ctx=ast.Store()
                                            )
                                        ],
                                        value=ast.Name(
                                            id=else_name, ctx=ast.Load()
                                        ),
                                    ),
                                    stmt,
                                )
                            )
                output.append(stmt)
                continue

            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parameter_aliases = {
                    argument.arg: argument.arg
                    for argument in (
                        list(stmt.args.posonlyargs)
                        + list(stmt.args.args)
                        + list(stmt.args.kwonlyargs)
                    )
                    if argument.arg in _GEOMETRY_PARAMETER_HINTS
                }
                declares_global_result = any(
                    isinstance(item, ast.Global)
                    and self.result_name in item.names
                    for item in stmt.body
                )
                initial_dynamic = (
                    {self.result_name: self.global_result_state}
                    if declares_global_result
                    and self.global_result_state is not None
                    else {}
                )
                if declares_global_result and self.global_result_state is not None:
                    for item in stmt.body:
                        if (
                            isinstance(item, ast.Global)
                            and self.result_name in item.names
                            and self.global_result_state not in item.names
                        ):
                            item.names.append(self.global_result_state)
                stmt.body, _, _ = self._lower_block(
                    stmt.body,
                    parameter_aliases,
                    initial_dynamic,
                    tracks_result=declares_global_result,
                )
                if not stmt.body:
                    stmt.body = [ast.copy_location(ast.Pass(), stmt)]
                output.append(stmt)
                continue

            if isinstance(stmt, ast.ClassDef):
                stmt.body, _, _ = self._lower_block(
                    stmt.body, {}, {}, tracks_result=False
                )
                if not stmt.body:
                    stmt.body = [ast.copy_location(ast.Pass(), stmt)]
                output.append(stmt)
                continue

            if isinstance(stmt, ast.Return) and stmt.value is not None:
                prefix, value, _ = self._lower_expr(
                    stmt.value, aliases, dynamic
                )
                output.extend(prefix)
                stmt.value = value
                output.append(stmt)
                continue

            if isinstance(stmt, ast.Expr):
                prefix, value, is_geometry = self._lower_expr(
                    stmt.value, aliases, dynamic
                )
                output.extend(prefix)
                if not is_geometry:
                    stmt.value = value
                    output.append(stmt)
                continue

            output.append(_rewrite_loads(stmt, aliases))

        return output, aliases, dynamic

    def transform(self, tree: ast.Module) -> ast.Module:
        tree.body, aliases, dynamic = self._lower_block(
            tree.body, {}, {}, tracks_result=True
        )
        terminal: ast.stmt | None = None
        if self.result_name in aliases:
            final_name = aliases[self.result_name]
            if not (
                final_name.startswith("wp") and final_name[2:].isdigit()
            ):
                wp_name = self._new_wp()
                tree.body.append(
                    ast.Assign(
                        targets=[ast.Name(id=wp_name, ctx=ast.Store())],
                        value=ast.Name(id=final_name, ctx=ast.Load()),
                    )
                )
                final_name = wp_name
            terminal = ast.Assign(
                targets=[ast.Name(id=self.result_name, ctx=ast.Store())],
                value=ast.Name(id=final_name, ctx=ast.Load()),
            )
        elif self.result_name in dynamic:
            wp_name = self._new_wp()
            tree.body.append(
                ast.Assign(
                    targets=[ast.Name(id=wp_name, ctx=ast.Store())],
                    value=ast.Name(id=dynamic[self.result_name], ctx=ast.Load()),
                )
            )
            terminal = ast.Assign(
                targets=[ast.Name(id=self.result_name, ctx=ast.Store())],
                value=ast.Name(id=wp_name, ctx=ast.Load()),
            )
        elif self.global_result_state is not None:
            wp_name = self._new_wp()
            tree.body.append(
                ast.Assign(
                    targets=[ast.Name(id=wp_name, ctx=ast.Store())],
                    value=ast.Name(
                        id=self.global_result_state, ctx=ast.Load()
                    ),
                )
            )
            terminal = ast.Assign(
                targets=[ast.Name(id=self.result_name, ctx=ast.Store())],
                value=ast.Name(id=wp_name, ctx=ast.Load()),
            )
        if terminal is not None:
            tree.body.append(terminal)
        elif not self.saw_result:
            self.report.structural_errors.append(
                f"No assignment to terminal name {self.result_name!r}"
            )
        return tree


# ---------------------------------------------------------------------------
# Structural checks and public API
# ---------------------------------------------------------------------------


def _is_terminal_assignment(stmt: ast.stmt, result_name: str) -> bool:
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id == result_name
    )


def _nested_cad_call_count(tree: ast.Module) -> int:
    count = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Call)
            and node.func.attr not in _NON_GEOMETRY_METHODS
            and (
                _is_workplane_constructor(node.func.value)
                or node.func.attr in _CADQUERY_GEOMETRY_METHODS
                or (
                    isinstance(node.func.value.func, ast.Attribute)
                    and node.func.value.func.attr
                    in _CADQUERY_GEOMETRY_METHODS
                )
            )
        ):
            count += 1
    return count


def validate_structure(code: str, result_name: str = "result") -> list[str]:
    """Return structural contract violations; an empty list means success."""

    errors: list[str] = []
    try:
        tree = ast.parse(code)
        compile(tree, "<cc-for>", "exec")
    except (SyntaxError, ValueError, TypeError) as error:
        return [f"parse/compile failed: {error}"]

    terminal_indices = [
        index
        for index, stmt in enumerate(tree.body)
        if _is_terminal_assignment(stmt, result_name)
    ]
    if len(terminal_indices) != 1:
        errors.append(
            f"expected one top-level {result_name} assignment, found {len(terminal_indices)}"
        )
    elif terminal_indices[0] != len(tree.body) - 1:
        errors.append(f"{result_name} assignment is not terminal")

    nested_calls = _nested_cad_call_count(tree)
    if nested_calls:
        errors.append(f"found {nested_calls} unlowered fluent CadQuery call(s)")
    return errors


def canonicalize_code(
    source: str, config: CCForConfig | None = None
) -> CanonicalizationResult:
    """Convert one top-level CadQuery script into the CC-for representation."""

    config = config or CCForConfig()
    report = CCForReport(loop_mode=config.loop_mode)
    tree = ast.parse(source)
    compile(tree, "<source>", "exec")

    if config.flatten_namespaces:
        tree = _flatten_simple_namespaces(tree, report)

    if config.loop_mode == "unroll":
        tree = _StaticLoopUnroller(config, report).transform_module(tree)
    else:
        _count_preserved_loops(tree, report)

    if config.remove_dead_none:
        tree.body = _remove_dead_none_from_block(tree.body, report)

    if config.version_reassignments:
        tree = _SSARenamer(tree, report, config.result_name).transform(tree)

    if config.hoist_parameters:
        tree = _hoist_parameter_assignments(tree, report, config.result_name)

    if config.explicit_workplanes:
        tree = _WorkplaneLowerer(tree, report, config.result_name).transform(tree)

    ast.fix_missing_locations(tree)
    code = ast.unparse(tree).rstrip() + "\n"
    report.structural_errors.extend(validate_structure(code, config.result_name))
    return CanonicalizationResult(code=code, report=report)


def decompose_actions(code: str, result_name: str = "result") -> list[CanonicalAction]:
    """Split CC-for code into the Step-ToCAD top-level action representation."""

    tree = ast.parse(code)
    preamble: list[ast.stmt] = []
    statements: list[ast.stmt] = []
    seen_modeling = False

    for stmt in tree.body:
        is_import = isinstance(stmt, (ast.Import, ast.ImportFrom))
        name = _simple_assignment_name(stmt)
        is_parameter = (
            name is not None
            and name != result_name
            and not name.startswith("wp")
            and not seen_modeling
        )
        if not seen_modeling and (is_import or is_parameter):
            preamble.append(stmt)
            continue
        seen_modeling = True
        statements.append(stmt)

    actions: list[CanonicalAction] = []
    if preamble:
        actions.append(
            CanonicalAction(
                kind="preamble",
                code=ast.unparse(ast.Module(body=preamble, type_ignores=[])),
            )
        )

    for stmt in statements:
        stmt_code = ast.unparse(stmt)
        if _is_terminal_assignment(stmt, result_name) and actions:
            previous = actions[-1]
            actions[-1] = CanonicalAction(
                kind=previous.kind, code=f"{previous.code}\n{stmt_code}"
            )
        else:
            actions.append(CanonicalAction(kind="statement", code=stmt_code))
    return actions


__all__ = [
    "CCForConfig",
    "CCForReport",
    "CanonicalAction",
    "CanonicalizationResult",
    "canonicalize_code",
    "decompose_actions",
    "validate_structure",
]
