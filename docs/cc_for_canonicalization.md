# CC-for canonicalization

This document defines the custom CadQuery representation used by Step-ToCAD.
The same converter also emits **CC-step**, which differs only in where the named
parameters go: see [`cc_step_canonicalization.md`](cc_step_canonicalization.md).
The converter is intentionally AST-based: executing a program and tracing CadQuery
calls, as the legacy CADEvolve standardizer does, destroys symbolic parameter
relationships and can freeze parameter-dependent plane orientations into literals.

## Representation contract

CC-for has four required properties:

1. **Named parameters.** Independent and derived parameters remain symbolic and are
   collected into a contiguous preamble immediately after imports whenever moving
   them is semantics-preserving. CC-step keeps the same parameters but places each
   group above the step that reads it instead.
2. **Loop-aware SSA.** A name with multiple definitions is versioned (`x_1`, `x_2`,
   ...). In loop-preserving mode, loop-carried state is the sole deliberate SSA
   exception because Python needs one stable name across iterations.
3. **Intact loops by default.** `for` loops preserve pattern semantics and compactness.
   A bounded, static unrolling mode exists for diagnostics and strict-SSA datasets.
4. **Explicit CadQuery steps.** Fluent Workplane chains are lowered to one assignment
   per call (`wp1`, `wp2`, ...), while symbolic arguments remain symbolic.

There is exactly one terminal `result = ...` assignment. For Step-ToCAD action
decomposition, the parameter preamble is one action, every other top-level AST
statement is one action, and the terminal `result` alias is folded into the preceding
modeling action. Under CC-step only the docstring and imports form a standalone
header; each parameter group joins the modeling action it was placed for.

## Pipeline

The structural pipeline runs in this order:

1. Parse and compile-check the input module.
2. Flatten top-level `SimpleNamespace` parameter containers when safe, rewriting
   `p.length` to a collision-free named parameter such as `length`. A symbolic
   compatibility namespace is retained for class-based programs that pass it through
   `self.m`, so flattening cannot create a missing runtime binding.
3. Optionally unroll statically bounded loops over `range`, literal containers,
   `enumerate`, or `zip`. Dynamic and oversized loops stay intact with a warning.
4. Delete a `name = None` initializer only when the same straight-line block
   overwrites it before any read.
5. Apply deterministic reaching-definition renaming. Every definition of a repeated
   name receives a suffix, including the first definition.
6. Hoist pure, CAD-independent assignments into a dependency-safe parameter preamble.
   Loop-local and loop-mutated values stay with their loop in preserve mode. Under
   `parameter_placement: late` this step sinks each parameter to the latest position
   that still precedes every read of it, producing CC-step. It runs before lowering,
   so a group lands above a whole fluent chain rather than inside the run of `wpN`
   steps that chain becomes.
7. Lower CadQuery Workplane, Sketch, and shape-factory chains without executing them,
   including chains inside helper functions and class methods. `cq.Plane`,
   `cq.Vector`, trigonometry, and other parameter expressions remain symbolic.
8. Reparse, compile, run structural checks, and optionally execute the original and
   canonical programs for geometric round-trip and prefix validation.

The converter operates on sampled CADEvolve-P scripts or readable Zero-to-CAD
programs. It must not use CADEvolve-C as its source: that representation has already
discarded the named bindings needed for parameter robustness metrics.

## Reassignment example

Given two statically unrolled iterations that both define `i`, `x`, and `y`, strict
mode emits distinct definitions:

```python
i_1 = 1
y_1 = i_1 * dy
x_1 = cell_size + x_off

i_2 = 2
y_2 = i_2 * dy
x_2 = r_mid * math.cos(phi)
```

All later uses point to the reaching version. In preserve mode, two separate loops
that both use `i` receive separate targets (`i_1`, `i_2`), but a variable intentionally
updated across iterations remains a stable loop-carried name.

## Validation gates

Each converted file records:

- parse/compile success;
- unresolved or deliberately preserved loops;
- names that remain loop-carried;
- parameter and Workplane counts;
- repeated-definition violations outside preserved control flow;
- optional original/canonical shape invariants (bounding box, volume, area, and
  topology counts);
- optional executable-prefix success;
- optional source/CC-for equivalence after CADEvolve-style numeric binarization;
- optional normalized symmetric surface Chamfer between the raw and binarized
  CC-for solids;
- optional perturbation consistency for named parameters.

The perturbation check is important for plane construction: changing an angle in the
canonical preamble must change any dependent `xDir` or `normal` expression in the same
way as changing it in the source program.

## Geometry normalization boundary

CC-for conversion preserves the source geometry exactly. The legacy dynamic
`standardize -> center -> scale -> binarize` path is not run after CC-for: tracing
removes symbols, and its scaler multiplies dimensionless values such as counts and
angles. Centering/scaling must therefore remain a separately validated, unit-aware
stage rather than being silently mixed into structural canonicalization.
