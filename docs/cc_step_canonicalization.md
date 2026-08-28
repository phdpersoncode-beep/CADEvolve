# CC-step canonicalization

CC-step is the CC-for representation with the parameter block broken up: instead
of one preamble after the imports, each parameter group sits directly above the
modelling step that consumes it. Everything else — loop preservation, named
parameters, reaching-definition renaming, one assignment per CadQuery call — is
byte-for-byte the CC-for pipeline described in
[`cc_for_canonicalization.md`](cc_for_canonicalization.md).

## Why a second placement

A CC-for preamble answers "what are this part's parameters?". It cannot answer
"which parameters belong to this feature?", because the answer was thrown away
when every parameter was lifted to the top. For a model that predicts one step
at a time, that is the more useful grouping: the parameters a step needs arrive
with the step, so a step is a self-contained edit.

```python
# CC-for                              # CC-step
import cadquery as cq                 import cadquery as cq
plate_width = 20.0                    plate_width = 20.0
leg_long = 80.0                       leg_long = 80.0
thickness = 8.0                       thickness = 8.0
leg_short = 60.0                      wp1 = cq.Workplane('XY')
chamfer_dist = 2.0                    wp2 = wp1.box(plate_width, leg_long, thickness)
hole_spacing_long = 20.0              wp3 = wp2.translate((0, leg_long / 2, thickness / 2))
...                                   leg_short = 60.0
wp1 = cq.Workplane('XY')              wp4 = cq.Workplane('XY')
wp2 = wp1.box(...)                    wp5 = wp4.box(leg_short, plate_width, thickness)
wp3 = wp2.translate(...)              wp6 = wp5.translate(...)
wp4 = cq.Workplane('XY')              wp7 = wp3.union(wp6)
...                                   chamfer_dist = 2.0
                                      wp8 = wp7.edges('|Z and <X and >Y')
                                      wp9 = wp8.chamfer(chamfer_dist)
```

Workplane steps stay split one call per line in both. The grouping is about
where the *parameters* go, not about re-chaining the modelling.

## Representation contract

CC-step keeps all four CC-for properties and adds a fifth:

5. **Parameters at their step.** Every named parameter sits in the group
   immediately above the first top-level statement that reads it. Parameters
   sharing a statement form one group, in dependency order.

## Placement rule

Placement runs after reaching-definition renaming and before Workplane lowering,
which is what makes the groups land where they do: a fluent chain is still a
single statement at that point, so a group is emitted above the whole chain
rather than in the middle of the `wp1, wp2, ...` run it lowers to. A `for` loop
is a statement too, so the parameters a loop reads group above the loop.

A statement is a *movable parameter* when it binds one name to a value that
reads no geometry, no control-flow-mutated state, and no name that itself had to
stay put. Both placements classify from this one predicate, so the two
representations can only differ in where a parameter goes, never in which
statements count as parameters. Three cases are worth naming:

- **Namespace reconstruction.** Flattening a `SimpleNamespace` leaves a
  compatibility object that rebuilds the namespace from the fields it just
  exposed. It holds nothing but those parameters, so it counts as one and
  travels with them; treating it as modelling would pin every field it names to
  the top of the program.
- **Reads inside a `def`.** Reader detection descends into function bodies and
  lambdas. Over-reporting a read can only pull an anchor earlier, which is
  safe; missing one would sink a parameter past the step that needs it.
- **`from math import ...`.** `radians(a)` is parameter algebra exactly as
  `math.radians(a)` is. Without this a program that writes the bare form looks
  like it starts modelling at its first derived angle.

CC-step declines to move one class of statement CC-for hoists: a name defined
more than once at top level, such as a loop-carried accumulator's initializer.
Hoisting every definition into one preamble preserves their relative order, so
CC-for gets away with it; sinking them to different steps would not.

Parameters nothing reads have no step to anchor to. They settle against their
own dependencies and readers rather than against the modelling code, and a
define-before-use check over the result falls back to source positions if that
settlement ever produced an order Python would reject.

## Actions

Step-ToCAD action decomposition follows the placement. Under CC-step the
docstring and imports are the only standalone header; every other action is one
parameter group plus the modelling statement it was placed for, with the
terminal `result` alias folded into the preceding action.

```python
decompose_actions(code, parameter_placement="late")
# ["import cadquery as cq",
#  "plate_width = 20.0\nleg_long = 80.0\nthickness = 8.0\nwp1 = cq.Workplane('XY')",
#  "wp2 = wp1.box(plate_width, leg_long, thickness)",
#  ...]
```

## Producing it

```python
from utils.canonicalization.cc_for import CCForConfig, canonicalize_code

canonicalize_code(source, CCForConfig(parameter_placement="late"))   # CC-step
canonicalize_code(source, CCForConfig(parameter_placement="preamble"))  # CC-for
```

In batch, `cfg_cc_step.yaml` sets `parameter_placement: late` and writes to its
own output and log paths:

```bash
cd results
PYTHONPATH=.. python ../canonicalization_run/cc_for_pipeline.py \
  --config ../canonicalization_run/cfg_cc_step.yaml
```

`--parameter-placement` overrides a config for a one-off run.

## Gates

The eval suite runs every CC-for gate against CC-step; only the layout gate
differs, because the two representations are making opposite claims about where
parameters belong:

| placement  | gate                     | passes when                                    |
| ---------- | ------------------------ | ---------------------------------------------- |
| `preamble` | `parameters_hoisted`     | the parameters form one contiguous block        |
| `late`     | `parameters_placed_late` | no parameter could be pushed into a later group |

A parameter is justified where it stands when its own group's statements read
it, or when another settled parameter of the same group does — settled meaning
justified, pinned by control flow, unread, or binding no simple name. Anything
else was defined earlier than it had to be.

The gate is a necessary condition, not a sufficient one. Lowering erases the
source statement boundaries, so a program whose parameters were never split at
all presents as one group in which everything is legitimately justified.
`group_count` is reported so that case is visible, and the corpus-level check
that placement actually splits the preamble lives in `tests/test_cc_step.py`.

```bash
PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_eval \
  --corpus fixtures --parameter-placement late
```

## Representation agreement

`evals/cc_for/representations.py` builds a program four ways — source,
CADEvolve-C, CC-for, CC-step — executes each and compares the solids pairwise.
The two symbolic representations are held to exact equality, since they copy
every argument expression verbatim; CADEvolve-C is held to shape agreement,
since its tracer replays recorded calls, may resolve a selector to a
different-but-equivalent entity, and discretizes parametric curves. Chamfer
bounds are calibrated against each solid's own sampling noise floor, the same
way the single-representation gate is.

The tracer runs in a subprocess: it monkeypatches CadQuery's `Workplane` and
`Shape` classes while recording, and the module executes other programs in the
same interpreter.

```bash
PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_representations \
  --corpus fixtures --report logs/representations.json
```

## Measured results

Zero-to-CAD and CADEvolve-P fixtures (12 programs), full geometry gates:

| corpus                        | placement  | programs | gates passed | exact voxel IoU |
| ----------------------------- | ---------- | -------- | ------------ | --------------- |
| `fixtures`                    | `preamble` | 12       | 12/12        | 1.000           |
| `fixtures`                    | `late`     | 12       | 12/12        | 1.000           |

Parameter retention and design-parameter coverage are 1.000 in both. CC-step
splits those 12 programs into a median of 4 parameter groups (min 2, max 9).

Four-way representation agreement over the same 12 programs: every
representation builds, and all four comparison pairs pass at voxel IoU 1.000.

Edge cases (`evals/cc_for/cases`, 24 programs, full geometry gates) fail the
same four programs under both placements, for the reasons recorded in
[`cc_for_eval_suite.md`](cc_for_eval_suite.md); the 21 that build reach voxel
IoU 1.000 with zero Chamfer distance against the source solid.

Structural sweep over 500 demo programs:

| placement  | passed  | `parameters_hoisted` / `parameters_placed_late` | parameter groups (mean) |
| ---------- | ------- | ----------------------------------------------- | ----------------------- |
| `preamble` | 497/500 | 500/500                                         | 1                       |
| `late`     | 497/500 | 500/500                                         | 4.23                    |

Both placements fail exactly the same three programs — two `parameters_preserved`
and one `chains_lowered` — all pre-existing CC-for defects rather than placement
failures.
