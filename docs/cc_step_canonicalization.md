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
statements count as parameters. Four cases are worth naming:

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
- **A name a second statement binds.** No parameter may cross a statement that
  rebinds it, so a name bound anywhere else at top level is not a parameter for
  either representation. Reaching-definition renaming versions the ordinary
  case, but a name a loop or a branch also writes has to keep one stable
  binding, and both definitions reach this stage. Moving the plain one is then
  unsound in either direction: CC-for hoists an accumulator's initializer above
  the loop that updates it, CC-step sinks it below, and both read a value the
  source never produced. Neither shows up as an error -- the program runs, keeps
  the structural contract, and builds a different part.

Parameters nothing reads have no step to anchor to. They settle against their
own dependencies and readers rather than against the modelling code, and a
define-before-use check over the result falls back to source positions if that
settlement ever produced an order Python would reject.

## Loops

Placement does not touch loops, and that is a claim worth stating rather than
assuming, because CC-step is the representation that moves code *past* them.
Three properties hold, each measured over the 5,000-program Zero-to-CAD snapshot
(1,371 of which contain a `for` loop):

- **The loop survives as one statement.** `for` count and nesting depth are
  identical in source and canonical code for every program. Lowering descends
  into the body and the `else` clause, so a chain inside a loop becomes its own
  run of `wpN` steps, but the `for` itself is never unrolled, split, or merged.
  It is therefore one action: a search picks a whole loop or none of it.
- **A loop's parameters arrive with the loop.** Reader detection walks the whole
  statement, so a name read anywhere inside a loop -- header, body, `else`, or a
  helper the body calls -- anchors the parameter to the loop, and a parameter
  only a *later* loop reads sinks past the earlier one.
- **Loop-carried state stays where it is.** A name the loop rebinds is not a
  parameter at all (see the placement rule), so an accumulator's initializer
  keeps its position relative to the loop that updates it.

The one loop shape the representation does not fully lower is a fluent chain in
the loop *header*, as in `for face in part.faces('>Z').vals():`. The body and
the header's names are rewritten, the iterable expression is not, so
`validate_structure` reports an unlowered call rather than emitting a program
that quietly means something else. It does not occur in the demo corpus;
`evals/cc_for/cases/loop_over_geometry_iterable.py` keeps it visible.

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

A search builds its program by concatenating the actions it chose, so that
concatenation has to be the text the converter emits -- otherwise an exact-match
score or a dedup hash sees two different programs. `join_actions` is that
concatenation:

```python
join_actions(decompose_actions(code, parameter_placement="late")) == code
```

The one place a plain `"\n".join` drifts is a top-level `def` or `class`:
unparsing a module spaces it out from the statement before it, unparsing it
alone as its own action does not. That affects 798 of the 5,000 demo programs
under both placements; `join_actions` restores the separator and reproduces the
canonical text for all 5,000.

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

Zero-to-CAD and CADEvolve-P fixtures (12 programs), full geometry gates
including quantization and parameter perturbation:

| corpus                        | placement  | programs | gates passed | exact voxel IoU |
| ----------------------------- | ---------- | -------- | ------------ | --------------- |
| `fixtures`                    | `preamble` | 12       | 12/12        | 1.000           |
| `fixtures`                    | `late`     | 12       | 12/12        | 1.000           |

Parameter retention and design-parameter coverage are 1.000 in both. CC-step
splits those 12 programs into a median of 4 parameter groups (min 2, max 9).

Four-way representation agreement over the same 12 programs: every
representation builds, and all four comparison pairs pass at voxel IoU 1.000.

Edge cases (`evals/cc_for/cases`, 29 programs, full geometry gates) fail only
the programs recorded in [`cc_for_eval_suite.md`](cc_for_eval_suite.md); the
rest reach voxel IoU 1.000 with zero Chamfer distance against the source solid.

Structural sweep over the full 5,000-program Zero-to-CAD snapshot, both
placements, 2,082 loops preserved and 110,696 Workplane steps emitted:

| gate                                            | `preamble` | `late`    |
| ----------------------------------------------- | ---------- | --------- |
| `converts` / `structure`                        | 5000/5000  | 5000/5000 |
| `loops_preserved` / `literals_stable`           | 5000/5000  | 5000/5000 |
| `loop_bindings_preserved`                       | 5000/5000  | 5000/5000 |
| `actions_reassemble`                            | 5000/5000  | 5000/5000 |
| `parameters_hoisted` / `parameters_placed_late` | 4993/5000  | 4998/5000 |
| `parameters_preserved`                          | 4990/5000  | 4990/5000 |
| `chains_lowered`                                | 4995/5000  | 4995/5000 |

CC-step splits the snapshot into a mean of 4.00 parameter groups per program
(median 4, max 12) against CC-for's single preamble. Parameter retention is
0.9999 in both; design-parameter coverage is 0.995.

Every failing set above is the same set of programs the converter failed before
the placement rule changed — measured by running the same scan on the merge base:
`parameters_preserved` 10, `parameters_hoisted` 7, `chains_lowered` 5,
`parameters_placed_late` 2, identical program-for-program under both placements.
They are the pre-existing CC-for defects recorded in
[`cc_for_eval_suite.md`](cc_for_eval_suite.md). Both `parameters_placed_late`
failures are the conservative-pinning limit described above: a value derived from
a name control flow rebinds cannot move, so it stays where the source put it and
the layout gate reports it.
