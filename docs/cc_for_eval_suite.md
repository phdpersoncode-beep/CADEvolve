# CC-for evaluation results

Independent evaluation of the CC-for canonicalization pipeline
(PR [#1](https://github.com/phdpersoncode-beep/CADEvolve/pull/1),
branch `agent/cc-for-canonicalization`) using the suite in
[`evals/cc_for/`](../evals/cc_for/README.md).

Corpus: the 5,000 checked-in Zero-to-CAD programs in
`demo_data/zero_to_cad_5k/`, plus the hand-written edge cases in
`evals/cc_for/cases/`. Environment: CadQuery 2.5.2 / OCP 7.7.2, Python 3.11.

A later pass over CC-step found five more, recorded in
[a second section below](#a-second-pass-what-the-cc-step-verification-found);
those are fixed, with the two that a structural scan can see now gated.

## Headline

The canonicalizer does what the PR claims, on essentially the whole corpus:
named parameters survive, loops stay intact, and the canonical program builds
the *same solid* as the source. Five defects sit underneath that result, three
of which produce a wrong program at runtime while passing every check the PR
currently runs.

## What holds

Full corpus, 5,000 programs, AST gates:

| Gate | Result |
|---|---|
| `converts` | 5000 / 5000 |
| `structure` (one terminal `result`, no unlowered modelling chains) | 5000 / 5000 |
| `loops_preserved` (same `for` count in preserve mode) | 5000 / 5000 |
| `literals_stable` (no numeric or string literal invented or dropped) | 5000 / 5000 |
| `parameters_preserved` | 4990 / 5000 |
| `parameters_hoisted` (contiguous preamble) | 4991 / 5000 |
| `chains_lowered` | 4995 / 5000 |

Parameter retention: mean **0.9999**, median 1.0, min 0.833.
Preamble coverage (share of a program's numeric design parameters bound by name
in the canonical preamble, n=4,711 programs that have any): mean **0.9951**,
median 1.0. 2,082 loops preserved, 110,696 explicit workplane steps emitted.

Geometry, 250-program sample with every gate (24 edge cases give the same
picture, and are covered under *Defects* below):

| Gate | Result |
|---|---|
| `topology_identical` | 241 / 241 |
| `shape_identical` | 241 / 241 |
| `prefixes_execute` | 241 / 241 |
| `quantization_commutes` | 241 / 241 |
| `quantized_shape_close` | 241 / 241 |
| `parameter_perturbation` | 241 / 241 |

Nine programs are excluded from the comparison gates: eight fail to build *as
source programs* and one is not reproducible (see the reproducibility section).
Exact voxel IoU: mean **0.99998**, median 1.0. Exact Chamfer: median **0**,
p95 0.0007.

- **Exact solid equivalence.** Source and canonical agree on solid, shell, face,
  wire, edge and vertex counts, on face and edge *type* histograms, and on
  volume, area, bounding box and centre of mass.
- **Action prefixes execute.** Every emitted `wpN` prefix runs standalone.
- **Binarization commutes.** Binarizing the source and binarizing the canonical
  program give the same solid wherever the binarizer can build either.
- **Parameter perturbation is consistent.** Scaling a named parameter by 1.37 in
  the source and in the canonical program produces the same solid — the check
  that the symbolic relationships are genuinely intact rather than frozen.

Of the 250 programs, **240 pass** with the corpus-wide `idempotent` defect
baselined out. Every one of the 10 failures is individually accounted for:

| Cause | Programs |
|---|---:|
| Corpus program fails to build as *source* (fails identically as canonical) | 5 |
| Corpus program is not reproducible | 1 |
| **Defect 1** — `AugAssign` → `NameError` | 1 |
| **Defect 3** — silent geometry loss | 1 |
| **Defect 5** — hand-rolled `Measures` class not flattened | 1 |
| **Defect 5** — same program, second gate | (1) |

Three programs fail for canonicalization reasons; the other seven are corpus
quality. Defect 2 does not appear in this particular sample because the program
carrying it (`019172f8`) failed to build as a source on this run — it has a
deterministic minimal reproduction above.

## Defects

### 1. `AugAssign` breaks the canonical program — `NameError`

The SSA renamer versions a name's plain-assignment definition but leaves every
`AugAssign` occurrence on the original name. The canonical program then
references a name that was never bound.

```python
# source
total = 1.0
total += 2.0
result = cq.Workplane('XY').box(total, 2, 2)

# canonical  ->  NameError: name 'total' is not defined
total_1 = 1.0
total += 2.0
result = ... .box(total_1, 2, 2)
```

Two things are wrong: the `AugAssign` target is unversioned, and the consumer
reads `total_1` (the pre-increment value) rather than the accumulated one, so
fixing only the `NameError` would still change the geometry.

Corpus impact: 46/5,000 programs use `AugAssign` on a plain name; **4 of them
produce a canonical program that raises `NameError`** while passing the
structural scan. `evals/cc_for/cases/augmented_assignment_offsets.py` covers it.

### 2. A statement reading a loop-carried `result` is not rewritten

When `result` is rebound inside control flow it becomes `result_state`, but a
bare statement that reads `result` before the terminal alias is left untouched.

```python
# source
result = cq.Workplane('XY').box(size, size, size)
for selector in ('>X', '<X'):
    result = result.faces(selector).workplane().hole(4.0)
assert result.solids().size() == 1
result = result.edges('|Z').fillet(1.0)

# canonical  ->  NameError: name 'result' is not defined
result_state = wp2
for selector in ('>X', '<X'):
    ...
    result_state = wp5
assert result.solids().size() == 1     # <- not rewritten
...
result = wp8                            # <- bound only here
```

Corpus impact: 2/5,000. Small, but `assert result.solids().size() == 1` is a
common self-check idiom in generated CAD programs, so the rate depends on the
generator rather than on anything intrinsic.

### 3. Silent geometry loss from copy-propagating a `self` attribute

The worst of the five, because nothing fails. An attribute alias is replaced by
the constructor argument it was initialised from, ignoring that an intervening
method call reassigns it:

```python
class Part:
    def __init__(self, workplane, size):
        self.wp = workplane
        self.size = size
        self.build()             # reassigns self.wp
        self.model = self.wp     # canonical rewrites this to `workplane`

    def build(self):
        self.wp = self.wp.box(self.size, self.size, self.size)

result = Part(cq.Workplane('XY'), 10.0).model
```

The canonical program runs cleanly, reports no structural error, and produces an
**empty** `Workplane` where the source produces a 1000 mm³ box.

Corpus impact: 6/5,000 programs match the store → call → read-back pattern, and
**all 6 lose all their geometry**. This is the case a structural scan can never
find, and an execute-only check ("does it run?") cannot find either — it needs
the solids compared.

### 4. Canonicalization is not idempotent

`canonicalize(canonicalize(x)) != canonicalize(x)` for **every one of the 5,000
programs**. The `wpN` counter skips any name already bound in the input, so
re-running the converter on its own output renumbers every step:

```
wp1 … wp24   ->   wp25 … wp48
```

The collision-avoidance is right in principle — a source program may have its
own `wp3` — but it does not distinguish a user's `wp3` from a `wp3` the
converter itself is about to replace. Consequences: canonical form is not a
fixed point, so content-hash dedup and resumed or re-run conversions are not
reproducible, and a re-processed shard trains on a different token distribution
than a first-pass shard.

### 5. Coverage gaps

Neither of these corrupts a program; both leave the representation short of the
contract's intent.

- **Chains inside `try`/`except` are not lowered.** The lowering pass handles
  `If`, `For` and `While` bodies but not `Try`, so a guarded modelling step stays
  a nested fluent chain. The converter self-reports the violation
  (`found 2 unlowered fluent CadQuery call(s)`), so with `keep_failed: false`
  the program is dropped rather than silently mis-converted — a coverage gap,
  not a correctness bug. `evals/cc_for/cases/try_except_fallback.py` covers it.
- **User-class parameter containers are not flattened.** 182/5,000 programs
  (3.64%) define their own `Measures`-style class instead of using
  `SimpleNamespace`. Their design parameters stay as literals inside the
  constructor call and never reach the preamble: of the 90 such programs that
  have numeric design parameters, mean preamble coverage is **0.78**, and the
  23 programs corpus-wide with *zero* coverage are almost entirely this idiom.
  A dimension buried in a constructor call is not editable by a downstream model
  even though the program builds correctly.
- **Nested builder-method chains stay nested.** 5/5,000 keep chains such as
  `self.build_base().cut_holes().add_ribs().apply_fillet()`, because the
  converter cannot tell that a user method returns geometry. Each is a modelling
  step that the action decomposition does not see.

## A second pass: what the CC-step verification found

Five further defects, found by running CC-step over the same 5,000 programs, by
hand-written probes around loops and parameter placement, and by promoting the
source-versus-canonical solid comparison from a sample to the whole corpus. All
five predate CC-step and affect both representations; all five are fixed on this
branch, with regression cases in `evals/cc_for/cases/`.

### 6. A `while` loop's geometry accumulator is never written back

`for` pre-declares a loop-carried geometry name so the body's assignment is
emitted; `while` did not, so the assignment was recorded as an alias and dropped.

```python
# source
pegs = None
index = 0
while index < peg_count:
    peg = cq.Workplane("XY").center(...).circle(peg_radius).extrude(peg_height)
    pegs = peg if pegs is None else pegs.union(peg)
    index = index + 1
result = base.union(pegs)

# canonical, before the fix -- note that nothing assigns `pegs`
while index < peg_count:
    ...
    if pegs is None:
        wp8 = wp6
    else:
        wp7 = pegs.union(wp6)
        wp8 = wp7
    index = index + 1
wp9 = wp2.union(pegs)          # pegs is still None: every peg is gone
```

The canonical program runs, keeps the structural contract, and builds the base
with no pegs on it. 2/5,000 corpus programs match the pattern, and both lose
geometry (one drops an entire internal lattice: volume 17,663 against 19,313).
`evals/cc_for/cases/while_carried_union_accumulator.py` covers it, and a
corpus-wide check — *is every name the source binds inside a loop body still
bound inside a loop body?* — now reports 0/5,000.

### 7. A parameter moved across the statement that rebinds it

A name a loop or a branch also writes keeps one stable binding through renaming,
so both definitions reach the placement stage. The movability predicate looked at
what a statement *read*, not at what it *bound*, so the plain definition counted
as a parameter and moved — CC-for above the loop, CC-step below it.

```python
tallest = 0.5
for rib_height in rib_heights:
    if rib_height > tall_threshold:
        tallest = rib_height
result = plate.faces(">Z").workplane().circle(6.0).extrude(tallest)
```

Under CC-step the initializer sank below the loop and the boss came out 0.5 tall
instead of 6.0; the mirror program, with the reset *after* the loop, catches
CC-for the same way. Excluding rebound names fixes both and changes 72/5,000
CC-for and 61/5,000 CC-step outputs, all of them corrections. Covered by
`conditional_loop_accumulator.py` and `late_reinitialized_accumulator.py`.

### 8. Actions did not reassemble into the canonical program

Concatenating the emitted actions is how a tree search builds its program, so the
concatenation has to be the text the converter emits. Unparsing a module spaces a
top-level `def` or `class` out from the statement before it; unparsing that
statement alone as its own action does not, so a plain `"\n".join` of the actions
differed from the canonical code for **798/5,000** programs under both
placements. `join_actions` restores the separator and now reproduces the
canonical text for 5,000/5,000.

### 9. A `for` target kept an alias from an earlier loop

Found by running the source-versus-canonical solid comparison over all 5,000
programs rather than a sample. Inside a `def` the renamer leaves locals alone,
so one name can be both an assignment target in one loop and the iteration
variable of the next. The lowerer aliased the first to its `wpN` and never
cleared it:

```python
# source                              # canonical, before the fix
for x, y in rib_coords:               for x, y in rib_coords:
    rib = ...build...                     wp17 = ...build...
    ribs.append(rib)                      ribs.append(wp17)
for rib in ribs:                      for rib in ribs:
    base = base.union(rib)                wp18 = base.union(wp17)
```

The union runs four times over the last rib. One Zero-to-CAD program hit it and
came out 432 mm³ light with 15 faces missing, while running cleanly and keeping
the structural contract. A `for` target now clears its alias, after the iterable
is rewritten, since the iterable is evaluated before the first bind.
`evals/cc_for/cases/loop_target_reuses_geometry_name.py` covers it.

### 10. An attribute target rebound inside a branch kept its alias

`_assigned_names` collects `Name` stores, so a branch that writes `self.model`
looks to it like a branch that writes nothing, and the alias recorded before the
branch survived it:

```python
if m.include_gusset:                  wp24 = wp9.union(wp23)
    self.model = self.model.union(gusset)
self.model = self.model.edges(...)    wp25 = wp9.edges(...)   # gusset gone
                   .chamfer(...)
```

The chamfer reads the model as it stood before the gusset was unioned in, so the
gusset is discarded. One corpus program lost 408 mm³ and 5 faces this way.
`_assigned_state_keys` now collects the dotted targets a block writes, and `for`,
`while` and `if` drop those aliases before lowering the block.
`evals/cc_for/cases/attribute_state_across_branch.py` covers it.

Between them the two changes alter 15 of the 5,000 canonical programs, and the
three that were building wrong parts now round-trip exactly under both
placements. Note that defect 3 above is *not* subsumed: it propagates across an
opaque call such as `self.build()` rather than across a branch, which gives this
analysis nothing to key on.

### Two limits left in place

- **A fluent chain in a `for` header is not lowered.** The loop body and the
  header's names are rewritten, `part.faces('>Z').vals()` in
  `for face in part.faces('>Z').vals():` is not. Self-reported by
  `validate_structure`, like the `try`/`except` gap above, and absent from the
  demo corpus. `evals/cc_for/cases/loop_over_geometry_iterable.py` keeps it
  visible.
- **A value derived from a rebound name cannot sink either.** Excluding rebound
  names also excludes everything derived from them, so under CC-step such a chain
  stays where the source put it and `parameters_placed_late` reports it — 2/5,000
  programs. Sinking it safely needs a per-parameter bound on how far it may
  travel rather than a yes/no predicate, which is a change to the representation
  rather than a fix to it.

## Why the existing validation did not catch these

The PR reports 5,000/5,000 passing and 32/32 on the geometry gates. Both numbers
are reproducible — but the 5,000 is a **structure-only** scan. `validate_structure`
parses, compiles, and checks for one terminal `result` and no unlowered chains.
Defects 1, 2 and 3 all satisfy every one of those conditions: they are runtime
failures or silent geometry loss.

The geometry gate that would catch them ran on 32 programs. At the measured rates
(4, 2 and 6 programs in 5,000), the chance that a 32-program sample contains any
of them is roughly 7%.

Two further observations about the existing gates:

- **The quantized-geometry threshold cannot fail.** `validate_quantized_geometry`
  accepts a normalized squared Chamfer up to `0.15`. On
  `evals/cc_for/cases/derived_parameter_chain.py` the legacy binarizer rounds
  `pad_length = lever * 0.45` to `lever * 0`, producing a solid with **negative
  volume (-2601)**, 3 bodies instead of 2, a Z extent of 92 against 26, and a
  voxel IoU against the source of **0.0094**. Its squared Chamfer is 0.067, so
  the gate passes it. A threshold that accepts a 0.9% IoU is not a geometry gate.
  (This is the legacy binarizer's damage, not CC-for's — see below — but the gate
  is what is supposed to detect it.)
- **A binarizer failure is reported as a canonicalization failure.** When the
  legacy binarizer cannot build the *source* either, the gate still reports
  `"source and CC-for diverged after binarization"`. They did not diverge; both
  failed identically. Three of the 22 edge cases hit this.

## A 1% reproducibility floor sits under any geometry comparison

Chasing the last unattributed gate failure turned up a property of the corpus
rather than of the converter: **some Zero-to-CAD programs do not build the same
solid twice.** Re-executing 499 buildable programs and comparing the two results
against each other:

| | Programs | Share |
|---|---:|---:|
| Reproducible | 494 | 99.00% |
| **Not reproducible** | **5** | **1.00%** |

The disagreements are not float noise:

| Program | First run vs second run |
|---|---|
| `0e1aca6c` | 1 shell vs 2; 36 faces vs 53 |
| `1fe43d5b` | 31 faces vs 25; 45 wires vs 52 |
| `194558e9` | 37 faces vs 35 |
| `0d4c40a1` | centre of mass Y flips sign, `-1.9083` vs `+1.9083` |
| `1821d3ee` | centre of mass moves by 0.075 in X |

`08cde529`, which prompted the investigation, swings between 156 and 28 faces
after binarization. OpenCascade takes a different path on a fillet or a boolean
between runs and lands on a different topology.

Two consequences:

1. **Any single-run geometry comparison on this corpus carries ~1% irreducible
   noise.** A small number of failures has to be checked for reproducibility
   before it can be attributed to canonicalization. The suite's
   `source_deterministic` gate does this: it re-executes the source and reports
   the comparison gates as not applicable when the baseline disagrees with
   itself. It catches consistently non-reproducible programs; an intermittently
   flaky one can still pass a single re-run.
2. **The existing 32-program geometry sample had roughly a 27% chance of
   containing at least one such program** (1 − 0.99³²), where it would have
   shown a spurious divergence — or, had one been excluded by hand, hidden a
   real one.

This is a corpus/toolchain property, not a defect in the PR. It is recorded here
because it bounds what any geometry gate over this dataset can claim.

## Binarizer damage is not canonicalization damage

The suite measures the two separately. Comparing each program's post-binarization
IoU against the source's own post-binarization IoU
(`binarizer_baseline`), the two distributions are **identical to six decimal
places** across the edge cases and the corpus sample. CC-for adds no measurable
geometric damage under the downstream binarization stage; all of the divergence
belongs to the legacy binarizer, which is severe on parts with sub-unit or
non-integer derived factors (`* 0.45` → `* 0`).

This supports the PR's own position that centering, scaling and binarization
should stay separate, explicitly validated stages rather than being folded into
canonicalization.

## Suggested priorities

1. **Defect 3** (silent geometry loss) — wrong data with no signal at all.
2. **Defect 1** (`AugAssign`) — smallest fix, clearest reproduction, and it
   affects the loop-accumulator programs the PR is specifically about.
3. **Defect 4** (idempotence) — one-line intent change (do not reserve `wpN`
   names the lowering pass is itself replacing), but it affects every output.
4. **Defect 2** (`result` read before the alias).
5. **Coverage gaps** — `try`/`except` lowering, then user-class containers,
   then a fluent chain in a `for` header.

Defects 6-8 from the second pass are already fixed on the CC-step verification
branch; the two structural ones now run corpus-wide as `loop_bindings_preserved`
and `actions_reassemble`, which is the shape the recommendation below asks for:
a defect the 5,000-program scan can see should be measured there rather than on
a sample.

Independently of the fixes: promote the geometry gate from a 32-program sample
to a corpus-scale run, since all three correctness defects are invisible to the
structural scan that does run at 5,000 — and pair it with a reproducibility
check, because roughly 1% of the corpus cannot serve as its own baseline.

## Reproducing

```bash
pip install -r requirements-cc-for.txt

PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_eval --corpus cases
PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_eval \
    --corpus demo --no-geometry --workers 4
PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_eval \
    --corpus demo --limit 250 --workers 3
PYTHONPATH=.:dataset_utils python -m pytest tests/test_cc_for_eval_suite.py -q
```
