# CC-step correctness audit: astra-and-beyond

This audit starts from fork `main` at
`1ee0d0d01cb02454e3cca0849f83cfe6b5b1d843` (merged PR #8). It follows the
Step-ToCAD objective of learning a point-cloud-conditioned CadQuery policy for
incremental search while retaining symbolic parameters and loops. The converter
is used once on original sources; idempotency is deliberately not an acceptance
gate.

**Assessment:** the corrected implementation is a useful filtered data MVP.
Remaining source/kernel failures and incomplete parameter/lowering coverage
should be excluded from initial experiments. MCTS action boundaries and training
integration remain design work, described in
[IMPROVEMENT_PLAN.md](../IMPROVEMENT_PLAN.md).

## What was reviewed

- Merged PRs #1–8, their descriptions and available review comments, and the
  implementation/evaluation documentation at the baseline commit.
- The supplied Zero-to-CAD and CADEvolve examples; relevant sections of the
  supplied ToCAD, CADEvolve, Zero-to-CAD, CAD-Recode, TS-LLM, and parametric-model
  robustness papers; official CadQuery and cadrille implementation sources.
- The checked-in 5,000-source archive, all 12 repository fixtures, and the
  adversarial cases accumulated during the earlier PRs.

The earlier agent work resolved substantial problems: augmented assignment,
loop-carried geometry, opaque builder mutations, branch/while aliases,
zero-iteration geometry, and action joining. This audit retains those fixes and
adds regression coverage for the remaining failures below.

## Correctness fixes

| Defect | Observable consequence before the fix | Change |
| --- | --- | --- |
| Scalar bindings assigned inside a zero-iteration `for` | A later use can read a version that was never assigned | Preserve runtime bindings for names assigned by the loop, including its target |
| Reassigned globals read by a helper | A helper reads an undefined/stale source name after SSA renaming | Keep bindings read from nested scopes stable, including defaults, decorators, and annotations |
| Annotated `result` assignment in a branch/loop (PR #5 review example) | `result_state` can remain undefined | Route simple-name annotated writes through ordinary assignment/state handling; preserve annotation evaluation and avoid evaluating attribute targets twice |
| Validation unwraps only `Workplane.val()` | Dropping another body can pass the old round-trip check | Compare every shape on the result stack |
| Signature-only equivalence | Opposite pairs of holes can have equal volume, bounds, area, and entity counts | Add same-frame occupied-volume comparison using two CAD differences |
| Independently normalized representation comparison | A translated solid can receive a perfect normalized mesh score | Require same-frame solid agreement as well as existing metrics |
| Negative shell/extrusion binarization | `shell(-1)` becomes outward `shell(1)`; sub-unit negative extrusion can reverse direction | Retain the sign when clamping signed nonzero arguments, including keyword arguments |
| Permissive lossy-geometry gate | Invalid quantized solids and heavily damaged derived dimensions can pass a surface-only threshold | Require valid solids and normalized volume IoU in addition to the existing surface diagnostic |
| Legacy Workplane constructor tracing | Keyword origin is dropped; positional origin is mistaken for `obj` | Serialize constructor origin and recorded object arguments correctly |
| Legacy tracer chooses its last Workplane | An unused later object or a Shape-valued result changes the returned part | Pass and retain the actual terminal `result` reference |
| Source stdout corrupts tracer JSON | A valid program containing `print()` fails tracing | Route executed source stdout to stderr |
| Fake unregistered execution module | Valid dataclasses with postponed annotations fail in the evaluator | Execute in a temporarily registered module namespace |
| Shared workers / Python-only deadlines | One native crash can invalidate unrelated jobs; native hangs can stall conversion | Disposable processes, parent-enforced deadlines, descendant cleanup on POSIX, and per-completion JSONL records |

Unrecorded legacy Shape/Workplane references now fail explicitly rather than
emitting code with undefined `wpNone`-style references. This makes unsupported
traces visible; it does not claim to add support for arbitrary callbacks or
unrecorded objects.

The initial targeted regression run reproduced **9 failures out of 14 cases**.
An independent run against the untouched baseline also reproduced all five
legacy-tracer regressions: dropped keyword origin, broken positional origin,
wrong terminal object, lost Shape-valued translation, and stdout corruption.
The corresponding corrected regressions pass.

## What “same solid” means here

Three comparisons must remain separate:

1. **Source versus CC-step:** preserve geometry and coordinates. Compare valid
   nonempty solids, all output bodies, bounds, mass properties, entity counts,
   and (when enabled) occupied-volume differences. Do not normalize the two
   outputs independently for this correctness decision.
2. **Source versus legacy standardized trace:** the same geometric intention,
   but the legacy tracer has unsupported constructs and may approximate curves.
   Both are checked against the source so a legacy bug is not treated as the
   reference geometry that CC-step must reproduce.
3. **Fully normalized/quantized CADEvolve-C:** centering and scaling change the
   coordinate frame; integer quantization can change the shape itself. The
   historical `build_cadevolve_c` helper only builds the unscaled standardization
   stage. The four-way representation tests are not an audit of the complete
   normalization/quantization pipeline.

For the solid-difference gate, equivalence requires

`volume(A \\ B) + volume(B \\ A) <= max(1e-7, 1e-7 * max(volume(A), volume(B)))`.

The two residual volumes are measured in the original coordinate system and
sum absolute solid volumes. Invalid/nonfinite results and Boolean exceptions
fail closed. This is a tolerance-based occupied-volume test, not proof of
identical B-Rep parameterization or of all topological properties. A CAD kernel
can fail to compare otherwise valid solids; that result is *indeterminate*, not
evidence that the converter changed the part.

The production CC-step config enables the Boolean gate. The old signature-only
API remains available for inexpensive screening and compatibility.

## Corpus and results

Archive: `demo_data/zero_to_cad_5k/raw_sources.tar.gz`, SHA-256
`319c840e39a6c756afef578efad22683df98b0e82f258bd9164f563ab54fa048`.
The pinned upstream revision is
`48bcf0a8c6fbfb47a27f5662007c24dff8d754ae`. This is the repository's selected
snapshot, not a representative random draw from the entire upstream dataset.

Environment: CPython 3.11.15, CadQuery 2.5.2, cadquery-ocp 7.7.2. Exact installed
versions and machine-readable evidence are in
[`reports/cc_step_astra/`](../reports/cc_step_astra/).
The final implementation commit is
`fb9d0f7ad205dbbfa1de6f2168180ca37e2c14be`.

### Full 5,000-source execution screen

The first pass executed source, CC-step, and source again in one disposable
worker per input. It compared topology/type histograms, volume, area, bounds,
extents, and center of mass at relative/absolute tolerances of `1e-7`. It did
**not** run Boolean differences for all 5,000 inputs.

After targeted fresh-process rechecks, the conservative classification is:

| Classification | Programs |
| --- | ---: |
| Signature match; no instability observed in these runs | 4,935 |
| Source instability observed | 36 |
| Source execution failure | 17 |
| Source produces invalid geometry | 3 |
| Native worker crash / wall-clock timeout | 9 |
| Unexplained stable-source conversion mismatch | 0 |
| Total | 5,000 |

“No instability observed” is a bounded observation, not a determinism guarantee.
The 4,935 are **screening candidates**, not 4,935 Boolean-certified training
labels. Run the production geometry gate before accepting them.

The raw first-pass counts are also retained: 4,932 matches, 22 source errors,
8 signature mismatches, 2 canonical execution errors, 6 invalid sources,
21 observed unstable sources, and 9 worker failures. Three initial source errors
were artifacts of the unregistered execution namespace and passed after that
harness fix. The other changes in classification are supported by source-only
measurement changes or source failures during repeated execution; a successful
rerun alone does not erase an earlier unexplained failure.

The first pass overlapped final fixes. Canonical output hashes were recomputed
from all 5,000 sources, and the eight changed outputs were executed again. The
three namespace failures and every first-pass nonmatch were also rechecked.
The evidence includes fresh-process repetitions of ambiguous cases. Two cases
needed twelve additional repetitions each, and a final small loft-volume
discrepancy needed eight; their source instability was observed, not assumed.

Examples include `c304814f-...` (bracket topology/holes differ across unchanged
source runs), `930c70e9-...` (intermittent source CAD failure), and `f5fa66f4-...`
(small repeat-dependent loft/shell volume differences). No source geometry was
arbitrarily edited to force those inputs through the gate.

### Structural coverage

| CC-step gate | Passes |
| --- | ---: |
| Conversion, structure, loop preservation, literal stability, loop bindings, action reassembly | 5,000 / 5,000 each |
| Parameter retention | 4,994 / 5,000 |
| Late parameter placement | 4,997 / 5,000 |
| Complete recognized-chain lowering | 4,995 / 5,000 |
| All non-idempotency structural/coverage gates | 4,986 / 5,000 |

The remaining 14 coverage cases are distinct from geometric corruption. The
uniform-format pilot should filter them or label their limitations explicitly.
Intersecting the signature and coverage screens leaves 4,921 candidates for
the production geometry/prefix acceptance checks.
All 2,082 `for` loops are preserved. The idempotency diagnostic still reports
failures and is intentionally ignored.

### Stronger sample and regressions

The final sample is 128 archive programs selected with seed 42, plus the
12 curated fixtures. In the archive sample, **125 pass source/CC-step Boolean
comparison**, two have indeterminate Boolean results, and one exceeds the
180-second whole-job deadline. Of those 125, **114 legacy standardized traces
also match the source**. The other 11 legacy traces cannot be replayed because
of callbacks or unrecorded Shape/object references. There are no observed
successful comparisons that show a stable-source CC-step solid mismatch in
this sample; indeterminate and unsupported comparisons are not counted as passes.

The full archive sample took about 383 seconds with two workers. The 5,000-source
signature screen took about 2,025 seconds with six workers; shared-machine load
and CAD-kernel behavior affect both timings. The final full test suite passed
**213 tests and 322 subtests**; the focused regression run passed **31 tests**.
The full suite took about 651 seconds. Logs are preserved in the report folder.

All nine Zero-to-CAD fixtures pass the production batch pipeline with source
round-trip, Boolean geometry, and prefix execution enabled. Of the 12 combined
fixtures, 11 pass Boolean comparison and their legacy trace also matches. The
remaining `looped_bumps.py` gives a null-shape exception during CAD difference;
its signature agrees, but its strict geometric comparison is indeterminate.

The adversarial case suite has one remaining structural failure,
`loop_over_geometry_iterable.py`: a chain in a loop iterable is not fully
lowered. It is reported rather than silently changed. Quantization and
perturbation are separate diagnostics and were disabled for that execution
screen.

Integer quantization is not a condition for accepting raw-unit CC-step.
`vented_cap.py` now correctly fails the lossy diagnostic because its quantized
solid is invalid; `derived_parameter_chain.py` is rejected for excessive
geometric damage. These are not raw-source/CC-step mismatches.

## Reproduce and use the MVP

From the repository root, using Python 3.11:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-cc-for.txt
PYTHONPATH=.:dataset_utils .venv/bin/python -m pytest -q --disable-warnings

PYTHONPATH=.:dataset_utils .venv/bin/python -m evals.cc_for.run_eval \
  --corpus demo --parameter-placement late --no-geometry --workers 4 \
  --report /tmp/cc-step-structure.json

PYTHONPATH=.:dataset_utils .venv/bin/python -m evals.cc_for.audit_corpus \
  --corpus demo --workers 6 --timeout 90 --report /tmp/cc-step-5k.jsonl

PYTHONPATH=.:dataset_utils .venv/bin/python -m evals.cc_for.audit_corpus \
  --corpus demo --limit 128 --seed 42 --workers 2 --timeout 180 \
  --boolean --trace --report /tmp/cc-step-trace-128.jsonl
```

The structural command exits nonzero for the documented coverage gaps.
`audit_corpus` is an evidence-producing audit command; its exit code is not a
dataset acceptance gate. Inspect statuses and its JSON summary. `--repeats N`
runs each selected input in N fresh processes, and a directory path can select
specific problem sources. A worker deadline covers the entire job, including
source repetitions and tracing, rather than each individual CAD execution.

For conversion, extract the checked-in archive and copy
`dataset_utils/canonicalization_run/cfg_cc_step.yaml` to a local run config.
Set `root_dir` to the extracted `raw` directory, `out_dir` to a **fresh** directory,
and `report_path` to a new JSONL path. Keep execution, prefix, and Boolean checks
enabled and `keep_failed: false`. Then run:

```bash
PYTHONPATH=dataset_utils .venv/bin/python \
  dataset_utils/canonicalization_run/cc_for_pipeline.py --config /path/to/run.yaml
```

Use successful pipeline records intersected with the source-stability and
coverage screens to select training pairs. The pipeline itself compares one
source/canonical pair and does not implement a repeated-source stability gate.
Do not train by globbing every file in a reused output directory. Preserve raw
sources, converter version, coordinate metadata, and rejected records. Generate
the target mesh/point cloud from the accepted executable label, including every
body.

Increasing a timeout may distinguish a slow case from a hang; it does not make
a previous crash a correctness pass. POSIX process-group cleanup also stops
descendant interpreters spawned by tracing jobs. Other platforms retain direct
worker termination but do not receive that process-group guarantee.

## Scope left for the next implementation

Source-block action metadata, coordinate-aware training adapters, resumable
dataset export, a learned policy/value model, and MCTS are intentionally left
for the improvement plan. No model training, GPU experiment, or complete
Zero-to-CAD dataset conversion is claimed in this PR. The aim here is a tested
converter and honest acceptance criteria on which those experiments can build.
