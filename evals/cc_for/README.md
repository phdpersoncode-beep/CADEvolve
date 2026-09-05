# CC-for evaluation suite

The [astra-and-beyond audit](../../docs/cc_step_astra_audit.md) adds
`audit_corpus.py` for disposable-process execution, per-input deadlines, streaming
JSONL evidence, and optional same-frame CAD Boolean comparison. Use it for large
geometry runs where native crashes must not invalidate neighboring inputs.

An independent check on the canonicalizer
(`dataset_utils/utils/canonicalization/cc_for.py`): does the canonical program
mean the same thing as the source, and does it still *say* it symbolically?

The converter emits two representations that differ only in where the named
parameters go — CC-for puts them in one preamble after the imports, CC-step puts
each group directly above the step that reads it — and every gate here runs
against both. Select one with `--parameter-placement`; see
[`docs/cc_step_canonicalization.md`](../../docs/cc_step_canonicalization.md).

The suite is deliberately not built on
`dataset_utils/utils/canonicalization/cc_for_validation.py`. Metrics written
alongside a transform tend to agree with it; these are written against the
contract instead.

## What it measures

Three artefacts are compared for every program: the **original** CadQuery code,
the **canonical** code, and the **solids** both produce.

| Gate | Question |
|---|---|
| `converts` | Does the converter run at all? |
| `structure` | One terminal `result`, no unlowered fluent modelling chains. |
| `idempotent` | Diagnostic only: is canonical output a fixed point? `f(f(x)) == f(x)`. The one-pass contract does not require this and the runner ignores it by default. |
| `parameters_preserved` | Does every source parameter survive, un-inlined? |
| `parameters_hoisted` | *(`--parameter-placement preamble`)* Do parameters form one contiguous preamble after imports? |
| `parameters_placed_late` | *(`--parameter-placement late`)* Could any parameter have been pushed into a later group? |
| `loops_preserved` | Same number of `for` loops in preserve mode. |
| `literals_stable` | No numeric or string literal invented or dropped. |
| `chains_lowered` | Every modelling call is its own `wpN` assignment. |
| `loop_bindings_preserved` | Does every loop still write the accumulator it wrote in the source? |
| `actions_reassemble` | Does `join_actions(decompose_actions(code))` give the canonical program back? |
| `source_executes` / `canonical_executes` | Both build a valid shape. |
| `source_deterministic` | Does the source program build the same solid twice? |
| `topology_identical` | Solids, shells, faces, wires, edges, vertices, face/edge type histograms, volume, area, bounding box, centre of mass. |
| `shape_identical` | Voxel IoU ≈ 1 and Chamfer ≈ 0 after normalization. Skippable with `--no-mesh-comparison`, which is what makes a corpus-scale geometry run affordable. |
| `prefixes_execute` | Every emitted action prefix runs on its own. |
| `quantization_commutes` | Binarizing source and canonical gives the same solid. |
| `quantized_shape_close` | The canonical program loses no more shape to binarization than the source does. |
| `parameter_perturbation` | Scaling a named parameter changes both programs identically. |

Two measurements are reported rather than gated, because the contract permits a
range of behaviour:

- **`parameter_retention`** — share of source parameters still named in the
  canonical program, allowing for SSA versioning and namespace flattening.
- **`preamble_coverage`** — share of a program's *numeric design parameters*
  bound by name in a canonical parameter block. This is the claim the change
  exists for: a dimension buried as a literal inside a constructor call is not
  editable downstream, even though the program still builds correctly. The
  question is the same under either placement; only where the block sits differs.
- **`parameter_groups`** — how many groups CC-step split the preamble into.
  Reported under `--parameter-placement late` only. One group is a legitimate
  outcome for a program with a single feature, which is why it is reported
  rather than gated; see the note on what the layout gate cannot see in
  [`docs/cc_step_canonicalization.md`](../../docs/cc_step_canonicalization.md).

## Running the geometry gates over the whole corpus

The mesh comparison is the expensive gate; topology counts, mass properties and
the bounding box already separate two different solids. Dropping it turns the
5,000-program geometry run from impractical into an overnight-free afternoon:

```bash
PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_eval \
    --corpus demo --parameter-placement late \
    --no-mesh-comparison --no-quantization --no-perturbations --no-prefixes \
    --workers 4 --report /tmp/demo-geometry.json
```

Three of the defects recorded in
[`../../docs/cc_for_eval_suite.md`](../../docs/cc_for_eval_suite.md) were found
this way and by nothing else: each one produces a canonical program that runs,
passes every structural gate, and builds a different part.

## Attribution

Two gate families deliberately measure the *difference* between the source and
the canonical program under the same downstream transform, rather than the
transform's own damage:

- `quantization_commutes` reports *not applicable* when the legacy binarizer
  cannot build the binarized **source** either. Both sides failing identically
  is a binarizer limitation, not a canonicalization defect.
- `quantized_shape_close` compares the canonical program's post-binarization IoU
  against the source's own post-binarization IoU (`binarizer_baseline`) and
  fails only on a shortfall. The absolute binarizer damage is still reported.

Similarly, `parameter_perturbation` counts a case as passing when *both*
programs fail to build under the same perturbation: not every parameter can be
scaled freely, and a symmetric failure is still agreement.

A third case is the corpus itself. Some programs are not reproducible —
OpenCascade can take a different path on a fillet or a boolean between runs and
land on a different topology, with face counts differing by a factor of five in
the extreme. Comparing a canonical program against a baseline that does not
agree with itself measures nothing, so `source_deterministic` re-executes the
source and, when it disagrees, the comparison gates are reported as not
applicable. The check catches *consistently* non-reproducible programs; an
intermittently flaky one can still slip through a single re-run. Pass
`--ignore-gate source_deterministic` to exclude corpus quality from the verdict,
or `--no-determinism-check` to skip the extra build entirely.

## Running it

```bash
pip install -r requirements-cc-for.txt

# 22 hand-written edge cases, every gate
PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_eval --corpus cases

# a sample of the checked-in Zero-to-CAD 5K snapshot
PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_eval \
    --corpus demo --limit 250 --workers 3 --report /tmp/demo.json

# the whole snapshot, AST gates only (no CadQuery required)
PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_eval \
    --corpus demo --no-geometry --workers 4

# the repo's own fixtures, or any directory of programs
PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_eval --corpus fixtures
PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_eval --corpus path/to/dir

# the same gates against CC-step instead of CC-for
PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_eval \
    --corpus fixtures --parameter-placement late
```

Do CADEvolve-C, CC-for and CC-step actually describe the same part? That is a
separate runner, because it compares the representations against each other
rather than scoring one against its source:

```bash
PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_representations \
    --corpus fixtures --report /tmp/representations.json

# symbolic representations only; skips the tracer subprocess entirely
PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_representations \
    --corpus demo --limit 200 --skip-cadevolve-c
```

As pytest assertions:

```bash
PYTHONPATH=.:dataset_utils python -m pytest tests/test_cc_for_eval_suite.py -q
PYTHONPATH=.:dataset_utils python -m pytest tests/test_cc_step.py -q
```

### Useful flags

| Flag | Why |
|---|---|
| `--execution-timeout` | A CAD program can loop forever, and a perturbed one more easily still. Each execution runs under a `SIGALRM` budget. |
| `--no-determinism-check` | Skip re-executing each source program. Saves one build per program; costs the ability to tell a flaky corpus program from a converter defect. |
| `--max-prefix-checks` | Prefix replay is O(n²) CAD calls; long programs are subsampled, first and last always included. |
| `--tasks-per-child` | Programs per worker before the pool is recreated. OpenCascade holds memory across programs. |
| `--voxel-resolution`, `--surface-points` | Similarity cost/precision. |
| `--ignore-gate NAME` | Stop a gate deciding the verdict while it keeps running and reporting. Repeatable; `idempotent` is included by default because it is diagnostic rather than contractual. |
| `--fail-under` | Exit non-zero below this pass rate, for CI. |
| `--parameter-placement` | `preamble` evaluates CC-for, `late` evaluates CC-step. Selects which layout gate decides the verdict. |

## Modules

| File | Role |
|---|---|
| `geometry.py` | Topology, mass properties, face/edge type histograms. Unwraps *every* body on the stack — `Workplane.val()` returns only the first solid, so comparing it cannot see a dropped body. |
| `similarity.py` | Mesh-based voxel IoU and surface Chamfer after independent centre/longest-extent normalization. |
| `code_metrics.py` | Parameter retention, block coverage, preamble contiguity and CC-step group layout, loop counts, literal drift, chain depth. |
| `harness.py` | Runs all gates for one program; never raises for a program-level problem. |
| `representations.py` | Builds a program as source, CADEvolve-C, CC-for and CC-step, then compares the solids pairwise. Runs the legacy tracer in a subprocess: it monkeypatches CadQuery while recording. |
| `run_eval.py` | Process-isolated corpus runner with per-batch pools and timeouts. |
| `run_representations.py` | The same runner shape for the four-way comparison. |
| `cases/` | Hand-written CadQuery edge cases; each docstring states what it stresses. |

### On the voxeliser

`similarity.voxelize` rasterises triangles onto the (x, y) grid columns, records
the z of every ray/triangle crossing, and fills each column by parity. It was
checked against OpenCascade's `BRepClass3d_SolidClassifier` on primitives,
filleted and bored solids, shells and multi-body compounds: exact agreement on
planar and cylindrical parts, and ≥0.98 IoU on curved and thin-walled ones,
where the difference is mesh tessellation rather than the fill rule. It is one
to two orders of magnitude faster than classifying every voxel.
