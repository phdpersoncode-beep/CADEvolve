# CC-for evaluation suite

An independent check on the CC-for canonicalizer
(`dataset_utils/utils/canonicalization/cc_for.py`): does the canonical program
mean the same thing as the source, and does it still *say* it symbolically?

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
| `idempotent` | Is canonical output a fixed point? `f(f(x)) == f(x)`. |
| `parameters_preserved` | Does every source parameter survive, un-inlined? |
| `parameters_hoisted` | Do parameters form one contiguous preamble after imports? |
| `loops_preserved` | Same number of `for` loops in preserve mode. |
| `literals_stable` | No numeric or string literal invented or dropped. |
| `chains_lowered` | Every modelling call is its own `wpN` assignment. |
| `source_executes` / `canonical_executes` | Both build a valid shape. |
| `topology_identical` | Solids, shells, faces, wires, edges, vertices, face/edge type histograms, volume, area, bounding box, centre of mass. |
| `shape_identical` | Voxel IoU ≈ 1 and Chamfer ≈ 0 after normalization. |
| `prefixes_execute` | Every emitted action prefix runs on its own. |
| `quantization_commutes` | Binarizing source and canonical gives the same solid. |
| `quantized_shape_close` | The canonical program loses no more shape to binarization than the source does. |
| `parameter_perturbation` | Scaling a named parameter changes both programs identically. |

Two measurements are reported rather than gated, because the contract permits a
range of behaviour:

- **`parameter_retention`** — share of source parameters still named in the
  canonical program, allowing for SSA versioning and namespace flattening.
- **`preamble_coverage`** — share of a program's *numeric design parameters*
  bound by name in the canonical preamble. This is the claim the change exists
  for: a dimension buried as a literal inside a constructor call is not editable
  downstream, even though the program still builds correctly.

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
```

As pytest assertions:

```bash
PYTHONPATH=.:dataset_utils python -m pytest tests/test_cc_for_eval_suite.py -q
```

### Useful flags

| Flag | Why |
|---|---|
| `--execution-timeout` | A CAD program can loop forever, and a perturbed one more easily still. Each execution runs under a `SIGALRM` budget. |
| `--max-prefix-checks` | Prefix replay is O(n²) CAD calls; long programs are subsampled, first and last always included. |
| `--tasks-per-child` | Programs per worker before the pool is recreated. OpenCascade holds memory across programs. |
| `--voxel-resolution`, `--surface-points` | Similarity cost/precision. |
| `--fail-under` | Exit non-zero below this pass rate, for CI. |

## Modules

| File | Role |
|---|---|
| `geometry.py` | Topology, mass properties, face/edge type histograms. Unwraps *every* body on the stack — `Workplane.val()` returns only the first solid, so comparing it cannot see a dropped body. |
| `similarity.py` | Mesh-based voxel IoU and surface Chamfer after independent centre/longest-extent normalization. |
| `code_metrics.py` | Parameter retention, preamble coverage and contiguity, loop counts, literal drift, chain depth. |
| `harness.py` | Runs all gates for one program; never raises for a program-level problem. |
| `run_eval.py` | Process-isolated corpus runner with per-batch pools and timeouts. |
| `cases/` | Hand-written CadQuery edge cases; each docstring states what it stresses. |

### On the voxeliser

`similarity.voxelize` rasterises triangles onto the (x, y) grid columns, records
the z of every ray/triangle crossing, and fills each column by parity. It was
checked against OpenCascade's `BRepClass3d_SolidClassifier` on primitives,
filleted and bored solids, shells and multi-body compounds: exact agreement on
planar and cylindrical parts, and ≥0.98 IoU on curved and thin-walled ones,
where the difference is mesh tessellation rather than the fill rule. It is one
to two orders of magnitude faster than classifying every voxel.
