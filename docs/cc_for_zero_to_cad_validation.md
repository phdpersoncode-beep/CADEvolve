# CC-for validation on Zero-to-CAD

This is a historical validation record. See the
[astra-and-beyond CC-step audit](cc_step_astra_audit.md) for later fixes and
stronger geometry checks. A scalar/topology signature or normalized surface
score alone is not a solid-equivalence test.

Validated on 2026-08-20 against
[`ADSKAILab/Zero-To-CAD-100k`](https://huggingface.co/datasets/ADSKAILab/Zero-To-CAD-100k)
revision `48bcf0a8c6fbfb47a27f5662007c24dff8d754ae`. The runner projects only
`uuid` and `cadquery_file` from remote Parquet shards; image, STL, and STEP
payloads are not materialized.

## Results

| Gate | Programs | Passed | Failed |
|---|---:|---:|---:|
| Current-code structural scan, first 10 train shards | 5,426 | 5,426 | 0 |
| Distributed structural sample, shard offsets 0/40/80/120 | 1,976 | 1,976 | 0 |
| Raw versus CC-for exact solid signature | 32 | 32 | 0 |
| Every emitted CC-for action prefix executes | 32 | 32 | 0 |
| Source/CC-for remain equivalent after numeric binarization | 32 | 32 | 0 |
| Raw versus binarized CC-for normalized surface Chamfer | 32 | 32 | 0 |

The 32-sample normalized symmetric squared Chamfer distribution used 2,048
surface points per solid: mean `0.000153`, p95 `0.000664`, and maximum
`0.000881`. All are far below the existing CADEvolve normalization threshold
of `0.15`.

An earlier exploratory stress scan reached 64,919 rows before its worker was
terminated at a session boundary: 64,865 passed structurally and 54 were
flagged (99.9168%). That predated per-batch checkpoints and did not leave a
durable failure report, so it is scale evidence rather than the reproducible
acceptance result above. The runner now atomically checkpoints every completed
shard batch and records the next shard offset and optional failing source.

## Geometry gates

Exact equivalence compares bounding-box coordinates, volume, area, and solid,
face, edge, and vertex counts with `1e-7` numeric tolerances. This is stricter
than a mesh-only similarity check and caught a loop-carried `None` geometry
accumulator that was not being written back on each iteration.

The quantized gate applies CADEvolve's integer-literal binarization to source
and CC-for. It additionally handles symbolic and `SimpleNamespace` parameters
feeding non-zero operations and preserves small comparison tolerances such as
edge-filter epsilons. It then checks:

1. the binarized source and binarized CC-for solids have the same exact
   signature; and
2. the raw source and binarized CC-for surfaces remain close after independent
   centering and longest-extent normalization.

The three user-supplied programs all pass. Their 2,048-point raw-to-binarized
CC-for Chamfer distances are:

| Program | Chamfer |
|---|---:|
| `mounting_base_with_boss.py` | 0.001218 |
| `sinusoidal_channel_housing.py` | 0.000505 |
| `vented_cap.py` | 0.000238 |

## Format inspection

The generated programs match the CC-for contract used by Step-ToCAD:

- imports followed by a contiguous symbolic parameter/derived-parameter
  preamble;
- intact `for` loops in the default mode, with loop-carried state made explicit;
- one assignment per CadQuery operation using monotonic `wp1`, `wp2`, ... names;
- symbolic dimensions and plane expressions retained in operation arguments;
- exactly one terminal `result = wpN` assignment.

For example, the sinusoidal housing retains its point-generation loop while the
subsequent spline, profile, sweep, and cut become explicit `wpN` steps. The
optional bounded `unroll` mode remains available when strict per-iteration SSA
is required.

## Reproduction

```bash
PYTHONPATH=dataset_utils python \
  dataset_utils/canonicalization_run/zero_to_cad_hf_validation.py \
  --split train \
  --loop-mode preserve \
  --max-shards 10 \
  --shard-batch-size 10 \
  --report /tmp/cc-for-structural.json
```

For the execution and quantization gates, add:

```text
--execution-samples 32
--validate-sample-prefixes
--validate-sample-quantization
--quantized-surface-points 2048
```

CC-for itself still preserves source coordinates and units. Centering, global
scaling, and numeric binarization remain explicit downstream stages rather than
being silently mixed into symbolic canonicalization.
