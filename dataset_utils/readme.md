## Folder structure overview

- `run.sh` — runs the full pipeline end-to-end.
- `canonicalization_run/` — sampling + canonicalization entrypoints and configs.
- `rotation_run/` — rotation augmentation entrypoints and configs.
- `utils/` — shared pipeline utilities (canonicalization + rotation helpers).
- `data/` — input databases (e.g., `database_toy_example.json`).

## Minimal run

### 1) Set the input database

Edit `./canonicalization_run/cfg_sampling.yaml` and set `code_db` to the database of parts you want to process.

A toy example database is provided at `./data/database_toy_example.json`.

### 2) Run the pipeline

From the project root:

    chmod +x ./run.sh
    ./run.sh

You can also run it from anywhere:

    bash /path/to/CADEvolve/run.sh

## Outputs

Running `./run.sh` creates a new folder `./results/` in the project root (this folder is generated locally and is not tracked in the repository).

Usable results are stored in:
- `results/rotated/`
- `results/rotated_stl/`

Important: the rotation stage expects a flat folder at `results/canonicalized_flat/`.

## Parameter-preserving CC-for output

The legacy `standardize -> center -> scale -> binarize` route creates CADEvolve-C by
executing programs and recording concrete calls. Do not use that route when named
parameter relationships are required.

`canonicalization_run/cc_for_pipeline.py` converts sampled CADEvolve-P scripts (or a
folder of readable Zero-to-CAD scripts) directly into the Step-ToCAD CC-for format:

```bash
cd results
PYTHONPATH=.. python ../canonicalization_run/cc_for_pipeline.py \
  --config ../canonicalization_run/cfg_cc_for.yaml
```

Edit `cfg_cc_for.yaml` to point `root_dir` at the source scripts. The default
`loop_mode: preserve` keeps pattern loops intact; `loop_mode: unroll` expands bounded
static loops and produces strict reaching-definition names. Geometry round-trip and
executable-prefix validation are enabled by default. Per-file records are written to
`logs/cc_for.jsonl`, with aggregate counts in `logs/cc_for.summary.json`.

The converter intentionally preserves source coordinates and scale. Unit-aware
centering/scaling is a separate stage; running the legacy tracer or scalar rewriter
after CC-for would destroy parameters or incorrectly scale counts and angles.

### CC-step: parameters at their step

`parameter_placement` selects where the named parameters go. The default
`preamble` collects them into one block after the imports (CC-for); `late` puts
each group directly above the modelling step that reads it (CC-step), so a step
arrives with the dimensions it needs. Everything else — loops, named parameters,
one assignment per CadQuery call — is identical, and both build the same solid.

```bash
cd results
PYTHONPATH=.. python ../canonicalization_run/cc_for_pipeline.py \
  --config ../canonicalization_run/cfg_cc_step.yaml
```

`--parameter-placement late` overrides a config for a one-off run. The
representation contract, the placement rule and measured results are in
[`docs/cc_step_canonicalization.md`](../docs/cc_step_canonicalization.md).

### Zero-to-CAD 100K validation

The Hugging Face stress runner reads only the `uuid` and `cadquery_file` Parquet
columns. Images, STL, and STEP fields are not materialized:

```bash
PYTHONPATH=. python canonicalization_run/zero_to_cad_hf_validation.py \
  --split train \
  --loop-mode preserve \
  --execution-samples 100 \
  --validate-sample-prefixes \
  --validate-sample-quantization \
  --report logs/zero_to_cad_100k_preserve.json
```

Use `--max-rows` or `--max-shards` for a smoke run. The JSON report includes the
dataset revision, strict structural pass rate, aggregate transformations and
warnings, bounded failure examples, exact geometry round-trip results, and optional
normalized post-binarization surface Chamfer. Reports are atomically checkpointed
after every shard batch. Use `--skip-shards` to resume from the reported
`next_shard_offset`, and `--failure-source-dir` to retain structural outliers for
regression work.

The reproducible Zero-to-CAD validation results and metric definitions are in
[`docs/cc_for_zero_to_cad_validation.md`](../docs/cc_for_zero_to_cad_validation.md).

### Offline Zero-to-CAD 5K demo snapshot

`demo_data/zero_to_cad_5k/raw_sources.tar.gz` contains 5,000 raw CadQuery
programs for repeatable offline structural tests. See its
[`README.md`](../demo_data/zero_to_cad_5k/README.md) for extraction, validation,
and provenance-reproduction commands. The archive avoids adding 5,000 loose Git
objects while still extracting to ordinary `<uuid>.py` source files.

## Notes on config paths

- `run.sh` executes stages with the working directory set to `./results/`, so paths inside YAML configs are resolved relative to `./results/`.
- Inputs stored in `./data/...` are typically referenced as `../data/...` inside YAML.
- Outputs should be written as relative paths like `sampled/...`, `canonicalized/...` so they land inside `./results/`.

## Output folder structure overview

The following folders are created after running the pipeline (not included in the repository by default):

- `results/` — all outputs are written here.
  - `results/sampled/` — sampling outputs (generated scripts + logs, depending on config).
  - `results/canonicalized/` — canonicalization stage outputs.
    - `results/canonicalized/binarized/` — final canonicalized scripts (nested structure).
  - `results/canonicalized_flat/` — **flat** scripts required by rotation stage (no subfolders).
  - `results/rotated/` — rotated scripts (main usable output).
  - `results/rotated_stl/` — STL exports for rotated scripts (main usable output).
  - `results/logs/` — pipeline logs.

## CC-for evaluation suite

`evals/cc_for/` holds an independent evaluation of this converter: it executes
the original and canonical programs and compares the resulting solids
(topology, mass properties, voxel IoU, surface Chamfer) as well as the code
itself (parameter retention, preamble placement, loop preservation, literal
drift). See [`evals/cc_for/README.md`](../evals/cc_for/README.md) for the gates
and [`docs/cc_for_eval_suite.md`](../docs/cc_for_eval_suite.md) for results.

```bash
PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_eval --corpus cases
PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_eval --corpus demo --no-geometry
PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_eval --corpus cases \
  --parameter-placement late
```

`evals/cc_for/run_representations.py` answers a different question: do
CADEvolve-C, CC-for and CC-step describe the same part? It builds all three plus
the source, executes each, and compares the solids pairwise.

```bash
PYTHONPATH=.:dataset_utils python -m evals.cc_for.run_representations --corpus fixtures
```
