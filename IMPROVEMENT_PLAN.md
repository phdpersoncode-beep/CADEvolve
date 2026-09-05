# CC-step for Step-ToCAD: implementation and experiment plan

## Recommendation

Use the corrected, **one-pass CC-step converter as the data MVP**, with execution
and geometry filtering enabled. Start a small whole-program supervised fine-tuning
(SFT) pilot before building the full MCTS system. Keep source programs and their
provenance so future representation changes do not require recovering information
from already canonicalized code. Idempotency is not a requirement.

CC-step is a useful executable Python representation: it preserves symbolic
expressions, repeated construction through loops, and much more parameter intent
than a concrete execution trace. It is **not yet a complete definition of a
feature-level MCTS action space**. The most important next decisions are action
boundaries, coordinate normalization, and what the value model predicts.

The code fixes in this PR are implemented. The representation, model, and search
changes below are proposals for review, not claims of completed implementation.
See [the audit report](docs/cc_step_astra_audit.md) for measured results and limits.

## 1. What the current implementation actually represents

The merged PRs #1–8 established namespace flattening, conservative binding
renaming, symbolic parameter placement, explicit CadQuery temporaries, loop/state
handling, and action decomposition. CC-step selects
`CCForConfig(parameter_placement="late")` in the same converter as CC-for.

Its main stages are:

1. Parse Python and expose supported namespace fields.
2. Rename bindings where doing so preserves Python's reaching definitions.
3. Move eligible parameter assignments before their first consuming **top-level
   source statement**. Keep mutations and unsafe assignments in place.
4. Lower recognized CadQuery call chains into `wpN` assignments while keeping
   compound statements, including loops, intact.
5. Expose the terminal `result` and decompose the resulting top-level statements.

This preserves a program, not just a mesh. Two prefixes with the same current
solid can have different workplanes, selectors, pending wires, bindings, tags,
and future behavior. A search state cannot safely be identified by its current
solid alone.

### Strengths and limitations for the paper

| Property | Assessment | Practical consequence |
| --- | --- | --- |
| Named dimensions and derived expressions | Useful and substantially preserved | Supports interpretable edits and a parameter-aware policy |
| Loops | Preserved as compound actions | Repeated features need not become long lists of unrelated coordinates |
| One call per assignment | Easy to inspect and execute | Helpful baseline; increases action depth and can expose incomplete sketches |
| Late parameter placement | Based on source statements, before lowering | Placement and the eventual action splitter have different boundaries |
| General Python helpers/classes | Execution support exceeds flattening/lowering coverage | Successful execution does not imply complete parameter exposure |
| Prefix execution | Checks whether a prefix runs | A valid sketch/setup prefix need not contain a solid yet |
| Scalar/topology signatures | Useful screening | Cannot establish geometric equivalence on their own |
| Parameter perturbation agreement | Checks transform consistency after an edit | Does not measure design quality or robustness by itself |

The supplied ToCAD paper searches over complete-program refinements with a
K-best-first strategy. Step-ToCAD's proposed incremental program construction,
learned continuation value, and MCTS are a substantive change in the state/action
formulation; they should be evaluated separately from improvements to the data
format or policy backbone.

## 2. First representation improvement: keep source-block boundaries

Currently, a source statement such as:

```python
width = 20
depth = 12
height = 4
result = cq.Workplane("XY").box(width, depth, height)
```

can yield one action containing the dimensions and `wp1 = cq.Workplane('XY')`,
followed by an action containing `wp2 = wp1.box(width, depth, height)`. The first
action commits dimensions before the second commits the construction. That is
executable, but it is a weak match for the intended modeling-step abstraction.

**Proposed first action unit:** the lowered statements originating from one
source statement, together with the parameters anchored to that statement.
A loop remains one compound action. Call this a *source block*, not a recovered
CAD feature: one source statement can contain several features, and a feature
can span several source statements.

### How to implement it

- Assign stable source-statement IDs after parsing and carry them through
  namespace rewriting, placement, and lowering. Preserve these as sidecar
  metadata; do not try to infer erased boundaries from final `wpN` names.
- Extend `CanonicalizationResult` with a versioned action manifest containing
  source ID, emitted span, introduced parameters, names read/written, block kind,
  and terminal eligibility. Retain the existing plain Python output.
- Add an explicit decomposition option, e.g. `action_mode="source_block"`.
  Keep `call` mode as an experimental baseline. Require lossless action joining
  and execution agreement with the same canonical program.
- Label prefixes as setup/sketch/solid/terminal where execution can establish
  that distinction. Do not reject all non-solid prefixes. A terminal candidate
  must define valid nonempty solid geometry in `result`.
- Define an explicit end-of-action/end-of-program protocol in the training
  adapter. A generated comment marker can delimit blocks without changing Python
  semantics; validate the whole block with `ast.parse` before executing it.

Later, test feature-boundary heuristics for extrusion, cut, hole, fillet, boolean,
and repeated-feature blocks. Introduce them only after measuring source-block
depth, token length, and rollout validity. Whole helper definitions or loops
should remain atomic until there is a reason to model their interiors.

The policy should generate **the complete action text, including numeric values
and structural choices**, conditioned on the target and prefix. MCTS chooses
among sampled actions; there is no need to restrict the LLM to structure-only
predictions or make numeric search a prerequisite.

## 3. Coordinate handling: settle this before exporting training pairs

There are three different objects in the legacy pipeline:

| Object | Geometry relationship |
| --- | --- |
| Original source and symbolic CC-step | Must agree in the original coordinate frame |
| Legacy standardized execution trace | Intended to preserve geometry before normalization; has tracing limitations |
| Fully centered, scaled, integer-quantized CADEvolve-C | Applies a coordinate transform and potentially loses geometry through quantization |

The repository's `build_cadevolve_c` comparison helper builds the **unscaled
standardization stage**. Its historical name must not be interpreted as testing
the whole center/scale/quantize pipeline. Equality to an incorrect legacy output
is not the target: compare both outputs to the executed source, and diagnose the
stage that disagrees.

Blindly scaling or rounding every numeric AST literal is unsuitable for symbolic
CC-step. Literals can be lengths, counts, angles, ratios, loop indices, and
selector values. Rounding a factor such as `0.45` can destroy a feature even
though parameter names remain present. The signed-shell fix in this PR addresses
one concrete defect; it does not make arbitrary integer quantization safe.

**MVP choice:** retain exact source-unit code. If points are normalized per object,
also provide the center and scale as conditioning metadata, and use the inverse
transform when comparing predictions in source coordinates. Alternatively, use a
single fixed dataset scale for the point encoder while retaining source-unit
labels. Measure the range before choosing that fixed scale.

Do not pair per-object normalized point clouds with unscaled labels while hiding
the normalization transform: identical normalized inputs can then require
different dimensions and translations, making the task ambiguous.

For a later normalized-code representation, implement unit-aware parameter and
operation handling, preserve dimensionless algebra and discrete values, and
validate against the original shape after applying the **same known transform**.
Appending a final transform to otherwise unnormalized code preserves geometry
but does not simplify the numeric predictions; keep that distinction explicit.

## 4. Dataset conversion and policy training

### A. Produce an accepted manifest before a large conversion run

The bundled 5,000 programs are a pinned snapshot, not a random sample of the
entire Zero-to-CAD distribution. Treat their pass rate as a local engineering
measurement, not a population estimate.

For each input, retain its dataset revision/row ID, family or generator ID when
available, source hash, canonical hash, converter commit, dependency versions,
coordinate transform, action-format version, execution status, geometry-check
status, code length, operation mix, and prefix statistics. Keep separate reasons
for source failure, source instability, conversion failure, timeout/crash, and
indeterminate geometric comparison.

Use these gates for the first accepted dataset:

1. Source executes to valid, finite, positive-volume geometry with all output
   bodies represented; repeated execution gives consistent measurements.
2. Canonicalization has no structural errors; code executes and matches the
   source in the same frame using signatures plus occupied-volume comparison.
3. Action assembly is lossless and sampled prefixes execute. Before search
   experiments, test every prefix in the evaluation subset.
4. Exclude unsupported parameter/lowering cases from the first uniform-format
   training set, or label them explicitly as a separate coverage bucket.

The batch entry point now isolates execution jobs and writes completed records
incrementally. Use a fresh output directory per conversion version and train
only from successful manifest rows. A directory glob alone is not a reliable
acceptance list, especially after retries or `keep_failed` runs. Execution
isolation handles faults; it is not a security sandbox for arbitrary Python.

For scale-up, add resumability by matching input hash, configuration, and converter
version. Resume from original sources; do not canonicalize CC-step again.

### B. Prevent train/test leakage before augmentation

Group related generator/template families and geometric near-duplicates before
splitting. Keep all parameter perturbations and alternate representations of one
family in the same split. UUID or random row splits alone do not protect against
synthetic template leakage. Reserve an untouched external evaluation set with
matching units and licensing/provenance records.

Generate meshes and point clouds from the **accepted executable label** and
include all bodies. Record sampling seeds and transforms. Keep complete labels:
do not truncate a program while retaining the full original target geometry.
Filter overlength examples or regenerate the matching target for an explicitly
defined prefix task.

### C. Start with a cadrille SFT adapter

The official cadrille code is a practical starting point: its dataset reader uses
`.pkl` annotations with `mesh_path` and `py_path`, then reads the Python file as
the answer. Its point branch samples 8,192 surface points and selects 256 by
farthest-point sampling. Its training and inference normalization paths assume
particular coordinate conventions; they must be made consistent with CC-step.

Implementation sequence:

1. Write a manifest-to-cadrille exporter and a small dataset adapter for coordinate
   metadata and the prompt/output contract. Pin the model/code revision and
   tokenizer; check actual token-length distributions before setting context size.
2. Run a 1,000-example overfit/smoke pilot, then a roughly 10,000-example validation
   pilot. Use held-out families and report valid-code rate plus geometric accuracy.
   These are proposed scales, not completed training runs or guaranteed budgets.
3. Compare raw source, existing call-level CC-step, and source-block CC-step with
   the same backbone, split, geometry, and training-token budget.
4. Train whole-program SFT first. Add next-block training by sampling canonical
   prefixes and supervising their next complete block, conditioned on the final
   target point cloud. Sample both early and late prefixes.
5. Scale toward 100,000 accepted examples only after the paired-data and held-out
   checks pass. Report acceptance coverage and length filtering alongside scores.

The public cadrille repository provides an SFT starting point; do not assume it
provides a ready-made CC-step MCTS trainer or the paper's entire RL workflow.
CAD-Recode is a useful simpler policy baseline, with a narrower construction
distribution to assess against CC-step's richer operator set.

## 5. Search and value learning

Define a state as the target observation plus the complete generated prefix and
its execution status. For the MVP, replay each child prefix in a fresh worker.
CadQuery workplanes share mutable construction context, so reusing an in-memory
parent object across siblings can contaminate the search. Add execution caching
or snapshots only after branch-independence tests; immutable text and carefully
managed model KV caches can be optimized earlier.

The target-conditioned continuation value should estimate:

`V(prefix, target) = expected terminal reward under the continuation policy`.

A current-shape IoU score is not this value. Later cuts and holes can improve a
prefix whose current solid has poor overlap; setup and sketch prefixes may not
have a solid at all. A code-only robustness/quality model can be a separate
component, but cannot by itself judge whether a prefix will reconstruct a
particular target.

Start with the following progression:

1. Greedy decoding and best-of-N complete programs using the same verifier.
2. Beam or best-first search over blocks using sampled continuation returns.
3. MCTS with sampled actions, a fixed initial policy, and explicit compute limits.
4. Learn values from completed rollouts, including failures, then evaluate whether
   value-guided MCTS improves quality per unit of compute.

For an effectively unbounded text action space, try progressive widening and
deduplicate identical sampled blocks. Specify how sequence probabilities are
used as priors: raw products favor shorter actions, while length normalization
changes their probabilistic interpretation. Compare sampled-uniform priors and
documented probability-based priors instead of silently choosing one.

Use terminal validity as a hard prerequisite. Begin with geometric agreement as
the primary objective and code quality as a tie-breaker within a geometry
tolerance. Later expose an explicit tradeoff or Pareto analysis. Otherwise a
simple, stable but incorrect shape can win by being easy to edit.

Keep these measurements distinct:

- **Transform consistency:** source and canonical code respond equivalently to
  the same perturbation.
- **Edit robustness:** plausible edits retain valid geometry under a specified
  edit distribution.
- **Design usefulness:** parameters affect intended features, constraints and
  relationships are meaningful, and the model supports relevant design changes.

A no-op or irrelevant parameter must not earn robustness credit. Perturb positive
lengths, bounded ratios, angles, and integer counts according to their roles;
report both validity and intended geometric change. The supplied parametric CAD
robustness paper motivates controlled edit sampling, not treating one arbitrary
multiplier as a complete robustness metric.

## 6. Alternatives and ablations worth the effort

| Option | What it tests | Priority |
| --- | --- | --- |
| Current CC-step vs source-block CC-step | Search depth and coupling of dimensions with construction | First representation ablation |
| Raw symbolic source | Whether canonicalization improves learning enough to justify complexity | First SFT baseline |
| Legacy flat trace, before and after explicit normalization | Benefit/cost of removing Python structure and quantizing geometry | Baseline with separately validated labels |
| Semantic parameter names vs deterministic `param_N` names | Language priors versus lexical variation; preserve the same expressions | After the first pilot |
| Best-of-N and beam/best-first search | Whether MCTS adds value at matched compute | Required search baselines |
| MCTS with rollout values vs learned values | Value-model contribution and calibration | Core paper ablation |
| LLM structure plus bounded numeric fitting | Whether geometry error is mostly numeric once structure is correct | Later; not required for the main policy formulation |
| Coarse voxel/SDF rejection before expensive CAD scoring | Evaluation throughput at controlled false-rejection rate | After profiling |
| Typed feature DSL or operation grammar | Stronger validity constraints versus coverage lost on rich CadQuery | Exploration, not an MVP dependency |

Do not collapse states solely by geometry for transpositions: identical solids
can encode different future modeling capabilities and different design intent.

## 7. A defensible experiment sequence

| Milestone | Deliverable | Exit criterion |
| --- | --- | --- |
| Data MVP | Frozen converter, accepted manifest, executable paired labels, point-cloud exporter | No known accepted transform mismatch; every exclusion has a reason |
| Policy pilot | Whole-program SFT and next-block adapter | Held-out validity and geometry beat or clarify the raw-source baseline |
| Search baseline | Best-of-N and beam/best-first curves | Reproducible budgets and consistent terminal scoring |
| MCTS experiment | Fixed-policy MCTS with rollout/learned value ablations | Quality-versus-compute improvement, or a well-supported negative result |
| Design-quality study | Defined edits, independent validation set, human/feature-aware assessment where needed | Quality gain without concealed loss of reconstruction accuracy |

Count **all** generated tokens (including invalid branches), policy/value calls,
CAD executions, and wall time. Report distributions and multiple seeds, not only
the best successful case. Keep policy, data, representation, and search changes
separable. The relevant claim is improved reconstruction/design quality at
matched resources; MCTS is not guaranteed to outperform simpler search.

## Sources and implementation anchors

- Repository history: merged PRs [#1](https://github.com/phdpersoncode-beep/CADEvolve/pull/1)
  through [#8](https://github.com/phdpersoncode-beep/CADEvolve/pull/8), including
  the unresolved annotated-assignment example in the review of
  [#5](https://github.com/phdpersoncode-beep/CADEvolve/pull/5).
- Current code: `dataset_utils/utils/canonicalization/cc_for.py`,
  `cc_for_validation.py`, `standardizing.py`, `binarization.py`, and
  `dataset_utils/canonicalization_run/cc_for_pipeline.py`.
- Supplied papers: *ToCAD: 3D CAD Reverse Engineering through Tree Search
  Algorithms and Multimodal LLMs*; CADEvolve, §3.5; Zero-to-CAD; CAD-Recode;
  TS-LLM; and *A Study on Sampling Strategies to Determine the Variability of
  Parametric History-Based 3D CAD Models* (IMECE2018-87404). These informed the
  recommendations; this document does not establish literature novelty.
- [CadQuery API](https://cadquery.readthedocs.io/en/latest/classreference.html)
  and [construction-context primer](https://cadquery.readthedocs.io/en/latest/primer.html).
- Official [cadrille](https://github.com/col14m/cadrille) implementation, especially
  `dataset.py` (inspected blob `ec3329cde888da6e9ffd2510e7067ffe38c1a1f5`)
  and `train.py` (`991d8b1f029b4e6f48f42ebe0e591045e24b5da1`).
- Official [CAD-Recode](https://github.com/filaPro/cad-recode) implementation.

Prepared for the `astra-and-beyond` audit. No training run or MCTS experiment is
claimed in this PR.
