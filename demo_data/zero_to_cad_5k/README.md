# Zero-to-CAD 5K raw-program snapshot

This folder contains 5,000 unmodified `cadquery_file` programs from the
`train` split of [`ADSKAILab/Zero-To-CAD-100k`][dataset].
The exact upstream revision and archive checksum are recorded in
`manifest.json`. The upstream dataset is Apache-2.0 licensed.

The checked-in `validation.json` records the original structural CC-for scan:
5,000 passed, 0 failed, 110,696 explicit workplane actions were emitted, and
2,082 loops were preserved. Geometry execution remains a sampled gate because
executing thousands of OpenCascade programs is intentionally much slower.
For the later CC-step execution audit, stronger geometry checks, and explicit
failure categories, see [the astra-and-beyond report](../../docs/cc_step_astra_audit.md).

The programs are stored as one deterministic `raw_sources.tar.gz` archive
instead of 5,000 loose Git objects. Extract them with:

```bash
mkdir -p /tmp/zero_to_cad_5k
tar -xzf demo_data/zero_to_cad_5k/raw_sources.tar.gz \
  -C /tmp/zero_to_cad_5k
```

This creates `/tmp/zero_to_cad_5k/raw/<uuid>.py`. To validate every source
against the CC-for structural contract without contacting Hugging Face:

```bash
PYTHONPATH=dataset_utils python \
  dataset_utils/canonicalization_run/zero_to_cad_snapshot.py validate \
  --archive demo_data/zero_to_cad_5k/raw_sources.tar.gz \
  --manifest demo_data/zero_to_cad_5k/manifest.json \
  --report /tmp/zero_to_cad_5k_validation.json
```

To reproduce the snapshot from the pinned dataset selection:

```bash
PYTHONPATH=dataset_utils python \
  dataset_utils/canonicalization_run/zero_to_cad_snapshot.py export \
  --output demo_data/zero_to_cad_5k/raw_sources.tar.gz \
  --manifest demo_data/zero_to_cad_5k/manifest.json \
  --split train --offset 0 --count 5000
```

[dataset]: https://huggingface.co/datasets/ADSKAILab/Zero-To-CAD-100k
