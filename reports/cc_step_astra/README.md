# CC-step audit evidence

See [the readable audit](../../docs/cc_step_astra_audit.md) and
[the improvement plan](../../IMPROVEMENT_PLAN.md). Implementation commit:
`fb9d0f7ad205dbbfa1de6f2168180ca37e2c14be`.

The reports preserve failures and uncertainty. A signature match is a screening
result, not a Boolean-certified training label. The 4,935 signature candidates
become 4,921 after also excluding the 14 representation-coverage cases. Source
stability is an observation over these runs, not a general guarantee.

| File | Contents |
| --- | --- |
| `manifest.json` | Commits, archive hash, environment, test counts, main results, and limits |
| `environment.txt` | Installed Python package versions for this audit |
| `audit-5k.jsonl.gz` | Every first-pass execution record, including failures |
| `recheck-*.jsonl.gz` | Targeted repetitions, changed-output checks, and remaining exceptions |
| `canonical_hashes.json.gz` | Source and final canonical hashes for all 5,000 programs |
| `corpus_decisions.jsonl.gz` | Conservative per-input classification and evidence pointers |
| `corpus_summary.json` | Classification totals and signature-plus-coverage candidate count |
| `coverage_failures.json` | Fourteen incomplete parameter/placement/lowering cases |
| `audit-tracer-release.jsonl.gz` | Seed-42 sample: 125/128 CC-step Boolean passes; 114 legacy trace passes |
| `audit-fixtures-confirmed.jsonl.gz` | All 12 fixtures: 11 strict matches, one indeterminate Boolean comparison |
| `structural-release.summary.json` | Final structural scan across all 5,000 sources |
| `cases-final.summary.json`, `case_failures.json` | 33/34 adversarial cases pass; loop-iterable lowering remains unsupported |
| `legacy-regressions-before.json` | Five tracer failures reproduced against the untouched baseline |
| `pipeline-smoke.summary.json` | 9/9 Zero-to-CAD fixtures accepted/written by the batch pipeline |
| `pytest-release.log`, `focused-final.log` | Final full suite and targeted regression results |

Each audit has an accompanying `.summary.json` with its invocation options and
elapsed time. Paths in those options describe the original temporary run layout;
the JSONL evidence itself identifies inputs by the archive filename and hashes.
Some code-level reports contain canonical output only for failed cases, so final
hashes were independently recomputed from every source.

To regenerate the classification and compressed decision file without executing
CAD programs:

```bash
PYTHONPATH=.:dataset_utils python reports/cc_step_astra/reproduce_summary.py
```

The classifier takes the union of observed source instability across the full
screen and targeted rechecks. It keeps worker failures flagged even when a later
run succeeds. The three specifically identified first-pass dataclass namespace
errors are excluded from source-instability inference because the evaluator
itself caused them; their corrected ordinary-module executions passed. A
signature candidate also needs a successful execution with its **final**
canonical hash. The script does not infer geometry equivalence from a source
hash, a successful rerun alone, or the absence of a Python exception.

The stricter 128-program sample is reported separately and is not extrapolated
to certify the 5,000-program corpus. Before training, intersect screening and
coverage decisions with successful production Boolean/prefix validation records.

JSONL evidence is gzip-compressed with a fixed gzip timestamp. To inspect it:

```bash
gzip -dc reports/cc_step_astra/audit-tracer-release.jsonl.gz
```
