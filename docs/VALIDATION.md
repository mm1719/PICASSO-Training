# Validation scope

Initial validation used the existing Python 3.11 research environment and its
installed evaluation dependencies. That environment was not modified. The
2026-11-15 fresh-environment check below is a separate evaluation-only check.

- 19 unit/integration tests passed, including a 100-row invented-manifest pipeline.
- Empty and incorrect expected manifest checksums are rejected before creating
  the output directory. The empty-checksum regression failed before the
  2026-11-15 fix; matching checksums remain covered by the synthetic pipeline.
- Synthetic checks cover deterministic 80/10/10 allocation and tiny-class
  fallback; validity metadata preservation; top-k metrics and source slices;
  paired-identity validation; deterministic bootstrap and zero deltas for
  identical predictions; malformed input rejection; no-overwrite behavior.
- Trainer checks execute selected AST definitions with a fake tensor backend and
  mocked compile/checkpoint calls. They check class-cap weights, missing-class
  rejection, compile argument forwarding/conflicts, resume metadata guards,
  safe configuration rejection and the config-only/opt-in branches.
- The audit validates the observed zero-based class space, not completeness
  against a separate model class inventory. Missing trailing model classes must
  be detected by the caller before interpreting the paired macro metrics.
- The trainer module was not imported by these initial checks. Torch, torchvision, timm,
  model construction, checkpoint deserialization, data loading, optimizer steps,
  compilation, CUDA and production services are outside the tested execution path.
  This does not mean the real trainer CLI is framework-free: its top-level imports
  run even for `--dry-run`. The standalone `scripts.validate_config` entrypoint is
  the framework-free option for an evaluation-only installation.
- The full test run also passed with ML/tracking framework imports and network
  connection calls explicitly blocked. The standalone synthetic CLI passed.
- Python source parses successfully. Candidate inventory is source/documentation
  only, without images, datasets, model artifacts or Git databases.

## 2026-11-15 fresh-environment and publication check

The reviewed executable source is commit
`8f84eb9f56395789ca9076cceece029f0561f102`; the accompanying documentation and
checksum refresh does not change that source, the tests or dependency versions.

- A new isolated `venv` on Linux x86_64 / Python 3.11.15 installed
  `requirements.txt` successfully from PyPI. It did not inherit the research
  environment's installed packages; torch was absent.
- All 19 tests passed in both the research environment and the isolated
  evaluation environment, including empty/wrong manifest checksum rejection.
- The standalone synthetic CLI passed in both environments: 100 invented rows,
  80/10/10 train/validation/test allocation, and no images or model construction.
- `python -m pip check` returned exit 0 with no broken requirements in the
  isolated evaluation environment.
- Direct versions: numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, PyYAML 6.0.3.
  Resolved transitive versions: scipy 1.17.1, joblib 1.6.0, threadpoolctl 3.8.0,
  python-dateutil 4.0.0, cloudpickle 3.1.2 and tzdata 2026.6. This is one tested
  resolution, not a complete cross-platform lockfile.
- All 22 source/documentation entries in `candidate-sha256.json` matched their
  file bytes. Python parsing and relative documentation links passed. Checks
  of reachable historical file contents found no matches for the inspected
  credential or machine-local-path patterns; this is not proof of absence of
  every possible secret or personal identifier.

The environment and generated fixtures were isolated from the research runs.
No original data, checkpoint, tracking credentials or production services were
used. Source-only publication and the existing no-reuse-license position are
unchanged; these checks do not extend permission to datasets or weights.

## 2026-11-15 single-ranking regression check

The earlier trainer ranked logits for overall test metrics but independently
ranked softmax scores for CSV export. Half-precision softmax can round distinct
scores to ties or underflow them to zero, allowing those rankings to disagree.
The updated `predict_test` ranks logits once, uses that ranking for all top-k
counts and exported labels, and gathers softmax scores at the same indices.
Exact ties in logits are not assigned a new cross-device tie-breaking policy.

- The new FP16 underflow test failed on the previous code and passed after the
  fix. No model weights or real images are involved.
- Five CPU tests in `tests/training/test_prediction_ranking.py` cover FP16
  underflow, FP16 rounding of distinct logits, exact logit ties, FP32 unequal
  batches and the multi-crop prediction branch. They check all five metric
  hit rates against CSV rankings, the ranked scores and top-one label mapping.
- These tests import the full trainer and call its actual prediction exporter
  with real PyTorch CPU tensors, a parameter-free flatten module and an
  in-memory iterable. This is not a DataLoader, training or model-accuracy test.
- The tests passed on Linux x86_64 / Python 3.11.15 with torch 2.11.0+cu130,
  torchvision 0.26.0+cu130 and timm 1.0.26 in the existing research environment,
  without modifying it. CUDA was hidden, autocast disabled, and FP16 inputs
  supplied explicitly. This is not evidence of CUDA/autocast execution.
- The separate 19 evaluation/mock tests and synthetic demo remain executable
  in the isolated evaluation-only environment without torch. The additional
  tests require the training dependencies and explicit discovery:
  `python -m unittest discover -s tests/training -v`.
- The current candidate checksum inventory covers 23 payload files, excluding
  the inventory itself. Historical source-input hashes in
  `source-provenance.json` remain unchanged.

No historical prediction, metric, checkpoint, model weight or license is changed.
The corresponding saved-results discrepancy is documented in the
[Models experiment report](https://github.com/mm1719/PICASSO-Models/blob/main/experiments/round03_stage02/README.md#historical-top-k-consistency-note).

## Not verified

Not performed: fresh training-stack installation, other operating systems or
Python versions, lint/type certification, real-data
validation, model loading, training, GPU/runtime benchmarking, numerical
equivalence to original runs, full pipeline reproduction, artifact-specific
data/weight rights clearance, or selection of a reuse license.
A passing synthetic suite is not evidence of model quality, production safety,
training throughput or permission to publish.
