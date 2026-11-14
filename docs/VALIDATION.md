# Validation scope

Validated in the existing Python 3.11 research environment using the recorded
installed evaluation dependencies. No packages were installed or upgraded.

- 17 unit/integration tests passed, including a 100-row invented-manifest pipeline.
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
- The trainer module is not imported by the checks. Torch, torchvision, timm,
  model construction, checkpoint deserialization, data loading, optimizer steps,
  compilation, CUDA and production services are outside the tested execution path.
  This does not mean the real trainer CLI is framework-free: its top-level imports
  run even for `--dry-run`. The standalone `scripts.validate_config` entrypoint is
  the framework-free option for an evaluation-only installation.
- The full test run also passed with ML/tracking framework imports and network
  connection calls explicitly blocked. The standalone synthetic CLI passed.
- Python source parses successfully. Candidate inventory is source/documentation
  only, without images, datasets, model artifacts or Git databases.

Not performed: fresh-environment install, full dependency resolution, lint/type
certification, complete trainer imports, real-data validation, model loading,
training, GPU/runtime benchmarking, numerical equivalence to original runs,
full pipeline reproduction, artifact-specific data/weight rights clearance, license selection or publication.
A passing synthetic suite is not evidence of model quality, production safety,
training throughput or permission to publish.
