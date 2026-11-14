# Provenance and candidate-only changes

Preparation date: 2026-11-14. Source HEAD:
`d90077c0e298f6754939d84a8efdbcedb9f6c3a7`.

The input was the selected **working tree**, not a pristine checkout of this
commit. `source-provenance.json` records the exact SHA-256 of each selected
source input. The trainer was modified relative to HEAD; the paired-audit script,
its test and the Round 3 configuration were untracked. No history was copied.
Those facts make the content reproducible to a reviewer who has authorized
access to the source files. Source publication has been explicitly authorized by
the project owner; hashes are provenance evidence, not a downstream reuse license.

## Retained source

- Reference timm trainer, early-round data preparer, paired validation audit.
- Evaluation `common` and `metrics` modules and their package marker.
- Two original model-free test files.
- Hyperparameters adapted from the original RandAugment magnitude-4 configuration.
- Dependency pins extracted from the original environment specification, narrowed
  to the candidate's evaluation and training import surfaces.

## Candidate-only adaptations

- Removed run-specific defaults, account identity and external tracking
  initialization; removed credential lookup entirely. Inactive tracking callback
  slots remain in the trainer to minimize changes to the loop.
- Default pretrained loading is false. Static configuration checks reject
  pretrained initialization, pretrained checkpoint paths, arbitrary model kwargs,
  remote model references and external tracking.
  Training requires an explicit local opt-in. `--dry-run` now returns before
  reading manifests or creating a model, unlike the historical runtime check.
  The trainer still imports its ML frameworks at module load, including for
  dry-run; only the separate `scripts.validate_config` entrypoint is framework-free.
- The preparer now requires explicit input/output/checksum arguments; invalid
  ratios are rejected and output directories must be new. Split-policy naming
  reflects configured ratios rather than hardcoding 80/10/10.
- Removed hardcoded favorable A2/A0 prose from the paired-audit report. Reports
  now contain only computed quantities plus general interpretation limits.
  Empty/duplicate/misaligned/non-validation rows and invalid class spaces are
  rejected within the observed validation class space. The audit has no full
  model-class inventory and cannot detect absent trailing classes; callers must
  check that all intended model classes are represented. Report input references
  use basenames, not resolved local paths.
- Added a validated prediction-evaluation entrypoint. Training counts explicitly
  feed support bins; macro-F1 and balanced-accuracy denominator conventions are
  named. Outputs require a fresh directory.
- Added invented-data demo, targeted synthetic/mock-only tests, safety-oriented
  template config, dependency lists and review documentation.

The original selected files were not edited. A separate private review record
compares before/after Git status and hashes for 191 source/config/test/environment/
instruction files. Dataset contents, images, weights and production runtime state
were not scanned or touched. A status/hash comparison of this scope is not a
claim that every dataset byte on the computer was independently verified.

`candidate-sha256.json` inventories candidate files, excluding itself. Generated
synthetic outputs and private review evidence are outside the candidate tree.
