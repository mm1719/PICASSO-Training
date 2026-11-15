# Publication scope and rights boundaries

The project owner has explicitly authorized publication of the Phase 4
training/evaluation source, excluding datasets, weights, private history and
credentials. That source-publication approval is accepted and is not a pending
gate. This selected tree is prepared for that authorized source-only showcase.

**No source-code reuse license is granted.** No license has been selected or
added, and this is not represented as an open-source release. Publication
permission does not itself create a blanket downstream license or clear every
separate artifact. Any future license choice needs the appropriate rightsholders'
authority and applicable attribution review.

## Source provenance and notices

No top-level source license or attribution notice was identified in the
inspected Phase 4 project. That is not proof of authorship or of permission.
Some selected inputs are modified or untracked relative to the recorded HEAD;
authorship and provenance need review for those exact file hashes. Check whether
parts derive from earlier phases, external implementations, collaborators or
restricted code. For example, the retained `panaiotis_relative` method name is
not itself a sufficient citation or attribution record. Preserve any applicable
notices when provenance is established. No copyright owner or license is
invented in this candidate.

## Model code, pretrained artifacts, fine-tuned weights and datasets are separate

- Library license metadata describes installed library distributions. It does
  not license this research source, a model zoo's artifacts, or user datasets.
- The original flagship config names `convnextv2_large.fcmae_ft_in22k_in1k` with
  `pretrained: true`. The prior artifact-specific review reported CC BY-NC 4.0
  for that pretrained artifact. Recheck the exact artifact and its model card
  before any future use/distribution. This candidate contains no such artifact.
- Fine-tuned checkpoint rights remain unresolved. No fine-tuned or pretrained
  weights, checkpoint bytes, exports or downloads are included. The adapted
  configuration validator rejects pretrained initialization, external pretrained
  checkpoints, arbitrary model kwargs and remote model references.
- Dataset terms must be reviewed source by source, including the separate
  EcoTaxa path. Do not infer that dataset licenses automatically attach to every
  trained weight, or that source-code approval clears weights or data. This is
  an unresolved, artifact-specific review question, not a legal conclusion.

## Dependency review

`dependency-metadata.json` records installed direct-dependency version/license
metadata without importing the ML frameworks. It is a starting inventory, not a
complete SBOM, compatibility opinion or redistribution clearance. No third-party
library implementation is vendored here. Review transitive packages, notices,
model implementations and exact pretrained artifact terms separately before
choosing the eventual source license.

## Scope checklist

- [x] Owner explicitly authorized Phase 4 training/evaluation source publication.
- [x] No project license selected or granted for this source-only showcase.
- [x] Dataset rows, original manifests/splits, images and all weights excluded.
- [x] Private history, logs, paths, account identifiers and credentials excluded.
- [x] Synthetic checks are labeled as plumbing tests, not model-performance evidence.
- [x] Fresh Linux x86_64 / Python 3.11.15 evaluation-only install, 19 tests,
  synthetic demo and `pip check` passed on 2026-11-15; see [validation](VALIDATION.md).
- [ ] Fresh training-stack installation, full trainer execution and other platforms remain unverified.
- [ ] Any future reuse license requires appropriate rights and attribution review.
- [ ] Any future data/weights require their own publication decision and permissions.

The preparation step created no Git database, remote or public upload and made
no license grant. Publication itself is a separate authorized step performed
from the reviewed source snapshot, not the private research workspace.
