# PICASSO Phase 4: training and evaluation source candidate

**Source-only showcase. Source publication is authorized; no code-reuse license
is granted.** This is not described as an open-source release. No project license
has been selected. Source publication, data redistribution and model-weight
redistribution are separate decisions; see [scope and rights boundaries](docs/RELEASE_GATES.md).

A curated source-level showcase of a provenance-aware image-classification
workflow. It contains selected research code and a small, model-free synthetic
example. It is not the complete research workspace or a reproduction package.

## What is useful here

| Component | Retained capability |
| --- | --- |
| `scripts/prepare_round00_stage01_data.py` | Check a manifest digest; apply an explicit target map; persist a class index; deterministic target-stratified splits; tiny-class fallback; source/label/validity/risk provenance. |
| `scripts/train_round00_stage01_timm.py` | Reference timm training loop with class weighting/capped sampling, AdamW/cosine scheduling, EMA selection, compile options, heartbeat logging, resumable state and top-k prediction export. |
| `scripts/evaluate_predictions.py` | Model-free evaluation of existing predictions: top-k, F1, balanced accuracy, source/validity/risk slices, training-support bins, threshold-retention tables and normalized confusion matrices. |
| `scripts/audit_round03_stage02_promotion.py` | Strictly aligned validation pairs, class-stratified paired bootstrap, global and per-class deltas, and a high-support slice. |
| `scripts/synthetic_demo.py` | An invented 100-row manifest through support-file preparation, evaluation and a paired audit. No images, models, weights, training or GPU use. |

This shows how data contracts and provenance survive the training/evaluation
boundary, rather than presenting a single aggregate score without context.

## Quick check: no model or dataset required

On 2026-11-15, a fresh isolated Linux x86_64 environment with Python 3.11.15
passed installation of the evaluation-only dependencies below, all 19 tests,
the synthetic demo, and `pip check`. These dependencies are not the training
stack; fresh training-stack installation and other platforms remain unverified.
See [validation scope](docs/VALIDATION.md) for the exact boundaries.

```bash
python -m pip install -r requirements.txt
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
python -m scripts.synthetic_demo --output-dir demo-output
```

Choose a new output directory. Preparation, evaluation and audit entrypoints
refuse to overwrite an existing output directory. The synthetic example writes:

- supporting class/split/provenance files for 80 train, 10 validation, 10 test rows;
- fabricated validation predictions and evaluation tables;
- an identical-baseline/candidate audit, whose paired deltas and intervals are zero;
- `SYNTHETIC_ONLY.json` explicitly recording that no model or images were used.

**These numbers test plumbing, not scientific or model performance.** The example
does not fabricate a trained model, load a checkpoint, or reproduce a real run.
Generated demo files are ignored and are not part of this source-only candidate.
Use module invocation (`python -m scripts...`) from this directory.

## Evaluate your own authorized prediction exports

```bash
python -m scripts.evaluate_predictions \
  --predictions local-data/predictions.csv \
  --class-index local-data/class_index.csv \
  --output-dir local-reports/evaluation
python -m scripts.audit_round03_stage02_promotion \
  --baseline-csv local-data/baseline-validation.csv \
  --candidate-csv local-data/candidate-validation.csv \
  --output-dir local-reports/paired-audit
```

The evaluator expects five distinct ranked predictions and probabilities for
each row, contiguous zero-based classes, and at least five declared classes.
`class_index.csv` needs `class_index`, `target_display_name`, and `train_count`.
Prediction metadata needs `image_path`, `dataset_id`, `valid_status`,
`invalid_reason`, `risk_flag`, and `true_class_index`; rank columns follow
`pred_top1_index`/`pred_top1_prob` through rank 5. Paired auditing additionally
requires `source_id`, `target_id`, `true_class_name`, and `split=val`, with exactly
matching identity/order. The audit requires contiguous zero-based indexes among
**observed validation classes** and infers the class count from those rows. It
receives no separate full model class inventory, so it cannot detect absent
trailing model classes. **The caller must check the complete model class index
and confirm that every intended class is represented** before interpreting its
macro metrics. The audit rejects duplicate image identities rather than silently
weighting them twice.

Support bins use **training split counts**, not validation/test support or total
manifest counts. Global macro F1 uses all declared classes; balanced accuracy and
per-slice classification reports use observed true classes. Those denominators
are intentionally distinguished. Confidence thresholds report retained sample
counts; they are not calibrated probabilities or guarantees of correctness.

## Training source and reproducibility limits

The example configuration is adapted from R3S2-A2: Round 3, Stage 2's second
augmentation-ablation variant. A2 retains the A0 ConvNeXtV2-L architecture,
20-target label space and crop settings, and reduces RandAugment magnitude
from 9 to 4. It is an experiment identifier, not an architecture generation,
release version or performance rank. The example preserves core hyperparameters as
source context, but replaces private paths/identifiers with local placeholders,
turns pretrained initialization off, and disables external tracking. It is
**not** the configuration that produced the original results.

`python -m scripts.validate_config configs/training.example.yaml` performs only
static safety/shape checks and intentionally fails until a real manifest SHA-256
is supplied. This standalone validator never imports the model framework or
opens data; use it with the evaluation-only dependency setup. By contrast, the
trainer module imports torch, timm and torchvision at module load, **even for
`--dry-run`**, so that command requires the training dependencies. Its dry-run
then checks configuration without reading datasets, constructing/loading a
model or performing GPU/runtime checks. Actual training also requires
`--allow-local-training`, the training dependencies,
your own authorized data/support files, and a working CUDA environment. No
training, model construction, checkpoint loading, or GPU validation was performed
when preparing this candidate.

The reference trainer now chooses one top-five ranking from logits for both
overall test metrics and prediction CSVs, gathering softmax scores at those
same indices. It does not rank rounded softmax values separately. This prevents
half-precision rounding/underflow from making the two outputs disagree. Exact
logit ties retain PyTorch's tie ordering; cross-device tie ordering is not
guaranteed. Historical experiment results are not rewritten by this fix.

Five additional CPU tensor regression tests import the actual trainer and call
its prediction exporter with synthetic tensors, without images, weights or
training. They require `requirements-training.txt`; they are separate from
the 19 framework-free quick checks above. With those dependencies installed:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests/training -v
```

Do not run this candidate as a substitute for the private workspace's
lock-managed launcher on a shared training/inference GPU. That deployment
integration is intentionally absent. Arrange exclusive compute or an appropriate
scheduler in any independent environment before using the reference trainer.
Checkpoint resumption uses `torch.load(..., weights_only=False)` to restore full
training state: **only load a locally produced, trusted checkpoint**. This
candidate is not a hardened service for untrusted configs, checkpoints or CSVs.

The original research lineage consumes a **six-source main manifest plus a
separate EcoTaxa manifest**. Neither is included. The retained early-round
preparer is a single-manifest target-mapping/splitting example; it does not
reconstruct the later 20-target ontology, multi-manifest integration, or exact
Round 3 splits. The trainer retains extra-manifest checksum support, but the
example config does not invent those private inputs.

Image-row splits and row-level paired bootstrap do not establish independence
between images from the same sample, cruise, instrument, or duplicate group.
They do not measure cross-domain generalization, prevent every form of leakage,
or estimate seed-to-seed training variation. Validate grouping policies and
independent test cohorts before making deployment claims.

## What is intentionally absent

No Git history/database, original datasets/manifests/splits, images, trained or
pretrained weights, result exports, account identifiers, private paths,
credentials, W&B payloads, private logs, agent instructions, production services,
GPU launchers, or outbound-message automation. No real accuracy, speedup, or
"best model" claim is made here. The plotting, full artifact packager, FCMAE
pretrainer, broad architecture sweep and production automation were not copied.

See [provenance and adaptations](docs/PROVENANCE.md),
[validation scope](docs/VALIDATION.md), and [release gates](docs/RELEASE_GATES.md).
