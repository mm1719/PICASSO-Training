#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import os
import random
import shutil
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from PIL import Image, ImageFile
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from scripts.validate_config import validate_config


ImageFile.LOAD_TRUNCATED_IMAGES = True
LOGGER = logging.getLogger("phase4_round00_stage01")
TAIPEI = ZoneInfo("Asia/Taipei")


@dataclass(frozen=True, slots=True)
class SplitRow:
    image_path: str
    class_index: int


@dataclass(frozen=True, slots=True)
class ManifestRow:
    image_path: str
    dataset_id: str
    valid_status: str
    invalid_reason: str
    risk_flag: str
    source_id: str
    target_id: str
    class_index: int
    split: str


@dataclass(frozen=True, slots=True)
class HeartbeatConfig:
    enabled: bool
    step_interval: int
    time_interval_seconds: float


@dataclass(frozen=True, slots=True)
class ResumeState:
    model: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    scaler: dict[str, Any]
    ema: dict[str, Any] | None
    completed_epoch: int
    best_metric_value: float
    best_val_accuracy: float
    best_epoch: int
    checkpoint_metric: str
    config_sha256: str
    num_classes: int
    class_names: list[str]
    target_ids: list[str]
    rng_state: dict[str, Any]
    criterion: dict[str, Any] | None = None


class ModelEma:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.module = copy.deepcopy(original_model(model)).eval()
        self.decay = decay
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source_state = original_model(model).state_dict()
        target_state = self.module.state_dict()
        for name, value in source_state.items():
            target_value = target_state[name]
            if torch.is_floating_point(target_value):
                target_value.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                target_value.copy_(value)


class ImageSplitDataset(Dataset):
    def __init__(self, rows: list[SplitRow], transform: Any) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.rows[idx]
        with Image.open(row.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, row.class_index


class ManifestSplitDataset(Dataset):
    def __init__(self, rows: list[ManifestRow], transform: Any) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict[str, str]]:
        row = self.rows[idx]
        with Image.open(row.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        metadata = {
            "image_path": row.image_path,
            "dataset_id": row.dataset_id,
            "valid_status": row.valid_status,
            "invalid_reason": row.invalid_reason,
            "risk_flag": row.risk_flag,
            "source_id": row.source_id,
            "target_id": row.target_id,
            "split": row.split,
        }
        return tensor, row.class_index, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train phase 4 round 0 stage 1 timm anchor.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional explicit source-run directory. Launchers use this to make run paths deterministic.",
    )
    parser.add_argument("--allow-local-training", action="store_true", help="Explicitly enable local training after arranging data rights and exclusive GPU access.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config only; do not read data, create a model, or touch CUDA.")
    parser.add_argument(
        "--test-num-workers",
        type=int,
        default=None,
        help="Override DataLoader workers for final test evaluation. Use 0 for recovery-safe eval.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume training from an atomic last_checkpoint.pt written by this script.",
    )
    return parser.parse_args()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def save_last_checkpoint(path: Path, state: ResumeState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    payload = {
        "checkpoint_kind": "last",
        "model": state.model,
        "optimizer": state.optimizer,
        "scheduler": state.scheduler,
        "scaler": state.scaler,
        "ema": state.ema,
        "completed_epoch": state.completed_epoch,
        "next_epoch": state.completed_epoch + 1,
        "best_val_accuracy": state.best_val_accuracy,
        "best_epoch": state.best_epoch,
        "best_validation_metric": {
            "metric": state.checkpoint_metric,
            "value": state.best_metric_value,
            "epoch": state.best_epoch,
            "epoch_numbering": "1-based",
        },
        "config_sha256": state.config_sha256,
        "num_classes": state.num_classes,
        "class_names": state.class_names,
        "target_ids": state.target_ids,
        "rng_state": state.rng_state,
        "criterion": state.criterion,
    }
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_last_checkpoint(
    path: Path,
    *,
    expected_config_sha256: str,
    expected_num_classes: int,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_kind") != "last":
        raise ValueError(f"Unsupported checkpoint kind: {checkpoint.get('checkpoint_kind')}")
    config_sha256 = checkpoint.get("config_sha256")
    if config_sha256 != expected_config_sha256:
        raise ValueError(f"Config hash mismatch: checkpoint={config_sha256} expected={expected_config_sha256}")
    num_classes = int(checkpoint.get("num_classes", -1))
    if num_classes != expected_num_classes:
        raise ValueError(f"num_classes mismatch: checkpoint={num_classes} expected={expected_num_classes}")
    return checkpoint


def restore_training_state(
    checkpoint: dict[str, Any],
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    model_ema: ModelEma | None,
    criterion: nn.Module | None = None,
) -> int:
    original_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    if model_ema is not None and checkpoint.get("ema") is not None:
        model_ema.module.load_state_dict(checkpoint["ema"])
    if criterion is not None and checkpoint.get("criterion") is not None:
        criterion.load_state_dict(checkpoint["criterion"])
    return int(checkpoint["next_epoch"])


def collect_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(rng_state: dict[str, Any]) -> None:
    if not rng_state:
        return
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and rng_state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest_checksums(project_root: Path, manifest_conf: dict[str, Any]) -> None:
    manifest_path = resolve_path(project_root, manifest_conf["path"])
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != manifest_conf["sha256"]:
        raise ValueError(f"Manifest checksum mismatch: got {manifest_hash}")
    for item in manifest_conf.get("extra_manifests", []):
        extra_path = resolve_path(project_root, item["path"])
        extra_hash = sha256_file(extra_path)
        if extra_hash != item["sha256"]:
            raise ValueError(f"Extra manifest checksum mismatch for {extra_path}: got {extra_hash}")


def timestamp() -> str:
    return os.environ.get("PHASE4_TIMESTAMP_OVERRIDE") or datetime.now(TAIPEI).strftime("%Y-%m-%d_%H%M%S")


def setup_logging(run_dir: Path) -> None:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "training.log")],
    )


def read_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_split_txt(path: Path) -> list[SplitRow]:
    rows: list[SplitRow] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                image_path, class_index = line.rsplit(maxsplit=1)
            except ValueError as exc:
                raise ValueError(f"Malformed split line {path}:{line_no}: {line}") from exc
            rows.append(SplitRow(image_path=image_path, class_index=int(class_index)))
    return rows


def read_manifest_split(path: Path, split: str) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            rows.append(
                ManifestRow(
                    image_path=row["image_path"],
                    dataset_id=row["dataset_id"],
                    valid_status=row["valid_status"],
                    invalid_reason=row["invalid_reason"],
                    risk_flag=row["risk_flag"],
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    class_index=int(row["class_index"]),
                    split=row["split"],
                )
            )
    return rows


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_transforms(model: nn.Module, conf: dict[str, Any]) -> tuple[Any, Any, Any]:
    image_size = int(conf["model"]["image_size"])
    data_cfg = resolve_model_data_config(model)
    mean = data_cfg.get("mean", (0.485, 0.456, 0.406))
    std = data_cfg.get("std", (0.229, 0.224, 0.225))
    interpolation_name = data_cfg.get("interpolation", "bicubic")

    train_cfg = conf.get("augmentation", {}).get("train", {})
    train_mode = train_cfg.get("mode", "timm_imagenet_standard")
    if train_mode == "timm_imagenet_standard":
        train_transform = create_transform(
            input_size=(3, image_size, image_size),
            is_training=True,
            color_jitter=train_cfg.get("color_jitter", 0.4),
            auto_augment=train_cfg.get("auto_augment", "rand-m9-mstd0.5-inc1"),
            scale=tuple(train_cfg.get("crop_scale", (0.08, 1.0))),
            ratio=tuple(train_cfg.get("crop_ratio", (0.75, 1.3333))),
            interpolation=interpolation_name,
            mean=mean,
            std=std,
            hflip=train_cfg.get("hflip", 0.5),
        )
    elif train_mode == "flowcam_center_rotate_gray":
        interpolation = transforms.InterpolationMode.BICUBIC
        train_transform = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.RandomRotation(
                            degrees=float(train_cfg.get("rotation_degrees", 180.0)),
                            interpolation=interpolation,
                            fill=0,
                        )
                    ],
                    p=float(train_cfg.get("rotation_p", 0.7)),
                ),
                transforms.RandomHorizontalFlip(p=float(train_cfg.get("hflip", 0.5))),
                transforms.RandomVerticalFlip(p=float(train_cfg.get("vflip", 0.5))),
                transforms.RandomApply(
                    [transforms.RandomAutocontrast()],
                    p=float(train_cfg.get("autocontrast", 0.3)),
                ),
                transforms.RandomApply(
                    [transforms.RandomEqualize()],
                    p=float(train_cfg.get("equalize", 0.1)),
                ),
                transforms.Resize(image_size, interpolation=interpolation),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        raise ValueError(f"Unsupported train augmentation mode: {train_mode}")
    eval_transform = create_transform(
        input_size=(3, image_size, image_size),
        is_training=False,
        interpolation=interpolation_name,
        mean=mean,
        std=std,
    )
    ten_crop_transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.TenCrop(image_size),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize(mean=mean, std=std)(transforms.ToTensor()(crop)) for crop in crops]
                )
            ),
        ]
    )
    return train_transform, eval_transform, ten_crop_transform


def compute_class_weights(rows: list[SplitRow], num_classes: int, mode: str, exponent: float) -> torch.Tensor:
    labels = np.fromiter((row.class_index for row in rows), dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    positive = counts > 0
    if not positive.all():
        missing = np.where(~positive)[0].tolist()
        raise ValueError(f"Class index has zero train examples for classes: {missing[:20]}")
    weights = np.zeros(num_classes, dtype=np.float64)
    if mode == "panaiotis_relative":
        weights[positive] = np.power(counts[positive].max() / counts[positive], exponent)
    elif mode == "balanced":
        weights[positive] = counts[positive].sum() / counts[positive]
    else:
        raise ValueError(f"Unsupported class weight mode: {mode}")
    return torch.tensor(weights, dtype=torch.float32)


def class_count_cap_sample_weights(rows: list[SplitRow], num_classes: int, cap: int) -> torch.Tensor:
    if cap <= 0:
        raise ValueError(f"sampler class_count_cap must be positive, got {cap}")
    labels = np.fromiter((row.class_index for row in rows), dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    positive = counts > 0
    if not positive.all():
        missing = np.where(~positive)[0].tolist()
        raise ValueError(f"Class index has zero train examples for classes: {missing[:20]}")
    per_class = np.minimum(1.0, cap / counts)
    return torch.tensor(per_class[labels], dtype=torch.double)


def build_scheduler(
    optimizer: optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int,
) -> optim.lr_scheduler.LambdaLR:
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(0, warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def heartbeat_config(conf: dict[str, Any]) -> HeartbeatConfig:
    heartbeat = conf.get("monitor", {}).get("heartbeat", {})
    return HeartbeatConfig(
        enabled=bool(heartbeat.get("enabled", True)),
        step_interval=max(1, int(heartbeat.get("step_interval", 500))),
        time_interval_seconds=max(1.0, float(heartbeat.get("time_interval_seconds", 300))),
    )


def configured_step_limit(conf: dict[str, Any], key: str) -> int | None:
    value = conf["training"].get(key)
    if value in (None, "", 0):
        return None
    limit = int(value)
    if limit <= 0:
        return None
    return limit


def effective_step_count(loader: DataLoader, limit: int | None) -> int:
    if limit is None:
        return len(loader)
    return min(len(loader), limit)


def checkpoint_metric_name(conf: dict[str, Any]) -> str:
    metric = conf["training"].get("checkpoint_metric", "val_accuracy")
    supported = {
        "val_accuracy",
        "val_balanced_accuracy",
        "val_macro_f1",
        "val_weighted_f1",
        "raw_val_accuracy",
        "raw_val_balanced_accuracy",
        "raw_val_macro_f1",
        "raw_val_weighted_f1",
    }
    if metric not in supported:
        raise ValueError(f"Unsupported checkpoint_metric: {metric}")
    return metric


def model_kwargs(conf: dict[str, Any]) -> dict[str, Any]:
    model_conf = conf.get("model", {})
    kwargs: dict[str, Any] = {}
    for key in ["drop_path_rate", "ls_init_value"]:
        value = model_conf.get(key)
        if value is not None:
            kwargs[key] = float(value)
    kwargs.update(model_conf.get("extra_kwargs", {}))
    return kwargs


def maybe_compile_model(model: nn.Module, conf: dict[str, Any]) -> nn.Module:
    compile_conf = conf["training"].get("torch_compile", {})
    if isinstance(compile_conf, bool):
        enabled = compile_conf
        mode = "default"
        dynamic = None
        options = None
    else:
        enabled = bool(compile_conf.get("enabled", False))
        mode = compile_conf.get("mode")
        dynamic = compile_conf.get("dynamic")
        options = compile_conf.get("options")
    if not enabled:
        return model
    if mode and options:
        raise ValueError("torch_compile.mode and torch_compile.options are mutually exclusive")
    compile_kwargs: dict[str, Any] = {"mode": mode or "default"}
    if dynamic is not None:
        compile_kwargs["dynamic"] = bool(dynamic)
    if options:
        compile_kwargs.pop("mode")
        compile_kwargs["options"] = dict(options)
    LOGGER.info("[runtime] torch.compile enabled kwargs=%s", compile_kwargs)
    return torch.compile(model, **compile_kwargs)


def original_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


class ParametricContrastiveLoss(nn.Module):
    requires_features = True

    def __init__(
        self,
        *,
        num_classes: int,
        feature_dim: int,
        temperature: float,
        ce_weight: float,
        paco_weight: float,
        class_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.class_centers = nn.Parameter(torch.randn(num_classes, feature_dim) * 0.02)
        self.temperature = temperature
        self.ce_weight = ce_weight
        self.paco_weight = paco_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, labels)
        features = torch.nn.functional.normalize(features.float(), dim=1)
        centers = torch.nn.functional.normalize(self.class_centers.float(), dim=1)
        contrast_bank = torch.cat([features, centers], dim=0)
        contrast_logits = features @ contrast_bank.T / self.temperature
        batch_size = features.shape[0]
        self_mask = torch.eye(batch_size, device=features.device, dtype=torch.bool)
        positive_samples = labels[:, None].eq(labels[None, :]) & ~self_mask
        positive_centers = torch.nn.functional.one_hot(labels, centers.shape[0]).bool()
        positive_mask = torch.cat([positive_samples, positive_centers], dim=1)
        contrast_logits[:, :batch_size] = contrast_logits[:, :batch_size].masked_fill(self_mask, -1e4)
        log_prob = torch.nn.functional.log_softmax(contrast_logits, dim=1)
        positive_count = positive_mask.sum(dim=1).clamp_min(1)
        paco_loss = -(log_prob * positive_mask).sum(dim=1) / positive_count
        return self.ce_weight * ce_loss + self.paco_weight * paco_loss.mean()


def forward_logits_and_features(model: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    base_model = original_model(model)
    features_map = base_model.forward_features(images)
    features = base_model.forward_head(features_map, pre_logits=True)
    logits = base_model.forward_head(features_map)
    return logits, features


def load_pretrained_encoder_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("encoder") or payload.get("model")
    if state is None:
        raise ValueError(f"Checkpoint has no encoder/model state: {checkpoint_path}")
    missing, unexpected = original_model(model).load_state_dict(state, strict=False)
    LOGGER.info(
        "[model] Loaded pretrained encoder checkpoint %s missing=%d unexpected=%d",
        checkpoint_path,
        len(missing),
        len(unexpected),
    )


def build_train_criterion(
    conf: dict[str, Any],
    class_weight_tensor: torch.Tensor | None,
    feature_dim: int,
) -> nn.Module:
    loss_conf = conf["training"].get("loss", {})
    loss_name = loss_conf.get("name", "cross_entropy") if isinstance(loss_conf, dict) else "cross_entropy"
    label_smoothing = float(conf["training"].get("label_smoothing", 0.0))
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=label_smoothing)
    if loss_name == "paco":
        return ParametricContrastiveLoss(
            num_classes=int(conf["model"]["num_classes"]),
            feature_dim=feature_dim,
            temperature=float(loss_conf.get("temperature", 0.2)),
            ce_weight=float(loss_conf.get("ce_weight", 1.0)),
            paco_weight=float(loss_conf.get("paco_weight", 0.2)),
            class_weight=class_weight_tensor,
            label_smoothing=label_smoothing,
        )
    raise ValueError(f"Unsupported training.loss.name: {loss_name}")


def log_heartbeat(
    *,
    epoch: int,
    total_epochs: int,
    step: int,
    steps_per_epoch: int,
    recent_loss: float,
    learning_rate: float,
    samples_per_second: float,
    elapsed_seconds: float,
    device: torch.device,
    wandb_run: Any | None,
) -> None:
    gpu_memory_mb = 0.0
    if device.type == "cuda":
        gpu_memory_mb = float(torch.cuda.memory_allocated(device) / (1024**2))
    train_step = (epoch - 1) * steps_per_epoch + step
    epoch_progress = (epoch - 1) + (step / steps_per_epoch)
    LOGGER.info(
        "[heartbeat] epoch=%03d/%03d step=%d/%d loss=%.5f lr=%.8f samples_per_sec=%.2f gpu_mem_mb=%.1f elapsed_sec=%.1f",
        epoch,
        total_epochs,
        step,
        steps_per_epoch,
        recent_loss,
        learning_rate,
        samples_per_second,
        gpu_memory_mb,
        elapsed_seconds,
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "train_step": train_step,
                "epoch_progress": epoch_progress,
                "heartbeat/epoch": epoch,
                "heartbeat/step": step,
                "heartbeat/train_step": train_step,
                "heartbeat/epoch_progress": epoch_progress,
                "heartbeat/steps_per_epoch": steps_per_epoch,
                "heartbeat/recent_loss": recent_loss,
                "heartbeat/learning_rate": learning_rate,
                "heartbeat/samples_per_second": samples_per_second,
                "heartbeat/gpu_memory_allocated_mb": gpu_memory_mb,
                "heartbeat/elapsed_seconds": elapsed_seconds,
            }
        )


def define_wandb_metric_axes(wandb_module: Any) -> None:
    """Keep monitoring heartbeats from changing epoch-metric chart semantics."""
    wandb_module.define_metric("epoch")
    wandb_module.define_metric("train_step")
    wandb_module.define_metric("epoch_progress")
    wandb_module.define_metric("train/*", step_metric="epoch")
    wandb_module.define_metric("val/*", step_metric="epoch")
    wandb_module.define_metric("lr", step_metric="epoch")
    wandb_module.define_metric("epoch_seconds", step_metric="epoch")
    wandb_module.define_metric("heartbeat/*", step_metric="train_step")


def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...]) -> list[float]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum().item()) for k in topk]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LambdaLR,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    *,
    epoch: int,
    total_epochs: int,
    run_start_time: float,
    heartbeat: HeartbeatConfig,
    wandb_run: Any | None,
    model_ema: ModelEma | None = None,
    max_steps: int | None = None,
    effective_steps_per_epoch: int | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    last_heartbeat_step = 0
    last_heartbeat_time = time.time()
    last_heartbeat_samples = 0
    steps_per_epoch = effective_steps_per_epoch or effective_step_count(loader, max_steps)
    for step, (images, labels) in enumerate(loader, start=1):
        if max_steps is not None and step > max_steps:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            if getattr(criterion, "requires_features", False):
                logits, features = forward_logits_and_features(model, images)
                loss = criterion(logits, features, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before_step:
            scheduler.step()
            if model_ema is not None:
                model_ema.update(model)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += accuracy_topk(logits.detach(), labels, (1,))[0]
        total_samples += batch_size
        if heartbeat.enabled:
            now = time.time()
            due_by_step = step - last_heartbeat_step >= heartbeat.step_interval
            due_by_time = now - last_heartbeat_time >= heartbeat.time_interval_seconds
            if due_by_step or due_by_time:
                interval_samples = total_samples - last_heartbeat_samples
                interval_seconds = max(1e-6, now - last_heartbeat_time)
                log_heartbeat(
                    epoch=epoch,
                    total_epochs=total_epochs,
                    step=step,
                    steps_per_epoch=steps_per_epoch,
                    recent_loss=float(loss.item()),
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    samples_per_second=interval_samples / interval_seconds,
                    elapsed_seconds=now - run_start_time,
                    device=device,
                    wandb_run=wandb_run,
                )
                last_heartbeat_step = step
                last_heartbeat_time = now
                last_heartbeat_samples = total_samples
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "lr": optimizer.param_groups[0]["lr"],
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    max_steps: int | None = None,
    num_classes: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    true_labels: list[int] = []
    pred_labels: list[int] = []
    for step, (images, labels) in enumerate(loader, start=1):
        if max_steps is not None and step > max_steps:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += accuracy_topk(logits, labels, (1,))[0]
        total_samples += batch_size
        true_labels.extend(labels.cpu().tolist())
        pred_labels.extend(logits.argmax(dim=1).cpu().tolist())
    labels_arg = list(range(num_classes)) if num_classes is not None else None
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=labels_arg,
        zero_division=0,
    )
    support_arr = np.asarray(support, dtype=np.float64)
    support_sum = float(support_arr.sum())
    weighted_f1 = float(np.average(f1, weights=support_arr)) if support_sum > 0 else 0.0
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, pred_labels)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": weighted_f1,
        "macro_recall": float(np.mean(recall)),
        "macro_precision": float(np.mean(precision)),
    }


@torch.no_grad()
def predict_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    target_ids: list[str],
    use_amp: bool,
    output_csv: Path,
) -> dict[str, Any]:
    model.eval()
    true_labels: list[int] = []
    pred_labels: list[int] = []
    total_top = {k: 0.0 for k in (1, 2, 3, 4, 5)}
    total_samples = 0
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_path",
        "dataset_id",
        "valid_status",
        "invalid_reason",
        "risk_flag",
        "source_id",
        "target_id",
        "true_class_index",
        "true_class_name",
        "pred_top1_index",
        "pred_top1_target_id",
        "pred_top1_name",
        "pred_top1_prob",
        "pred_top2_index",
        "pred_top2_prob",
        "pred_top3_index",
        "pred_top3_prob",
        "pred_top4_index",
        "pred_top4_prob",
        "pred_top5_index",
        "pred_top5_prob",
        "split",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for images, labels, metadata in loader:
            labels = labels.to(device, non_blocking=True)
            if images.ndim == 5:
                bsz, crops, channels, height, width = images.shape
                images = images.reshape(bsz * crops, channels, height, width).to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(images).reshape(bsz, crops, -1).mean(dim=1)
            else:
                images = images.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(images)

            probs = torch.softmax(logits, dim=1)
            # Rank once before softmax rounding can introduce additional ties.
            top_indices = logits.topk(5, dim=1).indices
            top_probs = probs.gather(1, top_indices)
            correct = top_indices.eq(labels[:, None])
            for k in total_top:
                total_top[k] += float(correct[:, :k].any(dim=1).sum().item())

            labels_cpu = labels.cpu().tolist()
            top_indices_cpu = top_indices.cpu().tolist()
            top_probs_cpu = top_probs.cpu().tolist()
            true_labels.extend(labels_cpu)
            pred_labels.extend(row[0] for row in top_indices_cpu)
            total_samples += len(labels_cpu)

            for idx, true_idx in enumerate(labels_cpu):
                top_idx = top_indices_cpu[idx]
                top_prob = top_probs_cpu[idx]
                writer.writerow(
                    {
                        "image_path": metadata["image_path"][idx],
                        "dataset_id": metadata["dataset_id"][idx],
                        "valid_status": metadata["valid_status"][idx],
                        "invalid_reason": metadata["invalid_reason"][idx],
                        "risk_flag": metadata["risk_flag"][idx],
                        "source_id": metadata["source_id"][idx],
                        "target_id": metadata["target_id"][idx],
                        "true_class_index": true_idx,
                        "true_class_name": class_names[true_idx],
                        "pred_top1_index": top_idx[0],
                        "pred_top1_target_id": target_ids[top_idx[0]],
                        "pred_top1_name": class_names[top_idx[0]],
                        "pred_top1_prob": top_prob[0],
                        "pred_top2_index": top_idx[1],
                        "pred_top2_prob": top_prob[1],
                        "pred_top3_index": top_idx[2],
                        "pred_top3_prob": top_prob[2],
                        "pred_top4_index": top_idx[3],
                        "pred_top4_prob": top_prob[3],
                        "pred_top5_index": top_idx[4],
                        "pred_top5_prob": top_prob[4],
                        "split": metadata["split"][idx],
                    }
                )

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    support_arr = np.asarray(support, dtype=np.float64)
    return {
        "checkpoint": "best_model.pt",
        "num_classes": len(class_names),
        "num_samples": total_samples,
        "topk_accuracy": {f"top{k}_accuracy": total_top[k] / total_samples for k in (1, 2, 3, 4, 5)},
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, pred_labels)),
        "precision": {
            "macro": float(np.mean(precision)),
            "weighted": float(np.average(precision, weights=support_arr)),
        },
        "recall": {
            "macro": float(np.mean(recall)),
            "weighted": float(np.average(recall, weights=support_arr)),
        },
        "f1_score": {
            "macro": float(np.mean(f1)),
            "weighted": float(np.average(f1, weights=support_arr)),
        },
        "predictions_csv": str(output_csv),
    }


def copy_source_refs(project_root: Path, conf: dict[str, Any], run_dir: Path) -> None:
    source_refs = run_dir / "source_refs"
    source_refs.mkdir(parents=True, exist_ok=True)
    support = conf["supporting_data"]
    for key in ["class_index", "target_mapping", "data_policy", "split_policy", "source_refs"]:
        src = resolve_path(project_root, support[key])
        if src.exists():
            shutil.copy2(src, source_refs / src.name)
    write_json(source_refs / "run_config.json", conf)


def write_failure_report(run_dir: Path, stage: str, error: BaseException) -> None:
    write_json(
        run_dir / "logs" / f"{stage}_failure.json",
        {
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now(TAIPEI).strftime("%Y-%m-%d_%H%M%S"),
        },
    )


def main() -> None:
    args = parse_args()
    conf = load_yaml(args.config)
    validate_config(conf)
    if args.dry_run:
        print("Config validated only. No data, runtime, model, or GPU checks performed.")
        return
    if not args.allow_local_training:
        raise ValueError("Training is opt-in: review README and pass --allow-local-training.")
    project_root = Path(conf["general"]["project_root"])
    config_sha256 = sha256_file(args.config)
    if args.resume_from is not None and args.run_dir is None:
        raise ValueError("--resume-from requires --run-dir so resume writes into the original run directory.")
    run_group = f"{conf['experiment']['round']}_{conf['experiment']['stage']}"
    if args.run_dir is None:
        run_dir = (
            resolve_path(project_root, conf["general"]["run_root"])
            / run_group
            / f"{timestamp()}_{conf['training']['run_name']}"
        )
    else:
        run_dir = resolve_path(project_root, args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=args.resume_from is not None)
    setup_logging(run_dir)
    LOGGER.info("[train] Run directory: %s", run_dir)
    LOGGER.info("[train] Config: %s", args.config)

    seed_everything(int(conf["training"].get("seed", 42)))
    support = conf["supporting_data"]
    validate_manifest_checksums(project_root, conf["manifest"])

    classes = read_classes(resolve_path(project_root, support["classes"]))
    target_ids = read_classes(resolve_path(project_root, support["target_ids"]))
    if len(classes) != int(conf["model"]["num_classes"]):
        raise ValueError(f"Config num_classes={conf['model']['num_classes']} but classes.txt has {len(classes)}")
    if len(classes) != len(target_ids):
        raise ValueError("classes.txt and target_ids.txt length mismatch")

    train_rows = read_split_txt(resolve_path(project_root, support["train_split"]))
    val_rows = read_split_txt(resolve_path(project_root, support["val_split"]))
    LOGGER.info("[data] train=%d val=%d classes=%d", len(train_rows), len(val_rows), len(classes))

    copy_source_refs(project_root, conf, run_dir)
    conf_dir = run_dir / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, conf_dir / "config.yaml")
    write_json(conf_dir / "conf.json", conf)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Phase 4 round 0 stage 1 training requires CUDA.")
    LOGGER.info("[runtime] Device: %s | %s", device, torch.cuda.get_device_name(0))
    probe = torch.ones((1,), device=device) * 2
    LOGGER.info("[runtime] CUDA probe result: %.1f", float(probe.item()))

    creation_kwargs = model_kwargs(conf)
    if creation_kwargs:
        LOGGER.info("[model] Extra timm create_model kwargs: %s", creation_kwargs)
    model = timm.create_model(
        conf["model"]["modelname"],
        pretrained=bool(conf["model"].get("pretrained", False)),
        num_classes=int(conf["model"]["num_classes"]),
        **creation_kwargs,
    )
    model.to(device)
    pretrained_checkpoint = conf["model"].get("pretrained_checkpoint")
    if pretrained_checkpoint:
        load_pretrained_encoder_checkpoint(model, resolve_path(project_root, pretrained_checkpoint))
    train_transform, eval_transform, ten_crop_transform = make_transforms(model, conf)

    ema_conf = conf["training"].get("ema", {})
    ema_enabled = bool(ema_conf.get("enabled", False))
    ema_decay = float(ema_conf.get("decay", 0.9998))
    ema_validate = bool(ema_conf.get("validate", ema_enabled))
    ema_primary = bool(ema_conf.get("primary", ema_enabled))
    model_ema = None
    if ema_enabled:
        model_ema = ModelEma(model, decay=ema_decay)
        model_ema.module.to(device)
        LOGGER.info(
            "[train] EMA enabled decay=%.6f validate=%s primary=%s",
            ema_decay,
            ema_validate,
            ema_primary,
        )

    model = maybe_compile_model(model, conf)

    batch_size = int(conf["training"]["batch_size"])
    num_workers = int(conf["dataset"].get("num_workers", 8))
    sampler_conf = conf["training"].get("sampler", {})
    train_sampler = None
    if bool(sampler_conf.get("enabled", False)):
        sampler_mode = sampler_conf.get("mode")
        if sampler_mode != "class_count_cap":
            raise ValueError(f"Unsupported sampler mode: {sampler_mode}")
        sampler_cap = int(sampler_conf["class_count_cap"])
        sampler_weights = class_count_cap_sample_weights(
            train_rows,
            num_classes=int(conf["model"]["num_classes"]),
            cap=sampler_cap,
        )
        train_sampler = WeightedRandomSampler(
            weights=sampler_weights,
            num_samples=len(train_rows),
            replacement=bool(sampler_conf.get("replacement", True)),
            generator=torch.Generator().manual_seed(int(conf["training"].get("seed", 42))),
        )
        weight_array = sampler_weights.numpy()
        write_json(
            run_dir / "source_refs" / "sampler.json",
            {
                "enabled": True,
                "mode": sampler_mode,
                "class_count_cap": sampler_cap,
                "replacement": bool(sampler_conf.get("replacement", True)),
                "min_sample_weight": float(weight_array.min()),
                "max_sample_weight": float(weight_array.max()),
                "mean_sample_weight": float(weight_array.mean()),
            },
        )
        LOGGER.info(
            "[train] Sampler enabled mode=%s class_count_cap=%d min_weight=%.4f max_weight=%.4f",
            sampler_mode,
            sampler_cap,
            float(weight_array.min()),
            float(weight_array.max()),
        )
    train_loader = DataLoader(
        ImageSplitDataset(train_rows, train_transform),
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        ImageSplitDataset(val_rows, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    class_weight_tensor = None
    if conf["training"].get("use_class_weights", False):
        class_weight_tensor = compute_class_weights(
            train_rows,
            num_classes=int(conf["model"]["num_classes"]),
            mode=conf["training"].get("class_weight_mode", "panaiotis_relative"),
            exponent=float(conf["training"].get("class_weight_exponent", 0.22)),
        ).to(device)
        positive_weights = class_weight_tensor[class_weight_tensor > 0]
        write_json(
            run_dir / "source_refs" / "class_weights.json",
            {
                "mode": conf["training"].get("class_weight_mode", "panaiotis_relative"),
                "exponent": float(conf["training"].get("class_weight_exponent", 0.22)),
                "min": float(positive_weights.min().item()),
                "max": float(positive_weights.max().item()),
                "mean": float(positive_weights.mean().item()),
                "weights": [float(value) for value in class_weight_tensor.cpu().tolist()],
            },
        )
        LOGGER.info(
            "[train] Class weights enabled: min=%.4f max=%.4f",
            float(positive_weights.min().item()),
            float(positive_weights.max().item()),
        )

    feature_dim = int(getattr(original_model(model), "num_features", conf["model"].get("feature_dim", 0)))
    if feature_dim <= 0:
        raise ValueError("Could not determine model feature dimension for training criterion")
    criterion = build_train_criterion(conf, class_weight_tensor, feature_dim).to(device)
    eval_criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor,
        label_smoothing=float(conf["training"].get("label_smoothing", 0.0)),
    ).to(device)
    criterion_params = [parameter for parameter in criterion.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(
        list(model.parameters()) + criterion_params,
        lr=float(conf["training"]["initial_lr"]),
        weight_decay=float(conf["training"].get("weight_decay", 0.05)),
    )
    epochs = int(conf["training"]["epochs"])
    max_train_steps = configured_step_limit(conf, "max_train_steps_per_epoch")
    max_val_steps = configured_step_limit(conf, "max_val_steps_per_epoch")
    effective_train_steps = effective_step_count(train_loader, max_train_steps)
    LOGGER.info(
        "[train] epochs=%d train_steps_per_epoch=%d/%d val_steps_per_epoch=%s/%d",
        epochs,
        effective_train_steps,
        len(train_loader),
        "all" if max_val_steps is None else str(effective_step_count(val_loader, max_val_steps)),
        len(val_loader),
    )
    scheduler = build_scheduler(
        optimizer,
        epochs=epochs,
        steps_per_epoch=effective_train_steps,
        warmup_epochs=int(conf["training"].get("warmup_epochs", 0)),
    )
    use_amp = bool(conf["training"].get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    heartbeat = heartbeat_config(conf)

    wandb_run = None
    ckpt_dir = run_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = run_dir / "logs" / "epoch_metrics.csv"
    checkpoint_metric = checkpoint_metric_name(conf)
    best_metric_value = -1.0
    best_val_accuracy = -1.0
    best_epoch = 0
    start_epoch = 1
    start_time = time.time()
    LOGGER.info("[train] Checkpoint metric: %s", checkpoint_metric)
    if args.resume_from is not None:
        resume_path = resolve_path(project_root, args.resume_from)
        checkpoint = load_last_checkpoint(
            resume_path,
            expected_config_sha256=config_sha256,
            expected_num_classes=len(classes),
        )
        start_epoch = restore_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            model_ema=model_ema,
            criterion=criterion,
        )
        restore_rng_state(checkpoint.get("rng_state", {}))
        best_validation_metric = checkpoint["best_validation_metric"]
        best_metric_value = float(best_validation_metric["value"])
        best_val_accuracy = float(checkpoint["best_val_accuracy"])
        best_epoch = int(checkpoint["best_epoch"])
        if start_epoch > epochs:
            raise ValueError(f"Resume checkpoint already completed epoch {start_epoch - 1}; configured epochs={epochs}")
        LOGGER.info(
            "[resume] Loaded %s completed_epoch=%d next_epoch=%d best_%s=%.6f",
            resume_path,
            start_epoch - 1,
            start_epoch,
            checkpoint_metric,
            best_metric_value,
        )

    try:
        metrics_mode = "a" if start_epoch > 1 and metrics_csv.exists() else "w"
        with metrics_csv.open(metrics_mode, newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_accuracy",
                    "val_loss",
                    "val_accuracy",
                    "val_balanced_accuracy",
                    "val_macro_f1",
                    "val_weighted_f1",
                    "raw_val_loss",
                    "raw_val_accuracy",
                    "raw_val_balanced_accuracy",
                    "raw_val_macro_f1",
                    "raw_val_weighted_f1",
                    "lr",
                    "seconds",
                ],
            )
            if metrics_mode == "w":
                writer.writeheader()
            for epoch in range(start_epoch - 1, epochs):
                epoch_number = epoch + 1
                epoch_start = time.time()
                train_metrics = train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    scheduler,
                    device,
                    scaler,
                    use_amp,
                    epoch=epoch_number,
                    total_epochs=epochs,
                    run_start_time=start_time,
                    heartbeat=heartbeat,
                    wandb_run=wandb_run,
                    model_ema=model_ema,
                    max_steps=max_train_steps,
                    effective_steps_per_epoch=effective_train_steps,
                )
                raw_val_metrics = evaluate(
                    model,
                    val_loader,
                    eval_criterion,
                    device,
                    use_amp,
                    max_steps=max_val_steps,
                    num_classes=len(classes),
                )
                if ema_validate and model_ema is not None:
                    val_metrics = evaluate(
                        model_ema.module,
                        val_loader,
                        eval_criterion,
                        device,
                        use_amp,
                        max_steps=max_val_steps,
                        num_classes=len(classes),
                    )
                else:
                    val_metrics = raw_val_metrics
                elapsed = time.time() - epoch_start
                row = {
                    "epoch": epoch_number,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_weighted_f1": val_metrics["weighted_f1"],
                    "raw_val_loss": raw_val_metrics["loss"],
                    "raw_val_accuracy": raw_val_metrics["accuracy"],
                    "raw_val_balanced_accuracy": raw_val_metrics["balanced_accuracy"],
                    "raw_val_macro_f1": raw_val_metrics["macro_f1"],
                    "raw_val_weighted_f1": raw_val_metrics["weighted_f1"],
                    "lr": train_metrics["lr"],
                    "seconds": elapsed,
                }
                writer.writerow(row)
                handle.flush()
                LOGGER.info(
                    "[epoch %03d/%03d] train_acc=%.5f val_acc=%.5f val_bal=%.5f val_macro_f1=%.5f train_loss=%.5f val_loss=%.5f lr=%.8f sec=%.1f",
                    epoch_number,
                    epochs,
                    train_metrics["accuracy"],
                    val_metrics["accuracy"],
                    val_metrics["balanced_accuracy"],
                    val_metrics["macro_f1"],
                    train_metrics["loss"],
                    val_metrics["loss"],
                    train_metrics["lr"],
                    elapsed,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "epoch": epoch_number,
                            "train_step": epoch_number * len(train_loader),
                            "effective_train_step": epoch_number * effective_train_steps,
                            "epoch_progress": epoch_number,
                            "train/loss": train_metrics["loss"],
                            "train/accuracy": train_metrics["accuracy"],
                            "val/loss": val_metrics["loss"],
                            "val/accuracy": val_metrics["accuracy"],
                            "val/balanced_accuracy": val_metrics["balanced_accuracy"],
                            "val/macro_f1": val_metrics["macro_f1"],
                            "val/weighted_f1": val_metrics["weighted_f1"],
                            "raw_val/loss": raw_val_metrics["loss"],
                            "raw_val/accuracy": raw_val_metrics["accuracy"],
                            "raw_val/balanced_accuracy": raw_val_metrics["balanced_accuracy"],
                            "raw_val/macro_f1": raw_val_metrics["macro_f1"],
                            "raw_val/weighted_f1": raw_val_metrics["weighted_f1"],
                            "lr": train_metrics["lr"],
                            "epoch_seconds": elapsed,
                        }
                    )
                candidate_value = float(row[checkpoint_metric])
                if candidate_value > best_metric_value:
                    best_metric_value = candidate_value
                    best_val_accuracy = float(row["val_accuracy"])
                    best_epoch = epoch_number
                    best_state_source = (
                        model_ema.module
                        if ema_primary and model_ema is not None
                        else original_model(model)
                    )
                    torch.save(
                        {
                            "model": best_state_source.state_dict(),
                            "conf": conf,
                            "epoch": best_epoch,
                            "best_val_accuracy": best_val_accuracy,
                            "best_validation_metric": {
                                "metric": checkpoint_metric,
                                "value": best_metric_value,
                            },
                            "best_val_metrics": val_metrics,
                            "raw_val_metrics": raw_val_metrics,
                            "checkpoint_source": "ema" if best_state_source is getattr(model_ema, "module", None) else "raw",
                            "class_names": classes,
                            "target_ids": target_ids,
                        },
                        ckpt_dir / "best_model.pt",
                    )
                save_last_checkpoint(
                    ckpt_dir / "last_checkpoint.pt",
                    ResumeState(
                        model=original_model(model).state_dict(),
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),
                        scaler=scaler.state_dict(),
                        ema=model_ema.module.state_dict() if model_ema is not None else None,
                        completed_epoch=epoch_number,
                        best_metric_value=best_metric_value,
                        best_val_accuracy=best_val_accuracy,
                        best_epoch=best_epoch,
                        checkpoint_metric=checkpoint_metric,
                        config_sha256=config_sha256,
                        num_classes=len(classes),
                        class_names=classes,
                        target_ids=target_ids,
                        rng_state=collect_rng_state(),
                        criterion=criterion.state_dict() if criterion.state_dict() else None,
                    ),
                )

        torch.save(
            {
                "model": original_model(model).state_dict(),
                "conf": conf,
                "class_names": classes,
                "target_ids": target_ids,
                "checkpoint_source": "raw",
            },
            ckpt_dir / "final_model.pt",
        )
        if model_ema is not None:
            torch.save(
                {
                    "model": model_ema.module.state_dict(),
                    "conf": conf,
                    "class_names": classes,
                    "target_ids": target_ids,
                    "checkpoint_source": "ema",
                },
                ckpt_dir / "final_ema_model.pt",
            )
    except BaseException as exc:
        write_failure_report(run_dir, "training", exc)
        if wandb_run is not None:
            wandb_run.summary["training_failed"] = True
            wandb_run.summary["training_error"] = str(exc)
        raise

    training_time = time.time() - start_time
    write_json(
        run_dir / "source_refs" / "training_stats.json",
        {
            "training_time_seconds": training_time,
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": best_epoch,
            "epochs": epochs,
            "best_validation_metric": {
                "metric": checkpoint_metric,
                "value": best_metric_value,
                "epoch": best_epoch,
                "epoch_numbering": "1-based",
            },
            "checkpoint_selection": {
                "metric": checkpoint_metric,
                "source": "ema" if ema_primary and model_ema is not None else "raw",
                "ema_enabled": model_ema is not None,
                "ema_decay": ema_decay if model_ema is not None else None,
            },
            "heartbeat": {
                "enabled": heartbeat.enabled,
                "step_interval": heartbeat.step_interval,
                "time_interval_seconds": heartbeat.time_interval_seconds,
            },
            "max_train_steps_per_epoch": max_train_steps,
            "max_val_steps_per_epoch": max_val_steps,
            "effective_train_steps_per_epoch": effective_train_steps,
        },
    )

    if conf["training"].get("use_test", True):
        try:
            test_transform = ten_crop_transform if conf["training"].get("test_tta") == "ten_crop" else eval_transform
            test_rows = read_manifest_split(resolve_path(project_root, support["split_manifest"]), "test")
            test_num_workers = (
                int(args.test_num_workers)
                if args.test_num_workers is not None
                else int(conf["dataset"].get("test_num_workers", num_workers))
            )
            test_loader = DataLoader(
                ManifestSplitDataset(test_rows, test_transform),
                batch_size=batch_size,
                shuffle=False,
                num_workers=test_num_workers,
                pin_memory=True,
                persistent_workers=test_num_workers > 0,
            )
            checkpoint = torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=False)
            original_model(model).load_state_dict(checkpoint["model"])
            metrics = predict_test(
                model,
                test_loader,
                device,
                classes,
                target_ids,
                use_amp,
                run_dir / "predictions" / "best_model.pt+test+predictions.csv",
            )
            write_json(run_dir / "predictions" / "best_model.pt+test+metrics.json", metrics)
            if wandb_run is not None:
                wandb_run.summary["test_top1_accuracy"] = metrics["topk_accuracy"]["top1_accuracy"]
                wandb_run.summary["test_top3_accuracy"] = metrics["topk_accuracy"]["top3_accuracy"]
                wandb_run.summary["test_top5_accuracy"] = metrics["topk_accuracy"]["top5_accuracy"]
                wandb_run.summary["test_balanced_accuracy"] = metrics["balanced_accuracy"]
                wandb_run.summary["test_macro_f1"] = metrics["f1_score"]["macro"]
                wandb_run.summary["test_weighted_f1"] = metrics["f1_score"]["weighted"]
        except BaseException as exc:
            write_failure_report(run_dir, "test_evaluation", exc)
            if wandb_run is not None:
                wandb_run.summary["test_evaluation_failed"] = True
                wandb_run.summary["test_evaluation_error"] = str(exc)
            raise

    if wandb_run is not None:
        wandb_run.summary["best_val_accuracy"] = best_val_accuracy
        wandb_run.summary["best_validation_metric"] = checkpoint_metric
        wandb_run.summary["best_validation_metric_value"] = best_metric_value
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["best_checkpoint_source"] = (
            "ema" if ema_primary and model_ema is not None else "raw"
        )
        wandb_run.finish()

    LOGGER.info(
        "[done] run_dir=%s best_%s=%.6f best_epoch=%d",
        run_dir,
        checkpoint_metric,
        best_metric_value,
        best_epoch,
    )


if __name__ == "__main__":
    main()
