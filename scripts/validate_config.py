"""Static configuration checks. This module never imports ML frameworks or opens data."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml


def validate_config(conf: dict) -> None:
    if not isinstance(conf, dict):
        raise ValueError("Configuration must be a mapping.")
    model = conf.get("model", {})
    if not re.fullmatch(r"[A-Za-z0-9_.]+", str(model.get("modelname", ""))):
        raise ValueError("Use a local timm registry name, not a remote model reference.")
    if model.get("extra_kwargs"):
        raise ValueError("Arbitrary model kwargs are disabled in this source-only candidate.")
    if conf.get("model", {}).get("pretrained", False) is not False:
        raise ValueError("Pretrained downloads are disabled in this source-only candidate.")
    if conf.get("model", {}).get("pretrained_checkpoint"):
        raise ValueError("External pretrained checkpoints are disabled in this candidate.")
    if conf.get("monitor", {}).get("use_wandb", False):
        raise ValueError("External experiment tracking is disabled in this candidate.")
    if int(conf.get("model", {}).get("num_classes", 0)) < 5:
        raise ValueError("This retained top-5 pipeline requires at least five classes.")
    manifest = conf.get("manifest", {})
    for item in [manifest, *manifest.get("extra_manifests", [])]:
        if not item.get("path") or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))):
            raise ValueError("Supply each manifest path and its actual lowercase SHA-256.")
    for key in ("classes", "target_ids", "train_split", "val_split", "split_manifest"):
        if not conf.get("supporting_data", {}).get(key):
            raise ValueError(f"Missing supporting_data.{key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    validate_config(yaml.safe_load(args.config.read_text()))
    print(json.dumps({"config_only": True, "data_opened": False, "model_created": False}))


if __name__ == "__main__":
    main()
