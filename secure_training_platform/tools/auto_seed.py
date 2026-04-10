"""
Auto-Seed — Provisions small built-in datasets on startup.
Uses sklearn bundled datasets (no download, instant).
Each dataset is serialized, encrypted with AES-256-GCM, and registered in the vault.
"""

import pickle
import logging
import numpy as np

from secure_training_platform.dataset_vault.vault import DatasetVault
from secure_training_platform.config import SUPPORTED_MODELS

logger = logging.getLogger(__name__)

BUILTIN_DATASETS = [
    {
        "name": "Iris",
        "description": "Classic 3-class flower classification. 150 samples, 4 features (sepal/petal length & width).",
        "loader": "load_iris",
        "num_classes": 3,
        "input_shape": [4],
    },
    {
        "name": "Wine",
        "description": "Chemical analysis of 3 wine cultivars. 178 samples, 13 features.",
        "loader": "load_wine",
        "num_classes": 3,
        "input_shape": [13],
    },
    {
        "name": "Breast Cancer Wisconsin",
        "description": "Binary classification of malignant vs benign tumors. 569 samples, 30 features.",
        "loader": "load_breast_cancer",
        "num_classes": 2,
        "input_shape": [30],
    },
    {
        "name": "Digits",
        "description": "Handwritten digit recognition (0-9). 1797 samples, 8x8 grayscale images.",
        "loader": "load_digits",
        "num_classes": 10,
        "input_shape": [1, 8, 8],
    },
    {
        "name": "Diabetes",
        "description": "Regression: disease progression prediction. 442 samples, 10 baseline variables.",
        "loader": "load_diabetes",
        "num_classes": 0,
        "input_shape": [10],
        "task": "regression",
    },
    {
        "name": "Linnerud",
        "description": "Multivariate regression: 3 exercise variables predict 3 physiological variables. 20 samples.",
        "loader": "load_linnerud",
        "num_classes": 0,
        "input_shape": [3],
        "task": "regression",
    },
]


def _load_sklearn_dataset(loader_name: str) -> tuple[bytes, int]:
    from sklearn import datasets as sk_datasets

    loader_fn = getattr(sk_datasets, loader_name)
    ds = loader_fn()

    data = np.array(ds.data, dtype=np.float32)
    if hasattr(ds, "target"):
        labels = np.array(ds.target)
    else:
        labels = np.zeros(len(data), dtype=np.int64)

    num_samples = len(data)
    payload = {"data": data, "labels": labels}
    if hasattr(ds, "feature_names"):
        payload["feature_names"] = list(ds.feature_names)
    if hasattr(ds, "target_names"):
        payload["target_names"] = list(ds.target_names)

    serialized = pickle.dumps(payload)
    return serialized, num_samples


def seed_builtin_datasets(vault: DatasetVault):
    existing = {d["name"] for d in vault.list_datasets()}
    seeded = 0

    logger.info("━" * 50)
    logger.info("  Auto-Seed: Provisioning built-in datasets")
    logger.info("━" * 50)

    for ds_cfg in BUILTIN_DATASETS:
        name = ds_cfg["name"]

        if name in existing:
            logger.info(f"  ⏭  {name} — already encrypted, skipping")
            continue

        try:
            raw_bytes, num_samples = _load_sklearn_dataset(ds_cfg["loader"])

            dataset_id = vault.register_dataset(
                name=name,
                description=ds_cfg["description"],
                raw_data=raw_bytes,
                allowed_models=SUPPORTED_MODELS,
                num_classes=ds_cfg["num_classes"],
                input_shape=ds_cfg["input_shape"],
                num_samples=num_samples,
            )

            enc_size = len(raw_bytes)
            raw_bytes = b"\x00" * len(raw_bytes)
            del raw_bytes

            logger.info(
                f"  🔐 {name} — {num_samples} samples → "
                f"encrypted ({enc_size:,} bytes) → ID: {dataset_id[:8]}..."
            )
            seeded += 1

        except Exception as e:
            logger.warning(f"  ⚠  {name} — skipped: {e}")

    logger.info("━" * 50)
    total = len(vault.list_datasets())
    logger.info(f"  Vault: {total} datasets ({seeded} newly encrypted)")
    logger.info("━" * 50)
