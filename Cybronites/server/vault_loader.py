"""
Vault Loader — Bridge between Code Laboratory and Privacy Vault.

Provides a `vault` object that users can use in their Lab code:
    data, labels, info = vault.load("Iris")
    print(vault.list())

All decryption happens in RAM. Data is tracked for secure wipe after execution.
"""

import pickle
import logging
import numpy as np

logger = logging.getLogger("VaultLoader")

_tracked_buffers = []


class VaultLoader:
    """Injected into the Code Laboratory namespace as `vault`."""

    def __init__(self):
        self._vault = None
        self._initialized = False
        self._datasets_cache = None

    def _ensure_init(self):
        if self._initialized:
            return
        try:
            from secure_training_platform.database.db import init_db
            from secure_training_platform.key_manager.manager import KeyManager
            from secure_training_platform.dataset_vault.vault import DatasetVault
            init_db()
            km = KeyManager()
            self._vault = DatasetVault(km)
            self._initialized = True
        except Exception as e:
            logger.warning(f"Vault init failed: {e}")
            raise RuntimeError(
                "Privacy Vault is not available. "
                "Make sure the Secure Training Platform is configured."
            )

    def list(self):
        """List all available encrypted datasets.
        
        Returns:
            list of dicts with name, description, num_samples, num_classes, input_shape
        """
        self._ensure_init()
        datasets = self._vault.list_datasets()
        return [
            {
                "name": d["name"],
                "description": d["description"],
                "samples": d["num_samples"],
                "classes": d["num_classes"],
                "shape": d["input_shape"],
            }
            for d in datasets
        ]

    def load(self, name: str):
        """Load an encrypted dataset by name. Decrypts in RAM only.

        Args:
            name: Dataset name (e.g. "Iris", "Digits", "Wine")

        Returns:
            tuple: (data: np.ndarray, labels: np.ndarray, info: dict)

        Example:
            data, labels, info = vault.load("Iris")
            print(f"Loaded {info['name']}: {data.shape}")
        """
        self._ensure_init()

        datasets = self._vault.list_datasets()
        match = None
        name_lower = name.lower().strip()
        for d in datasets:
            if d["name"].lower() == name_lower:
                match = d
                break

        if not match:
            available = [d["name"] for d in datasets]
            raise ValueError(
                f"Dataset '{name}' not found in vault. "
                f"Available: {available}"
            )

        dataset_id = match["id"]
        logger.info(f"Decrypting '{match['name']}' (ID: {dataset_id[:8]}...) to RAM...")

        buf = self._vault.decrypt_dataset_to_memory(dataset_id, requester="laboratory")
        _tracked_buffers.append(buf)

        payload = pickle.loads(buf.read())

        data = np.array(payload["data"], dtype=np.float32)
        labels = np.array(payload["labels"])

        info = {
            "name": match["name"],
            "description": match.get("description", ""),
            "num_samples": match["num_samples"],
            "num_classes": match["num_classes"],
            "input_shape": match["input_shape"],
        }
        if "feature_names" in payload:
            info["feature_names"] = payload["feature_names"]
        if "target_names" in payload:
            info["target_names"] = payload["target_names"]

        logger.info(
            f"Loaded '{match['name']}': data={data.shape}, labels={labels.shape}"
        )
        return data, labels, info

    def load_torch(self, name: str):
        """Load dataset and return as PyTorch tensors.

        Returns:
            tuple: (X: torch.Tensor, y: torch.Tensor, info: dict)
        """
        import torch

        data, labels, info = self.load(name)
        X = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        return X, y, info

    def __repr__(self):
        try:
            self._ensure_init()
            ds = self._vault.list_datasets()
            names = [d["name"] for d in ds]
            return f"<VaultLoader: {len(names)} datasets — {names}>"
        except Exception:
            return "<VaultLoader: not connected>"


def wipe_tracked_buffers():
    """Securely wipe all decrypted buffers after Lab execution."""
    try:
        from secure_training_platform.training_worker.secure_memory import (
            secure_wipe_buffer,
        )
        for buf in _tracked_buffers:
            secure_wipe_buffer(buf)
    except Exception:
        for buf in _tracked_buffers:
            try:
                buf.close()
            except Exception:
                pass
    _tracked_buffers.clear()
    logger.info("All vault decrypted buffers wiped from memory.")
