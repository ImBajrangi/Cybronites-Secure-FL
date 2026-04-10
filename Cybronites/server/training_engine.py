import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import os
import re
import threading
import time
import logging
import sys
import numpy as np

logger = logging.getLogger("TrainingEngine")


_DISPLAY_PATTERNS = [
    (r'plt\.show\s*\(.*?\)',           'pass  # [sanitized: plt.show]'),
    (r'plt\.savefig\s*\(.*?\)',        'pass  # [sanitized: plt.savefig]'),
    (r'\.show\s*\(\s*\)',              'pass  # [sanitized: .show()]'),
    (r'cv2\.imshow\s*\(.*?\)',         'pass  # [sanitized: cv2.imshow]'),
    (r'cv2\.waitKey\s*\(.*?\)',        'pass  # [sanitized: cv2.waitKey]'),
    (r'cv2\.destroyAllWindows\s*\(.*?\)', 'pass  # [sanitized: cv2.destroyAllWindows]'),
    (r'display\s*\(.*?\)',             'pass  # [sanitized: display()]'),
    (r'\.mainloop\s*\(\s*\)',          'pass  # [sanitized: .mainloop()]'),
    (r'input\s*\(.*?\)',               '"user_input"  # [sanitized: input()]'),
]

_HEADLESS_PREAMBLE = """
import os as _os
_os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
"""


def sanitize_code(code: str) -> str:
    sanitized = code
    for pattern, replacement in _DISPLAY_PATTERNS:
        sanitized = re.sub(pattern, replacement, sanitized)
    if 'matplotlib' in sanitized or 'plt' in sanitized:
        sanitized = _HEADLESS_PREAMBLE + sanitized
    return sanitized


def extract_error_line(tb_text: str) -> int:
    match = re.search(r'File "<laboratory>", line (\d+)', tb_text)
    if match:
        return int(match.group(1))
    return None


class TrainingSession:
    def __init__(self, code, hyperparams, bridge_broadcast_callback):
        self.original_code = code
        self.code = sanitize_code(code)
        self.epochs = hyperparams.get("epochs", 5)
        self.lr = hyperparams.get("lr", 0.01)
        self.batch_size = hyperparams.get("batch_size", 32)
        self.broadcast = bridge_broadcast_callback
        self.stop_event = threading.Event()
        self.model = None
        self.status = "IDLE"
        self.progress = 0
        self.mode = "FEDERATED"
        self.metrics = {"loss": [], "accuracy": []}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = None
        self._vault_data = None

    def run(self):
        from Cybronites.server.vault_loader import VaultLoader, wipe_tracked_buffers

        vault_instance = VaultLoader()

        try:
            self.status = "TRAINING"
            self.broadcast("LOG", f"SYSTEM: Starting execution on {self.device}...")

            namespace = {
                '__builtins__': __builtins__,
                'print': self._safe_print,
                'vault': vault_instance,
            }

            old_stdout = sys.stdout
            captured = _OutputCapture(self._safe_print)
            sys.stdout = captured

            try:
                exec(self.code, namespace)
            finally:
                sys.stdout = old_stdout

            model_class = None
            for name, obj in namespace.items():
                if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
                    model_class = obj
                    break

            vault_dataset = namespace.get("_vault_dataset", None)
            vault_labels = namespace.get("_vault_labels", None)
            vault_info = namespace.get("_vault_info", None)

            if not model_class:
                self.mode = "SCRIPT"
                self.status = "COMPLETE"
                self.progress = 100
                self.broadcast("LAB_PROGRESS", {
                    "epoch": 0, "total_epochs": 0, "loss": 0,
                    "accuracy": 0, "progress": 100,
                    "status": "COMPLETE", "mode": "SCRIPT"
                })
                self.broadcast("LAB_COMPLETE", {
                    "status": "COMPLETE", "mode": "SCRIPT",
                    "metrics": self.metrics
                })
                self.broadcast("LOG", "SYSTEM: Script execution complete (no model to train).")
                return

            self.mode = "FEDERATED"
            self.broadcast("LOG", f"SYSTEM: Model found ({model_class.__name__}). Starting training...")

            self.model = model_class().to(self.device)
            param_count = sum(p.numel() for p in self.model.parameters())
            self.broadcast("LOG", f"SYSTEM: Model has {param_count:,} trainable parameters.")

            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            train_loader, test_loader, ds_name = self._prepare_data(
                namespace, vault_instance, model_class
            )

            self.broadcast("LOG", f"SYSTEM: Dataset: {ds_name} (batch_size={self.batch_size})")
            self.broadcast("LOG", f"SYSTEM: Training for {self.epochs} epochs at lr={self.lr}...")

            for epoch in range(self.epochs):
                if self.stop_event.is_set():
                    self.status = "ABORTED"
                    self.broadcast("LOG", "SYSTEM: Training aborted by user.")
                    self.broadcast("LAB_PROGRESS", {"status": "ABORTED", "progress": self.progress})
                    return

                self.model.train()
                running_loss = 0.0
                correct_train = 0
                total_train = 0

                for batch_idx, (data, target) in enumerate(train_loader):
                    if self.stop_event.is_set():
                        break
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = nn.functional.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                    pred = output.argmax(dim=1)
                    correct_train += pred.eq(target).sum().item()
                    total_train += len(data)

                self.model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.model(data)
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += len(data)

                accuracy = correct / total if total > 0 else 0
                avg_loss = running_loss / max(len(train_loader), 1)
                train_acc = correct_train / total_train if total_train > 0 else 0

                self.metrics["loss"].append(avg_loss)
                self.metrics["accuracy"].append(accuracy)
                self.progress = ((epoch + 1) / self.epochs) * 100

                self.broadcast("LAB_PROGRESS", {
                    "epoch": epoch + 1,
                    "total_epochs": self.epochs,
                    "loss": avg_loss,
                    "accuracy": accuracy,
                    "train_accuracy": train_acc,
                    "progress": self.progress,
                    "status": "TRAINING",
                    "mode": "FEDERATED"
                })

                self.broadcast("LOG",
                    f"Epoch {epoch+1}/{self.epochs} — "
                    f"Loss: {avg_loss:.4f} | "
                    f"Train Acc: {train_acc:.2%} | "
                    f"Test Acc: {accuracy:.2%}"
                )

            save_dir = os.path.join(os.getcwd(), "exports")
            os.makedirs(save_dir, exist_ok=True)

            timestamp = int(time.time())
            self.model_path = os.path.join(save_dir, f"model_{timestamp}.pt")
            torch.save(self.model.state_dict(), self.model_path)
            self.broadcast("LOG", f"SYSTEM: Model weights saved → {self.model_path}")

            onnx_path = os.path.join(save_dir, f"model_{timestamp}.onnx")
            try:
                sample = next(iter(train_loader))[0][:1].to(self.device)
                torch.onnx.export(self.model, sample, onnx_path)
                self.broadcast("LOG", f"SYSTEM: ONNX export saved → {onnx_path}")
            except Exception as e:
                logger.warning(f"ONNX export failed: {e}")
                onnx_path = None

            self.status = "COMPLETE"
            self.broadcast("LAB_COMPLETE", {
                "status": "COMPLETE",
                "mode": "FEDERATED",
                "dataset": ds_name,
                "pt_path": self.model_path,
                "onnx_path": onnx_path,
                "metrics": self.metrics,
                "final_accuracy": self.metrics["accuracy"][-1] if self.metrics["accuracy"] else 0,
                "final_loss": self.metrics["loss"][-1] if self.metrics["loss"] else 0,
            })

            final_acc = self.metrics["accuracy"][-1] if self.metrics["accuracy"] else 0
            self.broadcast("LOG",
                f"SYSTEM: ✅ Training complete! "
                f"Final accuracy: {final_acc:.2%} | "
                f"Model exported as .pt and .onnx"
            )

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.status = "ERROR"
            error_line = extract_error_line(tb)
            error_msg = str(e)

            logger.error(f"Training Error: {error_msg}\n{tb}")
            self.broadcast("LAB_ERROR", {
                "error": error_msg,
                "line": error_line,
                "traceback": tb
            })
            self.broadcast("LOG", f"FATAL: {error_msg}")

        finally:
            wipe_tracked_buffers()
            self.broadcast("LOG", "SYSTEM: 🔐 Vault decrypted data wiped from memory.")
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _prepare_data(self, namespace, vault_instance, model_class):
        """Detect data source: vault dataset, user-provided, or fallback to MNIST."""

        # Check if user called vault.load() and stored results
        # Common patterns: data, labels, info = vault.load("X")
        # Or: X, y, info = vault.load_torch("X")
        vault_tensors_x = None
        vault_tensors_y = None
        ds_name = "MNIST (default)"

        for var_name, val in namespace.items():
            if var_name.startswith("_"):
                continue
            if isinstance(val, torch.Tensor) and val.dtype == torch.float32 and val.dim() >= 2:
                if vault_tensors_x is None:
                    vault_tensors_x = val
                    ds_name = f"Vault ({var_name})"
            if isinstance(val, torch.Tensor) and val.dtype == torch.long and val.dim() == 1:
                if vault_tensors_y is None:
                    vault_tensors_y = val

        if vault_tensors_x is None:
            for var_name, val in namespace.items():
                if var_name.startswith("_"):
                    continue
                if isinstance(val, np.ndarray) and val.dtype in (np.float32, np.float64) and val.ndim >= 2:
                    vault_tensors_x = torch.tensor(val, dtype=torch.float32)
                    ds_name = f"Vault ({var_name})"
                    break
            for var_name, val in namespace.items():
                if var_name.startswith("_"):
                    continue
                if isinstance(val, np.ndarray) and val.ndim == 1:
                    if val.dtype in (np.int32, np.int64, np.float64):
                        vault_tensors_y = torch.tensor(val, dtype=torch.long)
                        break

        if vault_tensors_x is not None and vault_tensors_y is not None:
            self.broadcast("LOG", f"SYSTEM: 🔐 Using vault-decrypted data: {vault_tensors_x.shape}")

            if vault_tensors_x.dim() == 2:
                vault_tensors_x = vault_tensors_x.unsqueeze(1)

            n = len(vault_tensors_x)
            split = int(0.8 * n)
            train_ds = TensorDataset(vault_tensors_x[:split], vault_tensors_y[:split])
            test_ds = TensorDataset(vault_tensors_x[split:], vault_tensors_y[split:])

            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=min(1000, n - split), shuffle=False)
            return train_loader, test_loader, ds_name

        self.broadcast("LOG", f"SYSTEM: Loading MNIST dataset (batch_size={self.batch_size})...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

        return train_loader, test_loader, ds_name

    def _safe_print(self, *args, **kwargs):
        msg = ' '.join(str(a) for a in args)
        self.broadcast("LOG", f"[stdout] {msg}")

    def abort(self):
        self.stop_event.set()


class _OutputCapture:
    def __init__(self, callback):
        self._callback = callback
    def write(self, text):
        text = text.strip()
        if text:
            self._callback(text)
    def flush(self):
        pass


_current_session = None

def start_training(code, hyperparams, broadcast_callback):
    global _current_session
    if _current_session and _current_session.status == "TRAINING":
        return False, "A training session is already in progress."

    _current_session = TrainingSession(code, hyperparams, broadcast_callback)
    thread = threading.Thread(target=_current_session.run, daemon=True)
    thread.start()
    return True, "Training started."

def abort_training():
    global _current_session
    if _current_session:
        _current_session.abort()
        return True
    return False

def get_session_status():
    if _current_session:
        return {
            "status": _current_session.status,
            "progress": _current_session.progress,
            "mode": _current_session.mode,
            "metrics": _current_session.metrics
        }
    return {"status": "IDLE"}
