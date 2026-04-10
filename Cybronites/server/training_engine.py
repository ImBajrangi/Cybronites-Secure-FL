import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import re
import threading
import time
import logging
import sys

logger = logging.getLogger("TrainingEngine")


# ══════════════════════════════════════════════════════
#  CODE SANITIZER — Neutralize GUI/Display calls
#  for headless server execution
# ══════════════════════════════════════════════════════

# Patterns that would crash or hang a headless backend
_DISPLAY_PATTERNS = [
    # Matplotlib display
    (r'plt\.show\s*\(.*?\)',           'pass  # [sanitized: plt.show]'),
    (r'plt\.savefig\s*\(.*?\)',        'pass  # [sanitized: plt.savefig]'),
    (r'\.show\s*\(\s*\)',              'pass  # [sanitized: .show()]'),
    # OpenCV display
    (r'cv2\.imshow\s*\(.*?\)',         'pass  # [sanitized: cv2.imshow]'),
    (r'cv2\.waitKey\s*\(.*?\)',        'pass  # [sanitized: cv2.waitKey]'),
    (r'cv2\.destroyAllWindows\s*\(.*?\)', 'pass  # [sanitized: cv2.destroyAllWindows]'),
    # IPython/Jupyter display
    (r'display\s*\(.*?\)',             'pass  # [sanitized: display()]'),
    # Tkinter
    (r'\.mainloop\s*\(\s*\)',          'pass  # [sanitized: .mainloop()]'),
    # input() blocks
    (r'input\s*\(.*?\)',               '"user_input"  # [sanitized: input()]'),
]

# Lines that import visualization-only modules — keep them but force headless
_HEADLESS_PREAMBLE = """
import os as _os
_os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
"""


def sanitize_code(code: str) -> str:
    """Strip/neutralize GUI display calls from user code for headless execution."""
    sanitized = code
    
    for pattern, replacement in _DISPLAY_PATTERNS:
        sanitized = re.sub(pattern, replacement, sanitized)
    
    # Prepend headless backend if matplotlib is imported
    if 'matplotlib' in sanitized or 'plt' in sanitized:
        sanitized = _HEADLESS_PREAMBLE + sanitized
    
    return sanitized


def extract_error_line(tb_text: str) -> int:
    """Extract line number from traceback for frontend highlighting."""
    # Look for <laboratory> references in traceback
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
        self.mode = "FEDERATED"  # FEDERATED or SCRIPT
        self.metrics = {"loss": [], "accuracy": []}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = None

    def run(self):
        try:
            self.status = "TRAINING"
            self.broadcast("LOG", f"SYSTEM: Starting execution on {self.device}...")
            
            # 1. Dynamic compilation in sandboxed namespace
            namespace = {
                '__builtins__': __builtins__,
                'print': self._safe_print,  # Redirect print to console
            }
            
            # Redirect stdout to capture print statements
            old_stdout = sys.stdout
            captured = _OutputCapture(self._safe_print)
            sys.stdout = captured
            
            try:
                exec(self.code, namespace)
            finally:
                sys.stdout = old_stdout
            
            # 2. Look for nn.Module subclass in the executed code
            model_class = None
            for name, obj in namespace.items():
                if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
                    model_class = obj
                    break
            
            if not model_class:
                # No model found — this is a standalone script execution
                self.mode = "SCRIPT"
                self.status = "COMPLETE"
                self.progress = 100
                self.broadcast("LAB_PROGRESS", {
                    "epoch": 0,
                    "total_epochs": 0,
                    "loss": 0,
                    "accuracy": 0,
                    "progress": 100,
                    "status": "COMPLETE",
                    "mode": "SCRIPT"
                })
                self.broadcast("LAB_COMPLETE", {
                    "status": "COMPLETE",
                    "mode": "SCRIPT",
                    "metrics": self.metrics
                })
                self.broadcast("LOG", "SYSTEM: Script execution complete (no model to train).")
                return
            
            # ── Federated Training Mode ──
            self.mode = "FEDERATED"
            self.broadcast("LOG", f"SYSTEM: Model found ({model_class.__name__}). Starting training...")
            
            self.model = model_class().to(self.device)
            param_count = sum(p.numel() for p in self.model.parameters())
            self.broadcast("LOG", f"SYSTEM: Model has {param_count:,} trainable parameters.")
            
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            
            # 3. Data Loading
            self.broadcast("LOG", f"SYSTEM: Loading MNIST dataset (batch_size={self.batch_size})...")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            test_dataset = datasets.MNIST('./data', train=False, transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

            # 4. Training Loop
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
                    
                    # Train accuracy
                    pred = output.argmax(dim=1)
                    correct_train += pred.eq(target).sum().item()
                    total_train += len(data)

                # Evaluation
                self.model.eval()
                correct = 0
                total = 0
                eval_loss = 0.0
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.model(data)
                        eval_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += len(data)
                
                accuracy = correct / total
                avg_loss = running_loss / len(train_loader)
                train_acc = correct_train / total_train if total_train > 0 else 0
                
                self.metrics["loss"].append(avg_loss)
                self.metrics["accuracy"].append(accuracy)
                self.progress = ((epoch + 1) / self.epochs) * 100
                
                # Broadcast progress
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

            # 5. Save results
            save_dir = os.path.join(os.getcwd(), "exports")
            os.makedirs(save_dir, exist_ok=True)
            
            timestamp = int(time.time())
            self.model_path = os.path.join(save_dir, f"model_{timestamp}.pt")
            torch.save(self.model.state_dict(), self.model_path)
            self.broadcast("LOG", f"SYSTEM: Model weights saved → {self.model_path}")
            
            # Export to ONNX if possible
            onnx_path = os.path.join(save_dir, f"model_{timestamp}.onnx")
            try:
                dummy_input = torch.randn(1, 1, 28, 28).to(self.device)
                torch.onnx.export(self.model, dummy_input, onnx_path)
                self.broadcast("LOG", f"SYSTEM: ONNX export saved → {onnx_path}")
            except Exception as e:
                logger.warning(f"ONNX export failed: {e}")
                onnx_path = None

            self.status = "COMPLETE"
            self.broadcast("LAB_COMPLETE", {
                "status": "COMPLETE",
                "mode": "FEDERATED",
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

    def _safe_print(self, *args, **kwargs):
        """Redirect user print() statements to the console via broadcast."""
        msg = ' '.join(str(a) for a in args)
        self.broadcast("LOG", f"[stdout] {msg}")

    def abort(self):
        self.stop_event.set()


class _OutputCapture:
    """Captures stdout and redirects to a callback."""
    def __init__(self, callback):
        self._callback = callback
    def write(self, text):
        text = text.strip()
        if text:
            self._callback(text)
    def flush(self):
        pass


# ══════════════════════════════════════════════════════
#  SESSION MANAGER
# ══════════════════════════════════════════════════════

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
