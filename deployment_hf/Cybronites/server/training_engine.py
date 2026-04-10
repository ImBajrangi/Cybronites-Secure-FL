import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import threading
# ─── System Health & Telemetry ───
import logging
import json
import os
import subprocess
import threading
import time

# 🧪 Performance Caching: Eliminate multi-second scan delays
_ENV_CACHE = {"data": None, "time": 0}
CACHE_TTL = 15 # 15-second scan lifecycle
import ast
import importlib.util
import subprocess
import contextlib
import io
import importlib
import venv
import sys
import shutil

# Resolve Matplotlib config permission warnings by using a writable path
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), 'tmp/matplotlib')
if not os.path.exists(os.environ['MPLCONFIGDIR']):
    os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

logger = logging.getLogger("TrainingEngine")

class BroadcastStream(io.StringIO):
    """Custom stream that redirects output to the dashboard in real-time."""
    def __init__(self, broadcast_callback):
        super().__init__()
        self.broadcast = broadcast_callback
        self.line_buffer = ""

    def write(self, data):
        super().write(data)
        self.line_buffer += data
        if '\n' in self.line_buffer:
            lines = self.line_buffer.split('\n')
            for line in lines[:-1]:
                # Send every line that was completed by a newline
                try:
                    self.broadcast("LOG", f"📜 {line}")
                except (BrokenPipeError, ConnectionResetError, RuntimeError):
                    pass
            self.line_buffer = lines[-1]

    def flush(self):
        super().flush()
        if self.line_buffer:
            try:
                self.broadcast("LOG", f"📜 {self.line_buffer}")
            except (BrokenPipeError, ConnectionResetError, RuntimeError):
                pass
            self.line_buffer = ""
def ensure_research_venv(broadcast_callback=None):
    """Ensure a dedicated virtual environment exists for research code."""
    sandbox_dir = os.path.join(os.getcwd(), "research_sandbox")
    if os.name == 'nt':
        python_exec = os.path.join(sandbox_dir, "Scripts", "python.exe")
    else:
        python_exec = os.path.join(sandbox_dir, "bin", "python")
        
    if not os.path.exists(python_exec):
        if broadcast_callback: 
            broadcast_callback("LOG", "🔭 INITIALIZING_RESEARCH_SANDBOX...")
            broadcast_callback("LOG", "📦 PRE_SEEDING_LIBRARIES: [torch, matplotlib, scikit-learn, torchvision]")
        
        # Create VENV
        venv.create(sandbox_dir, with_pip=True)
        
        # Pre-seed essential libraries
        try:
            subprocess.run([python_exec, "-m", "pip", "install", "torch", "matplotlib", "scikit-learn", "torchvision"], check=True, capture_output=True)
            if broadcast_callback: broadcast_callback("LOG", "✅ Sandbox environment successfully seeded.")
        except Exception as e:
            if broadcast_callback: broadcast_callback("LOG", f"⚠️ Sandbox seeding warning: {str(e)}")
            
    # Resolve site-packages path
    if os.name == 'nt':
        site_packages = os.path.join(sandbox_dir, "Lib", "site-packages")
    else:
        # Search for the site-packages in lib/pythonX.Y/site-packages
        lib_dir = os.path.join(sandbox_dir, "lib")
        if os.path.exists(lib_dir):
            py_dirs = [d for d in os.listdir(lib_dir) if d.startswith("python")]
            if py_dirs:
                site_packages = os.path.join(lib_dir, py_dirs[0], "site-packages")
            else:
                site_packages = ""
        else:
            site_packages = ""
            
    return python_exec, site_packages

def run_sandbox_command(cmd, broadcast_callback):
    """Standalone utility to execute a shell command in the research sandbox."""
    python_exec, _ = ensure_research_venv()
    
    # 🧪 VENV Contextualization
    # If the command starts with '!', strip it for the execution
    processed_cmd = cmd.strip()
    if processed_cmd.startswith('!'):
        processed_cmd = processed_cmd[1:].strip()
        
    if processed_cmd.startswith('pip'):
        processed_cmd = f'"{python_exec}" -m {processed_cmd}'
    elif processed_cmd.startswith('python'):
        processed_cmd = f'"{python_exec}" {processed_cmd[6:].strip()}'
    
    broadcast_callback("LOG", f"🐚 SANDBOX_EXEC: {cmd}")
    try:
        process = subprocess.Popen(
            processed_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        if process.stdout:
            for out_line in process.stdout:
                if out_line.strip():
                    broadcast_callback("LOG", f"🐚 {out_line.strip()}")
        process.wait()
        if process.returncode == 0:
            importlib.invalidate_caches()
            return True
        else:
            broadcast_callback("LOG", f"❌ Execution failed with code {process.returncode}")
            return False
    except Exception as e:
        broadcast_callback("LOG", f"❌ Sandbox Execution Error: {str(e)}")
    return False

def cleanup_research_sandbox():
    """Robustly removes the research sandbox and its virtual environment."""
    sandbox_dir = os.path.join(os.getcwd(), "research_sandbox")
    if os.path.exists(sandbox_dir):
        logger.info(f"🧺 LIQUIDATING_SANDBOX: Reclaiming resources at {sandbox_dir}...")
        try:
            # On Windows/Mac, some files might stay locked for a second
            shutil.rmtree(sandbox_dir, ignore_errors=True)
            # Second attempt if still exists
            if os.path.exists(sandbox_dir):
                time.sleep(1)
                shutil.rmtree(sandbox_dir, ignore_errors=True)
            return not os.path.exists(sandbox_dir)
        except Exception as e:
            logger.error(f"❌ Sandbox Liquidation Failed: {e}")
            return False
    return True

def get_environment_info():
    """Returns structured info about the sandbox environment (Root vs All packages)."""
    sandbox_dir = os.path.join(os.getcwd(), "research_sandbox")
    if not os.path.exists(sandbox_dir):
        return {"status": "NOT_INITIALIZED", "packages": []}
    
    python_bin = os.path.join(sandbox_dir, "bin", "python") if sys.platform != "win32" else os.path.join(sandbox_dir, "Scripts", "python.exe")
    
    global _ENV_CACHE
    now = time.time()
    if _ENV_CACHE["data"] and (now - _ENV_CACHE["time"]) < CACHE_TTL:
        return _ENV_CACHE["data"]

    try:
        # 🧪 Tier 1: Root Packages (Not required by others)
        root_cmd = [python_bin, "-m", "pip", "list", "--not-required", "--format=json"]
        roots_raw = subprocess.check_output(root_cmd).decode()
        roots = json.loads(roots_raw)
        
        # 🧪 Tier 2: Full Dependency Mesh
        all_cmd = [python_bin, "-m", "pip", "list", "--format=json"]
        all_raw = subprocess.check_output(all_cmd).decode()
        all_pkgs = json.loads(all_raw)
        
        # Identify Python Version
        ver_cmd = [python_bin, "--version"]
        python_ver = subprocess.check_output(ver_cmd).decode().strip().replace("Python ", "")
        
        result = {
            "status": "INITIALIZED_STABLE",
            "python": python_ver,
            "root_packages": roots,
            "all_packages": all_pkgs,
            "packages": roots # Legacy support for old UI
        }
        _ENV_CACHE = {"data": result, "time": now}
        return result
    except Exception as e:
        logger.error(f"Failed to scout environment: {e}")
        return {"status": "SCAN_FAILED", "error": str(e), "packages": []}

def inspect_dependencies(code):
    """Parses code to find all required top-level imports."""
    try:
        import ast
        tree = ast.parse(code)
    except SyntaxError:
        return []

    needed = set()
    # Common mapping for pypi vs import name
    mapping = {"sklearn": "scikit-learn", "cv2": "opencv-python", "PIL": "pillow"}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                pkg = n.name.split('.')[0]
                needed.add(mapping.get(pkg, pkg))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                pkg = node.module.split('.')[0]
                needed.add(mapping.get(pkg, pkg))
    
    return sorted(list(needed))

def inspect_parameters(code):
    """Parses code to find hyper-parameter assignments and their locations."""
    try:
        import ast
        tree = ast.parse(code)
    except Exception:
        return []

    # Extended list of research parameters to track
    target_params = {
        'epochs', 'learning_rate', 'lr', 'batch_size', 'batchsize',
        'dropout', 'momentum', 'weight_decay', 'privacy_epsilon', 'ε'
    }
    
    # Normalization mapping
    mapping = {'learning_rate': 'lr', 'batchsize': 'batch_size'}
    
    found = []
    
    for node in ast.walk(tree):
        # 🧪 Track Variable Assignments (e.g., epochs = 5)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.lower() in target_params:
                    # Capture value if it's a simple number
                    val = None
                    if isinstance(node.value, ast.Constant):
                        val = node.value.value
                    elif hasattr(node.value, 'n'): # Py3.7-
                        val = node.value.n
                    
                    found.append({
                        "name": mapping.get(target.id.lower(), target.id.lower()),
                        "value": val,
                        "lineno": node.lineno,
                    })
        
        # 🧪 Track Keyword Arguments (e.g., train(epochs=5))
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg and kw.arg.lower() in target_params:
                    val = None
                    if isinstance(kw.value, ast.Constant):
                        val = kw.value.value
                    found.append({
                        "name": mapping.get(kw.arg.lower(), kw.arg.lower()),
                        "value": val,
                        "lineno": node.lineno,
                    })

    unique = {}
    for p in found:
        unique[p['name']] = p
        
    return list(unique.values())

class TrainingSession:
    def __init__(self, code, hyperparams, bridge_broadcast_callback):
        self.code = code
        self.epochs = hyperparams.get("epochs", 5)
        self.lr = hyperparams.get("lr", 0.01)
        self.batch_size = hyperparams.get("batch_size", 32)
        self.broadcast = bridge_broadcast_callback
        self.stop_event = threading.Event()
        self.model = None
        self.status = "IDLE"
        self.progress = 0
        self.metrics = {"loss": [], "accuracy": []}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = None
        self.mode = "IDLE"
        self.namespace = {} # 🧪 Persistent Memory: Holds variables for terminal interaction

    def run(self):
        try:
            self.status = "TRAINING"
            # Ensure Sandbox is ready and get its interpreter
            self.research_python, self.site_packages = ensure_research_venv(self.broadcast)
            
            # 🧪 VENV INJECTION: Add sandbox site-packages to sys.path
            if self.site_packages and self.site_packages not in sys.path:
                sys.path.insert(0, self.site_packages)
                self.broadcast("LOG", f"✅ VENV_INJECTED: {self.site_packages}")
            
            self.broadcast("LOG", f"🔍 Sandbox Launcher: {self.research_python}")
            self.broadcast("LOG", f"SYSTEM: Starting execution on {self.device}...")
            
            # 0. Magic Commands (!) and Implicit CLI
            lines = self.code.split('\n')
            remaining_code_lines = []
            install_occurred = False
            for line in lines:
                l_stripped = line.strip()
                # Intercept '!' or lines starting with 'pip' / 'python' (implicit magic)
                if l_stripped.startswith('!') or l_stripped.lower().startswith('pip install ') or l_stripped.lower().startswith('python '):
                    magic_line = l_stripped if l_stripped.startswith('!') else f"!{l_stripped}"
                    success = self.execute_magic(magic_line, self.research_python)
                    if not success:
                        raise ValueError(f"Environment Command failed: {line}")
                    install_occurred = True
                else:
                    remaining_code_lines.append(line)
            
            if install_occurred:
                importlib.invalidate_caches()
            
            clean_code = '\n'.join(remaining_code_lines)

            # 0.1 Resolve Dependencies (JIT Pip for imports)
            self.broadcast("LOG", "📦 Analyzing research dependencies...")
            missing_libs = self.detect_missing_imports(clean_code)
            if missing_libs:
                for lib in missing_libs:
                    self.broadcast("LOG", f"📥 Installing missing dependency: {lib}...")
                    try:
                        # Use the sandbox python
                        subprocess.run([self.research_python, "-m", "pip", "install", lib], check=True, capture_output=True)
                        self.broadcast("LOG", f"✅ Successfully installed {lib}")
                        importlib.invalidate_caches()
                    except subprocess.CalledProcessError as e:
                        err_text = e.stderr.decode() if e.stderr else str(e)
                        self.broadcast("LOG", f"⚠️ Failed to install {lib}: {err_text}")
            
            # 1. Dynamic compilation & execution
            # Reset namespace for fresh run
            self.namespace = {
                "__name__": "__main__",
                "torch": torch,
                "nn": nn,
                "optim": optim,
                "DataLoader": DataLoader,
                "datasets": datasets,
                "transforms": transforms,
                "np": importlib.import_module('numpy') if importlib.util.find_spec('numpy') else None,
                "pd": importlib.import_module('pandas') if importlib.util.find_spec('pandas') else None,
            }
            # Inject headless Matplotlib config
            headless_config = "import matplotlib; matplotlib.use('Agg')\n"
            
            # Use real-time stream broadcast
            stream = BroadcastStream(self.broadcast)
            with contextlib.redirect_stdout(stream):
                try:
                    exec(headless_config + clean_code, self.namespace)
                    stream.flush()
                except Exception as e:
                    # Capture syntax/runtime errors during initial exec
                    logger.error(f"Exec failure: {e}")
                    raise e

            # Initial logs and dependency analysis are handled already.
            
            # Look for a class that is a subclass of nn.Module
            model_class = None
            for name, obj in self.namespace.items():
                if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
                    model_class = obj
                    break
            
            # MODE DETECTION: General Script vs Deep Learning Model
            if not model_class:
                self.mode = "SCRIPT"
                self.broadcast("LOG", "✅ Standalone script execution finalized.")
                self.status = "COMPLETE"
                self.broadcast("LAB_COMPLETE", {
                    "status": "COMPLETE",
                    "mode": "SCRIPT"
                })
                return

            self.mode = "MODEL"
            self.broadcast("LOG", "🤖 Neural architecture detected. Starting Deep Learning sequence...")
            self.model = model_class().to(self.device)
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            criterion = nn.Cross_entropy if hasattr(nn, 'Cross_entropy') else nn.functional.cross_entropy
            
            # 2. Data Loading
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            test_dataset = datasets.MNIST('./data', train=False, transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

            # 3. Training Loop
            for epoch in range(self.epochs):
                if self.stop_event.is_set():
                    self.status = "ABORTED"
                    self.broadcast("LOG", "SYSTEM: Training aborted by user.")
                    return

                self.model.train()
                running_loss = 0.0
                for batch_idx, (data, target) in enumerate(train_loader):
                    if self.stop_event.is_set(): break
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = nn.functional.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                # Evaluation
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
                
                accuracy = correct / total
                avg_loss = running_loss / len(train_loader)
                
                self.metrics["loss"].append(avg_loss)
                self.metrics["accuracy"].append(accuracy)
                self.progress = ((epoch + 1) / self.epochs) * 100
                
                # Broadcast progress
                self.broadcast("LAB_PROGRESS", {
                    "epoch": epoch + 1,
                    "total_epochs": self.epochs,
                    "loss": avg_loss,
                    "accuracy": accuracy,
                    "progress": self.progress,
                    "status": "TRAINING",
                    "mode": "MODEL"
                })
                
                # Also pipe to global LOG channel for total awareness
                self.broadcast("LOG", f"LAB_ENGINE: Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
                
                logger.info(f"Epoch {epoch+1}: Loss {avg_loss:.4f}, Acc {accuracy:.4f}")

            # 4. Save results
            save_dir = os.path.join(os.getcwd(), "exports")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            timestamp = int(time.time())
            self.model_path = os.path.join(save_dir, f"model_{timestamp}.pt")
            torch.save(self.model.state_dict(), self.model_path)
            
            # Export to ONNX if possible
            onnx_path = os.path.join(save_dir, f"model_{timestamp}.onnx")
            try:
                dummy_input = torch.randn(1, 1, 28, 28).to(self.device)
                torch.onnx.export(self.model, dummy_input, onnx_path)
            except Exception as e:
                logger.warning(f"ONNX export failed: {e}")
                onnx_path = None

            self.status = "COMPLETE"
            self.broadcast("LAB_COMPLETE", {
                "status": "COMPLETE",
                "pt_path": self.model_path,
                "onnx_path": onnx_path,
                "metrics": self.metrics,
                "mode": "MODEL"
            })
            self.broadcast("LOG", "SYSTEM: Training complete. Model weights exported.")

        except (BrokenPipeError, ConnectionResetError):
            logger.warning("Networking glitch detected (Broken Pipe). Training logic unaffected.")
            # We don't change status to ERROR because the training is actually fine.
            self.status = "COMPLETE" 
            
        except Exception as e:
            # Check if likely a broken pipe error wrapped in a RuntimeError
            if "Broken pipe" in str(e) or "Connection reset" in str(e):
                logger.warning(f"Likely networking glitch: {e}")
                self.status = "COMPLETE"
                return

            self.status = "ERROR"
            error_msg = f"Training Error: {str(e)}"
            logger.error(error_msg)
            try:
                self.broadcast("LAB_ERROR", {"error": error_msg})
                self.broadcast("LOG", f"FATAL: {error_msg}")
            except:
                pass

    def eval_cell(self, code_block):
        """Intelligently evaluate a code block in the current session's namespace."""
        if not self.namespace:
            return False, "Session memory not initialized. Run code first."
        
        self.broadcast("LOG", f">>> {code_block.strip()[:100]}{'...' if len(code_block) > 100 else ''}")
        
        stream = BroadcastStream(self.broadcast)
        with contextlib.redirect_stdout(stream):
            try:
                # 🧪 Smart Parsing: Handle last-line expressions (Jupyter style)
                tree = ast.parse(code_block)
                if not tree.body:
                    return True, "Empty cell"
                
                last_node = tree.body[-1]
                if isinstance(last_node, ast.Expr):
                    # Separate the body minus the last expression
                    exec_body = ast.Module(body=tree.body[:-1], type_ignores=[])
                    exec(compile(exec_body, '<interactive>', 'exec'), self.namespace)
                    
                    # Evaluate the last expression
                    eval_expr = compile(ast.Expression(body=last_node.value), '<interactive>', 'eval')
                    result = eval(eval_expr, self.namespace)
                    
                    if result is not None:
                        self.broadcast("LOG", f"Out: {repr(result)}")
                else:
                    # Execute full block
                    exec(compile(tree, '<interactive>', 'exec'), self.namespace)
                
                stream.flush()
                return True, "Cell executed"
            except Exception as e:
                err_msg = f"In-situ Error: {str(e)}"
                self.broadcast("LOG", f"❌ {err_msg}")
                return False, err_msg

    def detect_missing_imports(self, code):
        """Check if any detected imports are missing from the sandbox."""
        needed = inspect_dependencies(code)
        missing = []
        for lib in needed:
            # Check if module exists in site-packages or current path
            # Simple check for now: can we find the spec?
            if importlib.util.find_spec(lib.replace("-", "_")) is None:
                # Double check mapping
                missing.append(lib)
        return missing

    def execute_magic(self, line, python_exec=None):
        """Execute a shell command starting with '!' and stream output."""
        return run_sandbox_command(line, self.broadcast)

    def abort(self):
        self.stop_event.set()

# Singleton-like manager for training sessions
_current_session = None

def start_training(code, hyperparams, broadcast_callback):
    global _current_session
    if _current_session and _current_session.status == "TRAINING":
        return False, "A training session is already in progress."
    
    _current_session = TrainingSession(code, hyperparams, broadcast_callback)
    thread = threading.Thread(target=_current_session.run)
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
            "metrics": _current_session.metrics,
            "mode": _current_session.mode
        }
    return {"status": "IDLE"}
