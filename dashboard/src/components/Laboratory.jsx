import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Code, Play, ShieldCheck, Terminal, Zap,
  AlertCircle, RefreshCw, Download, BarChart2,
  Activity, Settings, CheckCircle2, StopCircle, Cpu, Box,
  ChevronRight, Package, Loader2, Database, Lock, Copy, Shield
} from 'lucide-react';
import { API_BASE_URL } from '../hooks/useSecureFederated';

const DEFAULT_MODEL_CODE = `# ╔══════════════════════════════════════════════════╗
# ║  Federated Learning — Secure Training Sandbox   ║
# ║  v2.0 | Privacy Vault Enabled                   ║
# ╚══════════════════════════════════════════════════╝
#
# AVAILABLE COMMANDS:
#   vault.list()                  → View all encrypted datasets
#   vault.load("Iris")            → Decrypt dataset to RAM (numpy)
#   vault.load_torch("Digits")    → Decrypt dataset to RAM (PyTorch)
#   !pip install [pkg]            → Install a package
#   !ls                           → List files
#
# HOW IT WORKS:
#   1. Click a dataset from "Vault Datasets" panel →
#   2. Your model trains on encrypted vault data
#   3. After training, decrypted data is wiped from RAM
# ─────────────────────────────────────────────────

import torch
import torch.nn as nn
import numpy as np

# ── Step 1: Load encrypted dataset from Privacy Vault ──
data, labels, info = vault.load("Iris")
print(f"🔐 Decrypted: {info['name']} → {data.shape[0]} samples, {info['num_classes']} classes")
print(f"   Features: {info.get('feature_names', 'N/A')}")

# ── Step 2: Define your model ──
class IrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)    # 3 classes: Setosa, Versicolor, Virginica
        )

    def forward(self, x):
        return self.net(x)

# ── Hyperparameters (adjust in sidebar) ──
def train(epochs=10, lr=0.001, batch_size=32):
    pass

print("🧪 Model ready. Press ▶ RUN to train on vault data.")
print("   Data stays encrypted at rest — decrypted in RAM only during training.")
`;


export default function Laboratory({ onAction, labState, onExecuteCommand, onEvalCode }) {
  const [code, setCode] = useState(DEFAULT_MODEL_CODE);
  const [status, setStatus] = useState('IDLE');
  const [logs, setLogs] = useState([]);
  const [errorLine, setErrorLine] = useState(null);
  const textAreaRef = useRef(null);
  const gutterRef = useRef(null);

  const [epochs, setEpochs] = useState(5);
  const [lr, setLr] = useState(0.001);
  const [batchSize, setBatchSize] = useState(32);
  const [envData, setEnvData] = useState(null);
  const [isEnvLoading, setIsEnvLoading] = useState(false);
  const [detectedDeps, setDetectedDeps] = useState([]);
  const [neededParams, setNeededParams] = useState([
    { name: 'epochs', value: 5, lineno: null },
    { name: 'lr', value: 0.001, lineno: null },
    { name: 'batch_size', value: 32, lineno: null }
  ]);
  const [isInspecting, setIsInspecting] = useState(false);
  const [isDeepScan, setIsDeepScan] = useState(false);
  const [isLiveSyncEnabled, setIsLiveSyncEnabled] = useState(true);
  const [syncingLines, setSyncingLines] = useState([]); // Track lines being updated

  // Terminal State
  const [terminalInput, setTerminalInput] = useState('');
  const [terminalHistory, setTerminalHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const terminalRef = useRef(null);

  // Console resize state
  const [consoleHeight, setConsoleHeight] = useState(280);
  const resizeRef = useRef({ dragging: false, startY: 0, startH: 0 });

  // Vault Datasets
  const [vaultDatasets, setVaultDatasets] = useState([]);
  const [isVaultLoading, setIsVaultLoading] = useState(false);
  const [copiedSnippet, setCopiedSnippet] = useState(null);

  useEffect(() => {
    const onMouseMove = (e) => {
      if (!resizeRef.current.dragging) return;
      const delta = resizeRef.current.startY - e.clientY;
      const newH = Math.min(600, Math.max(120, resizeRef.current.startH + delta));
      setConsoleHeight(newH);
    };
    const onMouseUp = () => {
      if (resizeRef.current.dragging) {
        resizeRef.current.dragging = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      }
    };
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, []);

  const startConsoleResize = (e) => {
    resizeRef.current = { dragging: true, startY: e.clientY, startH: consoleHeight };
    document.body.style.cursor = 'ns-resize';
    document.body.style.userSelect = 'none';
  };

  // Package Installation State
  const [isInstalling, setIsInstalling] = useState(false);
  const [installingPackages, setInstallingPackages] = useState([]);
  const [installProgress, setInstallProgress] = useState(0);
  const [installLogs, setInstallLogs] = useState([]);

  const runPipInstall = async (packages) => {
    const pkgList = Array.isArray(packages) ? packages : [packages];
    setIsInstalling(true);
    setInstallingPackages(pkgList);
    setInstallProgress(0);
    setInstallLogs([{ msg: `$ pip install ${pkgList.join(' ')}`, type: 'cmd' }]);

    for (let i = 0; i < pkgList.length; i++) {
      const pkg = pkgList[i];
      setInstallLogs(prev => [...prev, { msg: `Collecting ${pkg}...`, type: 'info' }]);
      setInstallProgress(Math.round(((i) / pkgList.length) * 60));

      try {
        await onExecuteCommand(`!pip install ${pkg}`);
        setInstallLogs(prev => [...prev,
        { msg: `  Downloading ${pkg}...`, type: 'info' },
        { msg: `  Installing collected packages: ${pkg}`, type: 'info' },
        { msg: `Successfully installed ${pkg}`, type: 'success' }
        ]);
      } catch {
        setInstallLogs(prev => [...prev, { msg: `ERROR: Failed to install ${pkg}`, type: 'error' }]);
      }
      setInstallProgress(Math.round(((i + 1) / pkgList.length) * 100));
    }

    setInstallLogs(prev => [...prev, { msg: `\n✓ Installation complete. ${pkgList.length} package(s) processed.`, type: 'success' }]);
    addLog(`Installed ${pkgList.length} package(s): ${pkgList.join(', ')}`, 'success');

    // Refresh environment after install
    setTimeout(() => {
      fetchEnvironment();
      setIsInstalling(false);
      setInstallingPackages([]);
      setInstallProgress(0);
    }, 1500);
  };

  const addLog = (msg, type = 'info') => {
    setLogs(prev => [{ msg, type, time: new Date().toLocaleTimeString() }, ...prev].slice(0, 50));
  };

  // Sync scroll between gutter and textarea
  useEffect(() => {
    const textarea = textAreaRef.current;
    const gutter = gutterRef.current;
    if (!textarea || !gutter) return;
    const handleScroll = () => { gutter.scrollTop = textarea.scrollTop; };
    textarea.addEventListener('scroll', handleScroll);
    return () => textarea.removeEventListener('scroll', handleScroll);
  }, []);
  useEffect(() => {
    if (!labState) return;
    if (labState.status === 'TRAINING') {
      setStatus('TRAINING');
    } else if (labState.status === 'COMPLETE') {
      setStatus('COMPLETE');
    } else if (labState.status === 'ERROR') {
      setStatus('ERROR');
      addLog(`Backend Error: ${labState.error} `, 'error');
    }
  }, [labState]);

  const fetchEnvironment = async () => {
    setIsEnvLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/laboratory/environment`);
      const data = await response.json();
      setEnvData(data);
    } catch (err) {
      console.error("Failed to fetch environment:", err);
    } finally {
      setIsEnvLoading(false);
    }
  };

  const inspectDependencies = async () => {
    if (!code.trim()) {
      setDetectedDeps([]);
      return;
    }
    setIsInspecting(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/laboratory/inspect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code })
      });
      const data = await response.json();
      if (data.success) {
        setDetectedDeps(data.dependencies);
        if (data.parameters && data.parameters.length > 0) {
          setNeededParams(data.parameters);

          // Sync state values to detected code values if they differ
          data.parameters.forEach(p => {
            if (p.value !== null) {
              if (p.name === 'epochs') setEpochs(p.value);
              if (p.name === 'lr') setLr(p.value);
              if (p.name === 'batch_size') setBatchSize(p.value);
            }
          });
        } else if (code.trim()) {
          setNeededParams([]);
        }
      }
    } catch (err) {
      console.error("Dependency inspection failed:", err);
    } finally {
      setIsInspecting(false);
    }
  };

  const fetchVaultDatasets = async () => {
    setIsVaultLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/laboratory/vault-datasets`);
      const data = await res.json();
      if (data.success) setVaultDatasets(data.datasets || []);
    } catch (err) {
      console.error('Failed to fetch vault datasets:', err);
    } finally {
      setIsVaultLoading(false);
    }
  };

  const insertVaultSnippet = (dsName) => {
    const snippet = `\n# Load encrypted dataset from Privacy Vault\ndata, labels, info = vault.load("${dsName}")\nprint(f"Loaded {info['name']}: {data.shape[0]} samples, {info['num_classes']} classes")\n`;
    setCode(prev => prev + snippet);
    setCopiedSnippet(dsName);
    setTimeout(() => setCopiedSnippet(null), 2000);
  };

  useEffect(() => {
    fetchEnvironment();
    fetchVaultDatasets();
    const interval = setInterval(fetchEnvironment, 45000);
    return () => clearInterval(interval);
  }, []);

  // 🧪 Auto-Cleanup on browser close
  useEffect(() => {
    const cleanup = () => {
      navigator.sendBeacon(`${API_BASE_URL}/api/v1/laboratory/purge`);
    };
    window.addEventListener('beforeunload', cleanup);
    return () => window.removeEventListener('beforeunload', cleanup);
  }, []);

  // Debounced inspection
  useEffect(() => {
    const timer = setTimeout(() => {
      inspectDependencies();
    }, 600); // 🧪 Accelerated from 2000ms for High-Frequency Sync
    return () => clearTimeout(timer);
  }, [code]);

  const syncParamToSource = (paramName, newValue) => {
    if (!isLiveSyncEnabled) return;

    const param = neededParams.find(p => p.name === paramName);
    if (!param || !param.lineno) return;

    const lines = code.split('\n');
    const lineIndex = param.lineno - 1;
    const line = lines[lineIndex];

    if (line) {
      // Robust regex for variable assignment/dictionary mapping
      // Captures group 1 (everything up to the value) and follows with the numeric value
      const regex = new RegExp(`(${paramName}[^\\n]*?[:=]\\s*)([0-9\\.e\\-]+)`, 'i');
      if (regex.test(line)) {
        lines[lineIndex] = line.replace(regex, `$1${newValue}`);
        setCode(lines.join('\n'));

        // Trigger visual sync indicator
        setSyncingLines(prev => [...new Set([...prev, param.lineno])]);
        setTimeout(() => {
          setSyncingLines(prev => prev.filter(l => l !== param.lineno));
        }, 1200);
      }
    }
  };

  const pushParamToSource = (paramName, newValue) => {
    const param = neededParams.find(p => p.name === paramName);
    if (!param || !param.lineno) return;

    const lines = code.split('\n');
    const lineIndex = param.lineno - 1;
    const line = lines[lineIndex];

    if (line) {
      const regex = new RegExp(`(${paramName}[^\\n]*?[:=]\\s*)([0-9\\.e\\-]+)`, 'i');
      if (regex.test(line)) {
        lines[lineIndex] = line.replace(regex, `$1${newValue}`);
        setCode(lines.join('\n'));
        addLog(`Pushed ${paramName}=${newValue} to source line ${param.lineno}.`, 'success');

        setSyncingLines(prev => [...new Set([...prev, param.lineno])]);
        setTimeout(() => {
          setSyncingLines(prev => prev.filter(l => l !== param.lineno));
        }, 1200);
      }
    }
  };

  const handleParamChange = (name, value, setter) => {
    setter(value);
    if (isLiveSyncEnabled) {
      syncParamToSource(name, value);
    }
  };

  const handlePurgeSandbox = async () => {
    if (!window.confirm("🧺 LIQUIDATE RESOURCE? This will delete all installed libraries in the sandbox to reclaim disk space. You will need to wait for a warm-up on next execution.")) return;
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/laboratory/purge`, { method: 'POST' });
      const data = await response.json();
      if (data.success) {
        addLog("Sandbox resources liquidated successfully.", "warning");
        fetchEnvironment(); // Refresh env status
      } else {
        addLog(data.error || "Purge failed.", "error");
      }
    } catch (err) {
      addLog("Failed to connect to liquidation engine.", "error");
    }
  };

  const handleCompile = async () => {
    setStatus('COMPILING');
    addLog('Initiating backend compilation protocol...', 'info');
    setErrorLine(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/laboratory/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code })
      });
      const data = await response.json();
      if (data.success) {
        setStatus('READY');
        addLog('Compilation successful. Architecture verified.', 'success');
      } else {
        setStatus('ERROR');
        setErrorLine(data.line || null);
        addLog(`[${data.type || 'Error'}] Line ${data.line || '?'}: ${data.error} `, 'error');
      }
    } catch (err) {
      setStatus('ERROR');
      addLog('FATAL: Could not connect to Institutional Compiler Service.', 'error');
    }
  };

  const handleTrain = async () => {
    if (status === 'TRAINING') return;

    // Auto-validate before training
    setStatus('COMPILING');
    addLog('Auto-validating code before training...', 'info');
    setErrorLine(null);
    
    try {
      const valResponse = await fetch(`${API_BASE_URL}/api/v1/laboratory/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code })
      });
      const valData = await valResponse.json();
      
      if (!valData.success) {
        setStatus('ERROR');
        setErrorLine(valData.line || null);
        addLog(`[${valData.type || 'Error'}] Line ${valData.line || '?'}: ${valData.error}`, 'error');
        return;
      }
    } catch (err) {
      setStatus('ERROR');
      addLog('Could not connect to validation service.', 'error');
      return;
    }

    // Validation passed — start training
    setStatus('TRAINING');
    addLog(`Starting execution (Epochs: ${epochs}, LR: ${lr}, Batch: ${batchSize})...`, 'info');
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/laboratory/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, hyperparams: { epochs, lr, batch_size: batchSize } })
      });
      const data = await response.json();
      if (!data.success) {
        setStatus('ERROR');
        addLog(`Failed to start training: ${data.message}`, 'error');
      }
    } catch (err) {
      setStatus('ERROR');
      addLog('Training service unreachable.', 'error');
    }
  };

  const handleAbort = async () => {
    try {
      await fetch(`${API_BASE_URL}/api/v1/laboratory/abort`, { method: 'POST' });
      setStatus('IDLE');
      addLog('Training abort signal sent.', 'warning');
    } catch (err) {
      addLog('Failed to abort training session.', 'error');
    }
  };

  const handleDownload = (format) => {
    window.open(`${API_BASE_URL}/api/v1/laboratory/download/${format}`, '_blank');
  };

  const handleTerminalSubmit = async (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const cmd = terminalInput.trim();
      if (!cmd) return;

      setIsEvaluating(true);
      setTerminalHistory(prev => [cmd, ...prev]);
      setHistoryIndex(-1);

      addLog(cmd, 'user'); // Distinguished User Echo

      try {
        if (cmd.startsWith('!')) {
          await onExecuteCommand(cmd);
        } else {
          const result = await onEvalCode(cmd);
          if (result && !result.success && result.error) {
            addLog(`❌ Error: ${result.error}`, 'error');
          }
        }
        setTerminalInput('');
        // Scroll console to bottom
        setTimeout(() => {
          if (terminalRef.current) terminalRef.current.scrollIntoView({ behavior: 'smooth' });
        }, 100);
      } catch (err) {
        onAction('Evaluation failed. Session may be dead.', 'error');
      } finally {
        setIsEvaluating(false);
      }
    } else if (e.key === 'ArrowUp') {
      if (historyIndex < terminalHistory.length - 1) {
        const nextIdx = historyIndex + 1;
        setHistoryIndex(nextIdx);
        setTerminalInput(terminalHistory[nextIdx]);
      }
    } else if (e.key === 'ArrowDown') {
      if (historyIndex > 0) {
        const nextIdx = historyIndex - 1;
        setHistoryIndex(nextIdx);
        setTerminalInput(terminalHistory[nextIdx]);
      } else {
        setHistoryIndex(-1);
        setTerminalInput('');
      }
    }
  };

  const lineCount = code.split('\n').length;
  const lineNumbers = Array.from({ length: Math.max(lineCount, 30) }, (_, i) => i + 1);
  const progress = labState?.progress || 0;

  const statusColor = status === 'READY' || status === 'COMPLETE' ? 'var(--success)'
    : status === 'ERROR' ? 'var(--error)'
      : status === 'TRAINING' ? 'var(--accent)'
        : 'var(--text-muted)';

  const statusLabel = {
    IDLE: 'Awaiting Input',
    COMPILING: 'Validating...',
    READY: 'Architecture Verified',
    TRAINING: labState?.mode === 'SCRIPT' ? 'Executing standalone script...' : `Training — Epoch ${labState?.epoch || 0}/${epochs}`,
    COMPLETE: labState?.mode === 'SCRIPT' ? 'Execution Complete' : 'Federated Training Complete',
    ERROR: 'Exception Raised'
  }[status] || status;

  return (
    <div className="lab-root">
      {/* ─── Compact Header Bar ─── */}
      <div className="lab-header">
        <div className="lab-header-left">
          <div className="lab-icon-box">
            <Code size={16} strokeWidth={2.5} />
          </div>
          <div className="lab-header-titles">
            <h2 className="lab-title">Research Laboratory</h2>
            <span className="lab-subtitle">Interpreter &amp; Compiler v1.5</span>
          </div>
          <div className="lab-status-pill" style={{ '--pill-color': statusColor }}>
            {status === 'COMPILING' || status === 'TRAINING' ? (
              <RefreshCw size={10} className="lab-spin" />
            ) : status === 'READY' || status === 'COMPLETE' ? (
              <CheckCircle2 size={10} />
            ) : status === 'ERROR' ? (
              <AlertCircle size={10} />
            ) : (
              <Cpu size={10} />
            )}
            <span>{statusLabel}</span>
          </div>
        </div>

        <div className="lab-header-actions">
          <button
            onClick={handleCompile}
            disabled={status === 'COMPILING' || status === 'TRAINING'}
            className={`lab-btn lab-btn-outline ${(status === 'READY' || status === 'COMPLETE') ? 'lab-btn-verified' : ''}`}
          >
            <ShieldCheck size={14} />
            <span>{status === 'READY' || status === 'COMPLETE' ? 'Verified' : 'Verify Code'}</span>
          </button>

          <button
            onClick={status === 'TRAINING' ? handleAbort : handleTrain}
            disabled={status === 'COMPILING'}
            className={`lab-btn ${status === 'TRAINING' ? 'lab-btn-danger' : 'lab-btn-primary'}`}
          >
            {status === 'TRAINING' ? <StopCircle size={14} /> : <Zap size={14} />}
            <span>{status === 'TRAINING' ? 'Abort' : 'Train Model'}</span>
          </button>
        </div>
      </div>

      {/* ─── Progress Strip (only during training/complete) ─── */}
      <AnimatePresence>
        {(status === 'TRAINING' || status === 'COMPLETE') && (
          <motion.div
            className="lab-progress-strip"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="lab-progress-bar-track">
              <motion.div
                className="lab-progress-bar-fill"
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.6, ease: 'easeOut' }}
              />
            </div>
            <div className="lab-progress-stats">
              {labState?.mode !== 'SCRIPT' && (
                <>
                  <div className="lab-stat">
                    <span className="lab-stat-label">Epoch</span>
                    <span className="lab-stat-value">{labState?.epoch || 0}/{epochs}</span>
                  </div>
                  <div className="lab-stat">
                    <span className="lab-stat-label">Loss</span>
                    <span className="lab-stat-value lab-stat-loss">{(labState?.loss || 0).toFixed(4)}</span>
                  </div>
                  <div className="lab-stat">
                    <span className="lab-stat-label">Accuracy</span>
                    <span className="lab-stat-value lab-stat-acc">{((labState?.accuracy || 0) * 100).toFixed(2)}%</span>
                  </div>
                </>
              )}
              <div className="lab-stat">
                <span className="lab-stat-label">Progress</span>
                <span className="lab-stat-value">{progress.toFixed(0)}%</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ─── Main Workspace ─── */}
      <div className="lab-workspace">
        {/* === Code Editor === */}
        <div className="lab-editor-panel">
          <div className="lab-editor-wrap">
            {/* Gutter */}
            <div className="lab-gutter" ref={gutterRef}>
              {lineNumbers.map(num => {
                const isSyncing = syncingLines.includes(num);
                return (
                  <div
                    key={num}
                    className={`lab-gutter-line ${errorLine === num ? 'lab-gutter-error' : ''} ${isSyncing ? 'lab-gutter-syncing' : ''}`}
                  >
                    {isSyncing ? (
                      <motion.div
                        initial={{ scale: 0.5, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        className="lab-sync-indicator"
                      >
                        <Zap size={8} />
                      </motion.div>
                    ) : num}
                  </div>
                );
              })}
            </div>
            {/* Textarea */}
            <textarea
              ref={textAreaRef}
              value={code}
              onChange={(e) => {
                setCode(e.target.value);
                if (status === 'READY' || status === 'COMPLETE') setStatus('IDLE');
              }}
              spellCheck={false}
              className="lab-textarea"
            />

            {/* 🪄 Smart Shell Command Detector — validates real packages */}
            <AnimatePresence>
            {(() => {
              const KNOWN_PACKAGES = [
                'numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'plotly', 'bokeh',
                'scikit-learn', 'sklearn', 'xgboost', 'lightgbm', 'catboost',
                'torch', 'torchvision', 'torchaudio', 'pytorch-lightning',
                'tensorflow', 'keras', 'jax', 'jaxlib', 'flax',
                'transformers', 'datasets', 'tokenizers', 'accelerate', 'diffusers',
                'opencv-python', 'cv2', 'pillow', 'imageio', 'albumentations',
                'flask', 'fastapi', 'uvicorn', 'django', 'requests', 'httpx', 'aiohttp',
                'sqlalchemy', 'psycopg2', 'pymongo', 'redis', 'celery',
                'pytest', 'unittest', 'coverage', 'black', 'flake8', 'mypy', 'ruff',
                'tqdm', 'rich', 'click', 'typer', 'pydantic', 'attrs',
                'networkx', 'sympy', 'statsmodels', 'lifelines',
                'nltk', 'spacy', 'gensim', 'textblob', 'langchain',
                'openai', 'anthropic', 'cohere', 'replicate',
                'wandb', 'mlflow', 'optuna', 'ray', 'dask',
                'jupyter', 'ipython', 'notebook', 'jupyterlab',
                'streamlit', 'gradio', 'panel', 'dash',
                'cryptography', 'pynacl', 'bcrypt', 'pyotp',
                'boto3', 'google-cloud-storage', 'azure-storage-blob',
                'docker', 'kubernetes', 'ansible',
                'flower', 'flwr', 'syft', 'pysyft',
                'onnx', 'onnxruntime', 'tensorrt',
                'huggingface-hub', 'safetensors', 'sentencepiece',
                'einops', 'timm', 'fairscale', 'deepspeed',
                'gym', 'gymnasium', 'stable-baselines3',
                'polars', 'pyarrow', 'duckdb', 'vaex',
                'beautifulsoup4', 'scrapy', 'selenium', 'playwright',
                'pyyaml', 'toml', 'python-dotenv', 'configparser',
                'Pillow', 'h5py', 'zarr', 'lmdb',
              ];

              const lines = code.split('\n');

              const nonInstallPatterns = [
                /^pip\s+(list|freeze|check)\s*$/,
                /^pip\s+show\s+\S{2,}/,
                /^python\s+\S+\.py/,
                /^ls(\s+-[a-zA-Z]+)?(\s+\S+)?$/,
              ];

              const pipInstallMatch = lines.find(l => {
                const t = l.trim();
                if (!t || t.startsWith('!') || t.startsWith('#')) return false;
                return /^(?:pip|conda)\s+install\s+/.test(t);
              });

              const shellMatch = !pipInstallMatch && lines.find(l => {
                const t = l.trim();
                if (!t || t.startsWith('!') || t.startsWith('#')) return false;
                return nonInstallPatterns.some(pat => pat.test(t));
              });

              // Non-install shell commands → just Run
              if (shellMatch) {
                const correctedCmd = `!${shellMatch.trim()}`;
                return (
                  <motion.div key="magic-hint" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 20 }} className="lab-magic-hint">
                    <div className="lab-magic-hint-icon"><Zap size={14} /></div>
                    <div className="lab-magic-hint-body">
                      <span className="lab-magic-hint-title">Shell command detected</span>
                      <span className="lab-magic-hint-sub">Execute <code>{correctedCmd}</code> in sandbox</span>
                    </div>
                    <div className="lab-magic-hint-actions">
                      <button className="lab-magic-btn lab-magic-btn-run" onClick={() => { onExecuteCommand(correctedCmd); setCode(''); }}><Play size={10} /> Run</button>
                    </div>
                  </motion.div>
                );
              }

              // pip/conda install → validate packages
              if (pipInstallMatch) {
                const trimmed = pipInstallMatch.trim();
                const pkgPart = trimmed.replace(/^(?:pip|conda)\s+install\s+/, '').trim();
                const pkgNames = pkgPart.split(/\s+/).filter(p => p && !p.startsWith('-'));

                if (pkgNames.length === 0) return null;

                const lastPkg = pkgNames[pkgNames.length - 1];
                if (lastPkg.length < 2) return null;

                const allValid = pkgNames.every(p =>
                  KNOWN_PACKAGES.some(kp => kp.toLowerCase() === p.toLowerCase())
                );

                const suggestions = !allValid
                  ? KNOWN_PACKAGES.filter(kp =>
                      kp.toLowerCase().startsWith(lastPkg.toLowerCase()) &&
                      kp.toLowerCase() !== lastPkg.toLowerCase()
                    ).slice(0, 5)
                  : [];

                // All valid → show Install button
                if (allValid) {
                  return (
                    <motion.div key="magic-hint" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 20 }} className="lab-magic-hint">
                      <div className="lab-magic-hint-icon"><Package size={14} /></div>
                      <div className="lab-magic-hint-body">
                        <span className="lab-magic-hint-title">Install {pkgNames.join(', ')}</span>
                        <span className="lab-magic-hint-sub">Package{pkgNames.length > 1 ? 's' : ''} verified — ready to install</span>
                      </div>
                      <div className="lab-magic-hint-actions">
                        <button className="lab-magic-btn lab-magic-btn-run" onClick={() => { runPipInstall(pkgNames); setCode(''); }}>
                          <Download size={10} /> Install Now
                        </button>
                      </div>
                    </motion.div>
                  );
                }

                // Partial match → show clickable suggestion pills that install directly
                if (suggestions.length > 0) {
                  return (
                    <motion.div key="magic-hint" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 20 }} className="lab-magic-hint">
                      <div className="lab-magic-hint-icon"><Package size={14} /></div>
                      <div className="lab-magic-hint-body">
                        <span className="lab-magic-hint-title">Did you mean?</span>
                        <span className="lab-magic-hint-sub lab-magic-suggestions">
                          {suggestions.map((s, i) => (
                            <button key={i} className="lab-magic-suggestion" onClick={() => {
                              runPipInstall([s]);
                              setCode('');
                            }}>
                              <Download size={8} /> {s}
                            </button>
                          ))}
                        </span>
                      </div>
                    </motion.div>
                  );
                }

                // Unknown package (≥3 chars) → warn but allow install attempt
                if (lastPkg.length >= 3) {
                  return (
                    <motion.div key="magic-hint" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 20 }} className="lab-magic-hint lab-magic-hint-warn">
                      <div className="lab-magic-hint-icon"><AlertCircle size={14} /></div>
                      <div className="lab-magic-hint-body">
                        <span className="lab-magic-hint-title">Unknown: {lastPkg}</span>
                        <span className="lab-magic-hint-sub">Not in verified list — may still exist on PyPI</span>
                      </div>
                      <div className="lab-magic-hint-actions">
                        <button className="lab-magic-btn lab-magic-btn-run" onClick={() => { runPipInstall(pkgNames); setCode(''); }}><Play size={10} /> Try Install</button>
                      </div>
                    </motion.div>
                  );
                }

                return null;
              }

              return null;
            })()}
            </AnimatePresence>
          </div>
        </div>

        {/* === Right Panel === */}
        <div className="lab-sidebar">
          {/* ⚙️ Academic Control Console (Adaptive) */}
          <div className="lab-panel">
            <div className="lab-panel-header flex justify-between items-center px-4 py-2 border-b border-border">
              <div className="flex items-center gap-2">
                <Settings size={14} className="text-primary" />
                <span className="type-label text-[10px]">Model_Regulation</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[8px] font-bold opacity-40 uppercase tracking-tighter">Live_Sync</span>
                <button
                  onClick={() => setIsLiveSyncEnabled(!isLiveSyncEnabled)}
                  className={`w-8 h-4 rounded-full relative transition-colors ${isLiveSyncEnabled ? 'bg-primary/20' : 'bg-white/5 border border-white/10'}`}
                >
                  <motion.div
                    animate={{ x: isLiveSyncEnabled ? 16 : 2 }}
                    className={`w-2.5 h-2.5 rounded-full absolute top-[3px] ${isLiveSyncEnabled ? 'bg-primary' : 'bg-gray-600'}`}
                  />
                </button>
              </div>
            </div>

            <div className="p-6 space-y-8 bg-surface overflow-y-auto max-h-[600px]">

              {/* Dynamic Parameter Instruments */}
              {neededParams.map((p, idx) => {
                const isCore = ['epochs', 'lr', 'batch_size'].includes(p.name);
                const isSlider = ['epochs', 'dropout', 'momentum', 'weight_decay', 'privacy_epsilon'].includes(p.name);

                // Determine value and setter mapping
                let val, setter, min = 0, max = 100, step = 1;
                if (p.name === 'epochs') { val = epochs; setter = setEpochs; min = 1; max = 50; step = 1; }
                else if (p.name === 'lr') { val = lr; setter = setLr; min = 0.0001; max = 1; step = 0.0001; }
                else if (p.name === 'batch_size') { val = batchSize; setter = setBatchSize; min = 16; max = 128; step = 16; }
                else {
                  // Fallback for dynamic extra params
                  val = p.value || 0;
                  setter = (v) => syncParamToSource(p.name, v);
                  min = 0; max = 1; step = 0.01;
                }

                return (
                  <div key={idx} className="transition-all opacity-100 group">
                    <div className="flex justify-between items-center mb-4">
                      <div className="flex items-center gap-2">
                        <div className={`link-indicator ${isLiveSyncEnabled ? 'active' : ''}`}></div>
                        <span className="text-[10px] font-bold uppercase tracking-wide text-text-main">
                          {p.name?.replace('_', ' ') || 'PARAM'}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        {!isLiveSyncEnabled && (
                          <button
                            onClick={() => pushParamToSource(p.name, val)}
                            className="px-2 py-0.5 bg-primary/10 hover:bg-primary/20 text-primary border border-primary/20 text-[7px] font-bold uppercase transition-all"
                          >
                            Push
                          </button>
                        )}
                        <span className="instrument-readout">
                          {typeof val === 'number' ? (p.name === 'epochs' ? val.toString().padStart(2, '0') : val.toFixed(4)) : (val || '0.00')}
                        </span>
                      </div>
                    </div>

                    {isSlider ? (
                      <input
                        type="range" min={min} max={max} step={step} value={val || 0}
                        onChange={(e) => handleParamChange(p.name, parseFloat(e.target.value), setter)}
                        className="instrument-slider"
                      />
                    ) : (
                      <div className="flex gap-2">
                        <input
                          type={typeof val === 'number' ? "number" : "text"}
                          step={step} value={val || ''}
                          onChange={(e) => handleParamChange(p.name, e.target.type === 'number' ? parseFloat(e.target.value) : e.target.value, setter)}
                          className="w-full bg-transparent border-b border-border text-[12px] font-mono py-1 focus:border-primary outline-none transition-all"
                        />
                      </div>
                    )}
                    <div className="mt-1 text-[7px] font-mono text-gray-500 opacity-0 group-hover:opacity-100 transition-opacity">
                      SYNCED_TO_SOURCE:LINE_{p.lineno || 'AUTO'}
                    </div>
                  </div>
                );
              })}

              {neededParams.length === 0 && (
                <div className="py-10 text-center border border-dashed border-border rounded">
                  <span className="text-[10px] text-text-muted italic">NO_LIVE_PARAMETERS_DETECTED</span>
                </div>
              )}
            </div>
          </div>


          {/* 🧪 Environment Monitor — Redesigned */}
          <div className="lab-panel lab-env-panel">
            <div className="lab-env-header">
              <div className="lab-env-header-left">
                <div className={`lab-env-status-dot ${envData ? 'lab-env-dot-ok' : 'lab-env-dot-off'}`} />
                <span className="lab-env-header-title">Environment</span>
              </div>
              <div className="lab-env-header-right">
                <span className={`lab-env-health-badge ${envData ? 'lab-env-health-ok' : 'lab-env-health-off'}`}>
                  {envData ? 'STABLE' : 'OFFLINE'}
                </span>
                <button
                  onClick={(e) => { e.stopPropagation(); fetchEnvironment(); }}
                  className="lab-env-refresh-btn"
                  title="Refresh"
                >
                  <RefreshCw size={10} className={isEnvLoading ? 'lab-spin' : ''} />
                </button>
              </div>
            </div>

            <div className="lab-env-body">
              {/* Status Row */}
              <div className="lab-env-row">
                <div className="lab-env-metric">
                  <span className="lab-env-metric-label">Runtime</span>
                  <span className="lab-env-metric-value">{envData?.python || 'Python'}</span>
                </div>
                <div className="lab-env-metric lab-env-metric-right">
                  <span className="lab-env-metric-label">Status</span>
                  <span className={`lab-env-metric-value ${envData ? 'lab-env-val-ok' : ''}`}>{envData?.status || 'Awaiting'}</span>
                </div>
              </div>

              {/* Packages Section */}
              <div className="lab-env-section">
                <div className="lab-env-section-header">
                  <span className="lab-env-section-title">
                    <Package size={10} />
                    Packages
                  </span>
                  <span className="lab-env-pkg-count">{(envData?.root_packages || envData?.packages || []).length}</span>
                </div>
                <div className="lab-env-pkg-list">
                  {(() => {
                    const all = isDeepScan ? (envData?.all_packages || []) : (envData?.root_packages || envData?.packages || []);
                    const sorted = [...all].sort((a, b) => {
                      const aActive = detectedDeps.includes(a.name);
                      const bActive = detectedDeps.includes(b.name);
                      if (aActive && !bActive) return -1;
                      if (!aActive && bActive) return 1;
                      return a.name.localeCompare(b.name);
                    });
                    if (sorted.length === 0) {
                      return <div className="lab-env-empty">No packages detected</div>;
                    }
                    return sorted.map((pkg, idx) => {
                      const isActive = detectedDeps.includes(pkg.name);
                      return (
                        <div key={idx} className={`lab-env-pkg-item ${isActive ? 'lab-env-pkg-active' : ''}`}>
                          <div className="lab-env-pkg-name">
                            {isActive && <Activity size={8} className="lab-env-pkg-pulse" />}
                            {pkg.name}
                          </div>
                          <span className="lab-env-pkg-ver">{pkg.version}</span>
                        </div>
                      );
                    });
                  })()}
                </div>
                <button onClick={() => setIsDeepScan(!isDeepScan)} className="lab-env-deepscan-btn">
                  {isDeepScan ? 'Show Root Only' : `Show All (${(envData?.all_packages?.length || 0)})`}
                </button>
              </div>

              {/* Dependencies Section */}
              <div className="lab-env-section">
                <div className="lab-env-section-header">
                  <span className="lab-env-section-title">
                    <Cpu size={10} />
                    Dependencies
                  </span>
                </div>
                <div className="lab-dep-badges">
                  {detectedDeps.length > 0 ? (
                    detectedDeps.map((dep, idx) => {
                      const allPkgs = envData?.all_packages || envData?.packages || [];
                      const isInstalled = allPkgs.some(p => p.name.toLowerCase() === dep.toLowerCase());
                      const isCurrentlyInstalling = isInstalling && installingPackages.includes(dep);
                      return (
                        <div
                          key={idx}
                          onClick={!isInstalled && !isCurrentlyInstalling ? () => runPipInstall(dep) : undefined}
                          className={`lab-dep-badge ${isInstalled ? 'lab-dep-installed' : isCurrentlyInstalling ? 'lab-dep-installing' : 'lab-dep-missing'}`}
                          title={isInstalled ? '✓ Installed' : isCurrentlyInstalling ? 'Installing...' : 'Click to install'}
                        >
                          {isCurrentlyInstalling ? (
                            <Loader2 size={8} className="lab-spin" />
                          ) : (
                            <div className={`lab-dep-dot ${isInstalled ? 'lab-dep-dot-ok' : 'lab-dep-dot-missing'}`} />
                          )}
                          <span>{dep}</span>
                          {!isInstalled && !isCurrentlyInstalling && <Download size={8} className="lab-dep-install-icon" />}
                          {isInstalled && <CheckCircle2 size={8} />}
                        </div>
                      );
                    })
                  ) : (
                    <span className="lab-dep-empty">No imports detected in source</span>
                  )}
                </div>
                {(() => {
                  const allPkgs = envData?.all_packages || envData?.packages || [];
                  const missing = detectedDeps.filter(dep => !allPkgs.some(p => p.name.toLowerCase() === dep.toLowerCase()));
                  if (missing.length === 0) return null;
                  return (
                    <button onClick={() => runPipInstall(missing)} disabled={isInstalling} className="lab-install-all-btn">
                      <Package size={12} />
                      <span>Install {missing.length} Missing</span>
                      {isInstalling && <Loader2 size={10} className="lab-spin" />}
                    </button>
                  );
                })()}
              </div>

              {/* Quick Tip */}
              <div className="lab-env-tip">
                <Zap size={10} />
                <span>Use <code>!pip install [pkg]</code> in the terminal to add packages.</span>
              </div>

              {/* Purge */}
              {envData && (
                <button onClick={handlePurgeSandbox} className="lab-env-purge-btn">Purge Sandbox</button>
              )}
            </div>
          </div>

          {/* ─── Vault Datasets Panel ─── */}
          <div className="lab-panel">
            <div className="lab-panel-header" style={{ padding: '8px 14px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Shield size={13} style={{ color: 'var(--primary)' }} />
                <span style={{ fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Vault Datasets</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ fontSize: '8px', fontWeight: 600, opacity: 0.4 }}>{vaultDatasets.length} ENCRYPTED</span>
                <button onClick={fetchVaultDatasets} style={{ background: 'none', border: 'none', cursor: 'pointer', opacity: 0.5, padding: '2px' }}>
                  <RefreshCw size={10} className={isVaultLoading ? 'lab-spin' : ''} style={{ color: 'var(--text-muted)' }} />
                </button>
              </div>
            </div>
            <div style={{ padding: '8px', maxHeight: '260px', overflowY: 'auto' }}>
              {vaultDatasets.length === 0 ? (
                <div style={{ padding: '20px 10px', textAlign: 'center', fontSize: '9px', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                  {isVaultLoading ? 'Loading encrypted datasets...' : 'No datasets in vault. Start STP server.'}
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  {vaultDatasets.map((ds, idx) => (
                    <div
                      key={idx}
                      onClick={() => insertVaultSnippet(ds.name)}
                      style={{
                        padding: '8px 10px',
                        background: copiedSnippet === ds.name ? 'rgba(34,197,94,0.08)' : 'rgba(255,255,255,0.02)',
                        border: copiedSnippet === ds.name ? '1px solid rgba(34,197,94,0.3)' : '1px solid var(--border)',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        transition: 'all 0.2s',
                      }}
                      onMouseEnter={e => { e.currentTarget.style.background = 'rgba(99,102,241,0.06)'; e.currentTarget.style.borderColor = 'rgba(99,102,241,0.2)'; }}
                      onMouseLeave={e => { e.currentTarget.style.background = copiedSnippet === ds.name ? 'rgba(34,197,94,0.08)' : 'rgba(255,255,255,0.02)'; e.currentTarget.style.borderColor = copiedSnippet === ds.name ? 'rgba(34,197,94,0.3)' : 'var(--border)'; }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                          <Lock size={9} style={{ color: 'var(--primary)', opacity: 0.7 }} />
                          <span style={{ fontSize: '10px', fontWeight: 700, color: 'var(--text-main)' }}>{ds.name}</span>
                        </div>
                        {copiedSnippet === ds.name ? (
                          <CheckCircle2 size={10} style={{ color: 'rgb(34,197,94)' }} />
                        ) : (
                          <Copy size={9} style={{ opacity: 0.3 }} />
                        )}
                      </div>
                      <div style={{ display: 'flex', gap: '8px', fontSize: '8px', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                        <span>{ds.samples?.toLocaleString()} samples</span>
                        <span>·</span>
                        <span>{ds.classes > 0 ? `${ds.classes} classes` : 'regression'}</span>
                        <span>·</span>
                        <span>shape {JSON.stringify(ds.shape)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
              <div style={{ marginTop: '8px', padding: '6px 8px', background: 'rgba(99,102,241,0.04)', borderRadius: '4px', display: 'flex', alignItems: 'flex-start', gap: '5px' }}>
                <Zap size={9} style={{ marginTop: '2px', color: 'var(--primary)', flexShrink: 0 }} />
                <span style={{ fontSize: '8px', color: 'var(--text-muted)', lineHeight: '1.4' }}>Click a dataset to insert <code style={{ background: 'rgba(255,255,255,0.05)', padding: '1px 3px', borderRadius: '2px' }}>vault.load()</code> into your code. Data decrypts in RAM only during training.</span>
              </div>
            </div>
          </div>

          {/* ─── Model Download Panel ─── */}
          <AnimatePresence>
            {status === 'COMPLETE' && (
              <motion.div
                className="lab-panel lab-download-panel"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
              >
                <div className="lab-panel-header lab-panel-header-success">
                  <CheckCircle2 size={14} />
                  <span>Model Assets Ready</span>
                </div>
                <div className="lab-panel-body lab-download-body">
                  <button className="lab-download-btn" onClick={() => handleDownload('pt')}>
                    <Download size={14} />
                    <span>Download Weights (.pt)</span>
                  </button>
                  <button className="lab-download-btn" onClick={() => handleDownload('onnx')}>
                    <BarChart2 size={14} />
                    <span>Export Model (.onnx)</span>
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* ─── Compiler Console ─── */}
          <div className="lab-panel lab-console-panel" style={{ height: consoleHeight, minHeight: 120 }}>
            <div className="lab-console-resize-handle" onMouseDown={startConsoleResize} />
            <div className="lab-panel-header">
              <Terminal size={14} />
              <span>Compiler Console</span>
              <button onClick={() => setLogs([])} className="lab-clear-btn">Clear</button>
            </div>
            <div className="lab-console-body" style={{ position: 'relative' }}>
              {/* ─── Package Installation Overlay ─── */}
              <AnimatePresence>
                {isInstalling && (
                  <motion.div
                    className="lab-install-overlay"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <div className="lab-install-header">
                      <div className="lab-install-header-left">
                        <Package size={14} />
                        <span>Package Manager</span>
                      </div>
                      <div className="lab-install-progress-text">{installProgress}%</div>
                    </div>
                    <div className="lab-install-bar-track">
                      <motion.div
                        className="lab-install-bar-fill"
                        initial={{ width: 0 }}
                        animate={{ width: `${installProgress}%` }}
                        transition={{ duration: 0.4, ease: 'easeOut' }}
                      />
                    </div>
                    <div className="lab-install-packages">
                      {installingPackages.map((pkg, i) => (
                        <div key={i} className="lab-install-pkg-row">
                          <div className="lab-install-pkg-left">
                            {installProgress >= ((i + 1) / installingPackages.length) * 100
                              ? <CheckCircle2 size={10} className="lab-install-pkg-done" />
                              : <Loader2 size={10} className="lab-spin lab-install-pkg-spin" />
                            }
                            <span className="lab-install-pkg-name">{pkg}</span>
                          </div>
                          <span className="lab-install-pkg-status">
                            {installProgress >= ((i + 1) / installingPackages.length) * 100 ? 'installed' : 'installing...'}
                          </span>
                        </div>
                      ))}
                    </div>
                    <div className="lab-install-log-area">
                      {installLogs.map((log, i) => (
                        <div key={i} className={`lab-install-log-line ${log.type === 'cmd' ? 'lab-ilog-cmd' : log.type === 'success' ? 'lab-ilog-success' : log.type === 'error' ? 'lab-ilog-error' : ''}`}>
                          {log.msg}
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Regular logs */}
              {logs.map((log, i) => (
                <div key={i} className="lab-log-line">
                  <span className="lab-log-time">[{log.time}]</span>
                  <span className={`lab-log-msg ${log.type === 'error' ? 'lab-log-error' : log.type === 'success' ? 'lab-log-success' : log.type === 'warning' ? 'lab-log-warn' : ''}`}>
                    {log.msg}
                  </span>
                </div>
              ))}
              {logs.length === 0 && !isInstalling && (
                <div className="lab-console-empty">Awaiting researcher interaction...</div>
              )}
              <div ref={terminalRef} />
            </div>

            {/* 🧪 Multi-line Interactive Terminal Input */}
            <div className="lab-terminal-input-wrapper">
              <div className="lab-terminal-prompt">
                <ChevronRight size={14} className={isEvaluating ? 'animate-pulse text-primary' : 'text-primary'} />
              </div>
              <textarea
                value={terminalInput}
                onChange={(e) => setTerminalInput(e.target.value)}
                onKeyDown={handleTerminalSubmit}
                placeholder={isEvaluating ? 'Executing command...' : 'Type code or !shell command...'}
                disabled={isEvaluating}
                className="lab-terminal-textarea"
                rows={terminalInput.split('\n').length || 1}
              />
              <div className="lab-terminal-hint">
                {terminalInput.includes('\n') ? 'ENTER to Run | SHIFT+ENTER for New Line' : 'ENTER to Run'}
              </div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        /* ═══════════════════════════════════════════════════════════
           LABORATORY SCOPED STYLES
           Uses project design tokens from :root
        ═══════════════════════════════════════════════════════════ */

        .lab-root {
          flex: 1;
          display: flex;
          flex-direction: column;
          height: 100%;
          overflow: hidden;
          background: var(--bg-surface);
          font-family: var(--font-sans);
        }

        /* ─── Header ─── */
        .lab-header {
          height: 56px;
          flex-shrink: 0;
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0 24px;
          border-bottom: 1px solid var(--border);
          background: var(--bg-surface);
        }
        .lab-header-left {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .lab-icon-box {
          width: 32px; height: 32px;
          display: flex; align-items: center; justify-content: center;
          background: var(--primary);
          color: #fff;
        }
        .lab-header-titles {
          display: flex;
          flex-direction: column;
          line-height: 1.2;
        }
        .lab-title {
          font-size: 12px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.15em;
          color: var(--text-main);
        }
        .lab-subtitle {
          font-size: 10px;
          font-family: var(--font-mono);
          color: var(--text-muted);
          letter-spacing: 0.05em;
        }
        .lab-status-pill {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 3px 12px;
          font-size: 9px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          color: var(--pill-color);
          background: color-mix(in srgb, var(--pill-color) 8%, transparent);
          border: 1px solid color-mix(in srgb, var(--pill-color) 20%, transparent);
          margin-left: 12px;
        }
        .lab-spin { animation: lab-spin-anim 1s linear infinite; }
        @keyframes lab-spin-anim { to { transform: rotate(360deg); } }

        .lab-header-actions {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        /* ─── Buttons ─── */
        .lab-btn {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          height: 36px;
          padding: 0 20px;
          font-size: 10px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          border: none;
          cursor: pointer;
          transition: all 0.15s ease;
          font-family: var(--font-sans);
        }
        .lab-btn:disabled {
          opacity: 0.35;
          cursor: not-allowed;
        }
        .lab-btn-outline {
          background: var(--bg-surface);
          color: var(--text-main);
          border: 1px solid var(--border);
        }
        .lab-btn-outline:hover:not(:disabled) {
          border-color: var(--primary);
          color: var(--primary);
        }
        .lab-btn-verified {
          background: #f0fdf4;
          border-color: var(--success) !important;
          color: var(--success) !important;
        }
        .lab-btn-primary {
          background: var(--primary);
          color: #fff;
        }
        .lab-btn-primary:hover:not(:disabled) {
          opacity: 0.9;
          transform: translateY(-1px);
        }
        .lab-btn-primary:active:not(:disabled) {
          transform: scale(0.97);
        }
        .lab-btn-danger {
          background: var(--error);
          color: #fff;
        }
        .lab-btn-danger:hover:not(:disabled) {
          opacity: 0.9;
        }

        /* ─── Progress Strip ─── */
        .lab-progress-strip {
          flex-shrink: 0;
          border-bottom: 1px solid var(--border);
          background: var(--bg-main);
          padding: 10px 24px;
          display: flex;
          align-items: center;
          gap: 24px;
          overflow: hidden;
        }
        .lab-progress-bar-track {
          flex: 1;
          height: 4px;
          background: var(--border);
          overflow: hidden;
        }
        .lab-progress-bar-fill {
          height: 100%;
          background: var(--primary);
          transition: width 0.5s ease;
        }
        .lab-progress-stats {
          display: flex;
          gap: 20px;
          flex-shrink: 0;
        }
        .lab-stat {
          display: flex;
          flex-direction: column;
          align-items: center;
          min-width: 60px;
        }
        .lab-stat-label {
          font-size: 8px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.15em;
          color: var(--text-muted);
        }
        .lab-stat-value {
          font-size: 13px;
          font-weight: 600;
          font-family: var(--font-mono);
          color: var(--text-main);
        }
        .lab-stat-loss { color: var(--error); }
        .lab-stat-acc { color: var(--success); }

        /* ─── Main Workspace ─── */
        .lab-workspace {
          flex: 1;
          display: flex;
          min-height: 0;
          overflow: hidden;
        }

        /* ─── Editor Panel ─── */
        .lab-editor-panel {
          flex: 1;
          display: flex;
          flex-direction: column;
          min-width: 0;
          border-right: 1px solid var(--border);
        }
        .lab-editor-wrap {
          flex: 1;
          display: flex;
          overflow: hidden;
          position: relative;
        }
        .lab-gutter {
          width: 52px;
          flex-shrink: 0;
          background: var(--bg-main);
          border-right: 1px solid var(--border);
          padding: 16px 0;
          overflow: hidden;
          user-select: none;
        }
        .lab-gutter-line {
          height: 24px;
          line-height: 24px;
          text-align: right;
          padding-right: 16px;
          font-family: var(--font-mono);
          font-size: 11px;
          color: #c4c8cc;
        }
        .lab-gutter-error {
          color: var(--error);
          font-weight: 700;
          background: rgba(239, 68, 68, 0.06);
        }
        .lab-gutter-syncing {
          background: color-mix(in srgb, var(--primary) 10%, transparent);
          color: var(--primary);
        }
        .lab-sync-indicator {
          display: flex;
          align-items: center;
          justify-content: center;
          height: 100%;
          color: var(--primary);
        }
        .lab-textarea {
          flex: 1;
          border: none;
          outline: none;
          resize: none;
          padding: 16px 20px;
          margin: 0;
          font-family: var(--font-mono);
          font-size: 13px;
          line-height: 24px;
          color: var(--text-main);
          background: var(--bg-surface);
          caret-color: var(--accent);
          tab-size: 4;
        }
        .lab-textarea::selection {
          background: color-mix(in srgb, var(--primary) 12%, transparent);
        }

        /* ═══ MAGIC COMMAND HINT ═══ */
        .lab-magic-hint {
          position: absolute;
          bottom: 16px;
          left: 64px;
          right: 16px;
          z-index: 50;
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 10px 16px;
          background: linear-gradient(135deg, #1e1b4b, #312e81);
          border: 1px solid rgba(129,140,248,0.35);
          border-radius: 8px;
          box-shadow: 0 8px 32px rgba(99,102,241,0.25), 0 0 0 1px rgba(99,102,241,0.1);
          backdrop-filter: blur(12px);
        }
        .lab-magic-hint-icon {
          width: 32px;
          height: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: rgba(99,102,241,0.2);
          border-radius: 6px;
          color: #a5b4fc;
          flex-shrink: 0;
          animation: lab-magic-pulse 2s ease-in-out infinite;
        }
        @keyframes lab-magic-pulse {
          0%, 100% { box-shadow: 0 0 8px rgba(99,102,241,0.3); }
          50% { box-shadow: 0 0 16px rgba(99,102,241,0.6); }
        }
        .lab-magic-hint-body {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 2px;
          min-width: 0;
        }
        .lab-magic-hint-title {
          font-size: 11px;
          font-weight: 700;
          color: #e0e7ff;
          letter-spacing: 0.02em;
        }
        .lab-magic-hint-sub {
          font-size: 10px;
          color: rgba(165,180,252,0.7);
        }
        .lab-magic-hint-sub code {
          color: #a5b4fc;
          font-weight: 700;
          font-family: var(--font-mono);
          background: rgba(99,102,241,0.15);
          padding: 1px 6px;
          border-radius: 3px;
          font-size: 11px;
        }
        .lab-magic-hint-actions {
          display: flex;
          gap: 6px;
          flex-shrink: 0;
        }
        .lab-magic-btn {
          padding: 6px 14px;
          font-size: 10px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          border: none;
          border-radius: 5px;
          cursor: pointer;
          transition: all 0.15s;
          font-family: var(--font-sans);
          display: flex;
          align-items: center;
          gap: 5px;
        }
        .lab-magic-btn-run {
          background: #6366f1;
          color: #fff;
        }
        .lab-magic-btn-run:hover {
          background: #7c3aed;
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(99,102,241,0.4);
        }
        .lab-magic-hint-warn {
          background: linear-gradient(135deg, #7c2d12, #9a3412);
          border-color: rgba(251,191,36,0.4);
          box-shadow: 0 8px 32px rgba(245,158,11,0.2);
        }
        .lab-magic-hint-warn .lab-magic-hint-icon {
          background: rgba(251,191,36,0.25);
          color: #fcd34d;
          animation: none;
        }
        .lab-magic-hint-warn .lab-magic-hint-title {
          color: #fef3c7;
        }
        .lab-magic-hint-warn .lab-magic-hint-sub {
          color: rgba(254,243,199,0.7);
        }
        .lab-magic-suggestions {
          display: flex;
          flex-wrap: wrap;
          gap: 4px;
        }
        .lab-magic-suggestion {
          padding: 4px 12px;
          background: rgba(255,255,255,0.12);
          border: 1px solid rgba(255,255,255,0.2);
          border-radius: 4px;
          color: #e0e7ff;
          font-family: var(--font-mono);
          font-size: 10px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.15s;
          display: inline-flex;
          align-items: center;
          gap: 4px;
        }
        .lab-magic-suggestion:hover {
          background: rgba(255,255,255,0.2);
          border-color: rgba(255,255,255,0.35);
          color: #fff;
          transform: translateY(-1px);
          box-shadow: 0 2px 8px rgba(99,102,241,0.3);
        }

        /* ─── Side Panel ─── */
        .lab-sidebar {
          width: 340px;
          flex-shrink: 0;
          display: flex;
          flex-direction: column;
          background: var(--bg-main);
          overflow-y: auto;
        }
        .lab-sidebar::-webkit-scrollbar { width: 4px; }
        .lab-sidebar::-webkit-scrollbar-track { background: transparent; }
        .lab-sidebar::-webkit-scrollbar-thumb { background: var(--border); }

        /* ═══ Environment Panel ═══ */
        .lab-env-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 12px 16px;
          background: var(--bg-surface);
          border-bottom: 1px solid var(--border);
        }
        .lab-env-header-left {
          display: flex;
          align-items: center;
          gap: 10px;
        }
        .lab-env-status-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          flex-shrink: 0;
        }
        .lab-env-dot-ok {
          background: var(--success);
          box-shadow: 0 0 8px rgba(16,185,129,0.4);
          animation: lab-env-glow 2s ease-in-out infinite;
        }
        .lab-env-dot-off {
          background: var(--border);
        }
        @keyframes lab-env-glow {
          0%, 100% { box-shadow: 0 0 6px rgba(16,185,129,0.3); }
          50% { box-shadow: 0 0 14px rgba(16,185,129,0.6); }
        }
        .lab-env-header-title {
          font-size: 11px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.15em;
          color: var(--text-main);
        }
        .lab-env-header-right {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .lab-env-health-badge {
          font-size: 8px;
          font-weight: 800;
          letter-spacing: 0.12em;
          padding: 2px 8px;
          border-radius: 3px;
          text-transform: uppercase;
        }
        .lab-env-health-ok {
          color: var(--success);
          background: rgba(16,185,129,0.08);
          border: 1px solid rgba(16,185,129,0.2);
        }
        .lab-env-health-off {
          color: var(--text-muted);
          background: rgba(100,116,139,0.06);
          border: 1px solid var(--border);
        }
        .lab-env-refresh-btn {
          width: 26px;
          height: 26px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 4px;
          border: 1px solid var(--border);
          background: var(--bg-main);
          color: var(--text-muted);
          cursor: pointer;
          transition: all 0.2s;
        }
        .lab-env-refresh-btn:hover {
          background: var(--primary);
          color: white;
          border-color: var(--primary);
        }
        .lab-env-body {
          padding: 14px 16px;
          display: flex;
          flex-direction: column;
          gap: 16px;
          background: var(--bg-main);
        }
        .lab-env-row {
          display: flex;
          gap: 10px;
        }
        .lab-env-metric {
          flex: 1;
          padding: 10px 12px;
          background: var(--bg-surface);
          border: 1px solid var(--border);
          border-radius: 6px;
          display: flex;
          flex-direction: column;
          gap: 4px;
        }
        .lab-env-metric-right {
          text-align: right;
          align-items: flex-end;
        }
        .lab-env-metric-label {
          font-size: 8px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          color: var(--text-muted);
        }
        .lab-env-metric-value {
          font-size: 12px;
          font-weight: 600;
          font-family: var(--font-mono);
          color: var(--text-main);
        }
        .lab-env-val-ok { color: var(--success); }
        .lab-env-section {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .lab-env-section-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
        }
        .lab-env-section-title {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 9px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          color: var(--text-muted);
        }
        .lab-env-pkg-count {
          font-size: 9px;
          font-weight: 700;
          font-family: var(--font-mono);
          color: var(--primary);
          background: rgba(54,78,104,0.06);
          padding: 2px 8px;
          border-radius: 10px;
        }
        .lab-env-pkg-list {
          max-height: 180px;
          overflow-y: auto;
          display: flex;
          flex-direction: column;
          gap: 1px;
          border-radius: 6px;
          border: 1px solid var(--border);
          background: var(--bg-surface);
        }
        .lab-env-pkg-list::-webkit-scrollbar { width: 3px; }
        .lab-env-pkg-list::-webkit-scrollbar-track { background: transparent; }
        .lab-env-pkg-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
        .lab-env-pkg-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 6px 10px;
          transition: background 0.15s;
          border-bottom: 1px solid rgba(226,230,236,0.5);
        }
        .lab-env-pkg-item:last-child { border-bottom: none; }
        .lab-env-pkg-item:hover {
          background: rgba(54,78,104,0.03);
        }
        .lab-env-pkg-active {
          background: rgba(16,185,129,0.04);
          border-left: 2px solid var(--success);
        }
        .lab-env-pkg-name {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 10px;
          font-family: var(--font-mono);
          font-weight: 500;
          color: var(--text-main);
        }
        .lab-env-pkg-active .lab-env-pkg-name {
          color: var(--success);
          font-weight: 700;
        }
        .lab-env-pkg-pulse {
          color: var(--success);
        }
        .lab-env-pkg-ver {
          font-size: 9px;
          font-family: var(--font-mono);
          color: var(--text-muted);
          opacity: 0.6;
        }
        .lab-env-empty {
          padding: 24px;
          text-align: center;
          font-size: 10px;
          color: var(--text-muted);
          font-style: italic;
        }
        .lab-env-deepscan-btn {
          width: 100%;
          padding: 6px;
          background: transparent;
          border: 1px dashed var(--border);
          border-radius: 4px;
          color: var(--text-muted);
          font-size: 9px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
          text-transform: uppercase;
          letter-spacing: 0.08em;
        }
        .lab-env-deepscan-btn:hover {
          border-color: var(--primary);
          color: var(--primary);
          background: rgba(54,78,104,0.03);
        }
        .lab-env-tip {
          display: flex;
          align-items: flex-start;
          gap: 8px;
          padding: 10px 12px;
          background: rgba(54,78,104,0.04);
          border: 1px solid rgba(54,78,104,0.1);
          border-radius: 6px;
          color: var(--text-muted);
          font-size: 9px;
          line-height: 1.5;
        }
        .lab-env-tip svg {
          color: var(--primary);
          flex-shrink: 0;
          margin-top: 1px;
        }
        .lab-env-tip code {
          font-family: var(--font-mono);
          color: var(--primary);
          font-weight: 700;
          background: rgba(54,78,104,0.08);
          padding: 1px 5px;
          border-radius: 3px;
          font-size: 9px;
        }
        .lab-env-purge-btn {
          width: 100%;
          padding: 7px;
          background: rgba(239,68,68,0.04);
          border: 1px solid rgba(239,68,68,0.15);
          border-radius: 4px;
          color: var(--error);
          font-size: 9px;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          cursor: pointer;
          transition: all 0.2s;
          opacity: 0.6;
        }
        .lab-env-purge-btn:hover {
          background: rgba(239,68,68,0.08);
          border-color: rgba(239,68,68,0.3);
          opacity: 1;
        }

        .lab-panel {
          border-bottom: 1px solid var(--border);
        }
        .lab-panel-header {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 14px 20px;
          font-size: 10px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.18em;
          color: var(--text-muted);
          border-bottom: 1px solid var(--border);
          background: var(--bg-surface);
        }
        .lab-panel-header-success {
          color: var(--success);
          background: #f0fdf4;
        }
        .lab-panel-body {
          padding: 20px;
        }

        /* ─── Parameters ─── */
        .lab-param-group {
          margin-bottom: 16px;
        }
        .lab-param-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
        }
        .lab-param-label {
          font-size: 9px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.15em;
          color: var(--text-muted);
        }
        .lab-param-badge {
          font-size: 11px;
          font-weight: 700;
          font-family: var(--font-mono);
          color: var(--primary);
          background: color-mix(in srgb, var(--primary) 8%, transparent);
          padding: 2px 10px;
        }
        .link-indicator {
          width: 4px; height: 4px;
          border-radius: 50%;
          background: var(--border);
          transition: all 0.2s;
        }
        .link-indicator.active {
          background: var(--primary);
          box-shadow: 0 0 5px var(--primary);
        }
        .lab-slider {
          width: 100%;
          height: 4px;
          appearance: none;
          -webkit-appearance: none;
          background: var(--border);
          outline: none;
          cursor: pointer;
        }
        .lab-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 14px; height: 14px;
          background: var(--primary);
          border-radius: 50%;
          cursor: pointer;
        }
        .lab-param-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
          margin-bottom: 16px;
        }
        .lab-select {
          width: 100%;
          height: 36px;
          padding: 0 10px;
          font-size: 12px;
          font-family: var(--font-mono);
          border: 1px solid var(--border);
          background: var(--bg-surface);
          color: var(--text-main);
          outline: none;
          cursor: pointer;
          transition: border-color 0.15s;
        }
        .lab-select:focus {
          border-color: var(--primary);
        }

        .lab-env-row {
          display: flex;
          gap: 20px;
          padding-top: 14px;
          border-top: 1px solid var(--border);
        }
        .lab-env-row > div {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }
        .lab-env-label {
          font-size: 8px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.15em;
          color: var(--text-muted);
        }
        .lab-env-value {
          font-size: 11px;
          font-weight: 600;
          font-family: var(--font-mono);
          color: var(--text-main);
        }
        .lab-env-device {
          color: var(--success);
        }

        /* ─── Download Panel ─── */
        .lab-download-body {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .lab-download-btn {
          display: flex;
          align-items: center;
          gap: 10px;
          width: 100%;
          height: 40px;
          padding: 0 16px;
          background: var(--bg-surface);
          border: 1px solid color-mix(in srgb, var(--success) 30%, transparent);
          color: #166534;
          font-size: 10px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          cursor: pointer;
          transition: all 0.15s;
          font-family: var(--font-sans);
        }
        .lab-download-btn:hover {
          background: #f0fdf4;
          border-color: var(--success);
          transform: translateX(2px);
        }

        /* ─── Console ─── */
        .lab-console-panel {
          display: flex;
          flex-direction: column;
          min-height: 120px;
          flex-shrink: 0;
          position: relative;
        }
        .lab-console-resize-handle {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 6px;
          cursor: ns-resize;
          z-index: 10;
          background: transparent;
          transition: background 0.15s;
        }
        .lab-console-resize-handle:hover,
        .lab-console-resize-handle:active {
          background: var(--primary);
          opacity: 0.4;
        }
        .lab-clear-btn {
          margin-left: auto;
          font-size: 9px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          color: var(--text-muted);
          background: none;
          border: none;
          cursor: pointer;
          padding: 2px 6px;
          transition: color 0.15s;
          font-family: var(--font-sans);
        }
        .lab-clear-btn:hover {
          color: var(--text-main);
        }
        .lab-console-body {
          flex: 1;
          overflow-y: auto;
          overflow-x: hidden;
          padding: 12px 20px;
          font-family: var(--font-mono);
          font-size: 10px;
          line-height: 22px;
          background: var(--bg-surface);
        }
        .lab-console-body::-webkit-scrollbar { width: 3px; }
        .lab-console-body::-webkit-scrollbar-track { background: transparent; }
        .lab-console-body::-webkit-scrollbar-thumb { background: var(--border); }

        .lab-log-line {
          display: flex;
          gap: 10px;
          align-items: start;
        }
        .lab-log-time {
          color: #c4c8cc;
          flex-shrink: 0;
          font-size: 9px;
          padding-top: 1px;
          user-select: none;
        }
        .lab-log-msg {
          color: var(--text-main);
          opacity: 0.7;
          word-break: break-word;
        }
        .lab-log-error {
          color: var(--error);
          opacity: 1;
          font-weight: 600;
        }
        .lab-log-success {
          color: var(--success);
          opacity: 1;
          font-weight: 600;
        }
        .lab-log-warn {
          color: var(--warning);
          opacity: 1;
        }
        .lab-log-user {
          color: var(--primary);
          opacity: 1;
          background: rgba(0, 255, 65, 0.05);
          padding: 0 4px;
          border-radius: 2px;
          font-weight: 700;
        }
         .lab-console-empty {
          color: #c4c8cc;
          font-style: italic;
          text-align: center;
          padding: 40px 0;
          font-size: 10px;
          text-transform: uppercase;
          letter-spacing: 0.12em;
        }

        /* ═══ PACKAGE INSTALLATION OVERLAY ═══ */
        .lab-install-overlay {
          position: absolute;
          inset: 0;
          z-index: 20;
          background: #0c0f18;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }
        .lab-install-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 12px 16px;
          background: linear-gradient(135deg, rgba(79,70,229,0.12), rgba(16,185,129,0.08));
          border-bottom: 1px solid rgba(255,255,255,0.06);
          flex-shrink: 0;
        }
        .lab-install-header-left {
          display: flex;
          align-items: center;
          gap: 8px;
          color: #a78bfa;
          font-size: 11px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.15em;
        }
        .lab-install-progress-text {
          font-family: var(--font-mono);
          font-size: 13px;
          font-weight: 800;
          color: #10b981;
          letter-spacing: 0.05em;
        }
        .lab-install-bar-track {
          height: 3px;
          background: rgba(255,255,255,0.06);
          flex-shrink: 0;
        }
        .lab-install-bar-fill {
          height: 100%;
          background: linear-gradient(90deg, #6366f1, #10b981);
          border-radius: 0 2px 2px 0;
        }
        .lab-install-packages {
          padding: 12px 16px;
          display: flex;
          flex-direction: column;
          gap: 6px;
          flex-shrink: 0;
          border-bottom: 1px solid rgba(255,255,255,0.04);
        }
        .lab-install-pkg-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 6px 10px;
          background: rgba(255,255,255,0.02);
          border: 1px solid rgba(255,255,255,0.04);
        }
        .lab-install-pkg-left {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .lab-install-pkg-done { color: #10b981; }
        .lab-install-pkg-spin { color: #a78bfa; }
        .lab-install-pkg-name {
          font-family: var(--font-mono);
          font-size: 11px;
          font-weight: 600;
          color: rgba(255,255,255,0.85);
        }
        .lab-install-pkg-status {
          font-size: 9px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          color: rgba(255,255,255,0.3);
        }
        .lab-install-log-area {
          flex: 1;
          overflow-y: auto;
          padding: 10px 16px;
          font-family: var(--font-mono);
          font-size: 10px;
          line-height: 20px;
        }
        .lab-install-log-area::-webkit-scrollbar { width: 3px; }
        .lab-install-log-area::-webkit-scrollbar-track { background: transparent; }
        .lab-install-log-area::-webkit-scrollbar-thumb { background: #1e293b; }
        .lab-install-log-line {
          color: rgba(255,255,255,0.35);
          white-space: pre-wrap;
        }
        .lab-ilog-cmd {
          color: #a78bfa;
          font-weight: 700;
          padding: 4px 0;
        }
        .lab-ilog-success {
          color: #10b981;
          font-weight: 600;
        }
        .lab-ilog-error {
          color: #ef4444;
          font-weight: 600;
        }

        /* ═══ DEPENDENCY BADGES ═══ */
        .lab-dep-section-title {
          font-size: 8px;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.2em;
          color: rgba(255,255,255,0.25);
          margin-bottom: 4px;
        }
        .lab-dep-badges {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
          margin-bottom: 8px;
        }
        .lab-dep-badge {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 4px 10px;
          font-size: 10px;
          font-family: var(--font-mono);
          font-weight: 600;
          border: 1px solid;
          cursor: pointer;
          transition: all 0.2s;
          user-select: none;
        }
        .lab-dep-installed {
          color: var(--primary);
          border-color: rgba(0,255,65,0.15);
          background: rgba(0,255,65,0.05);
          cursor: default;
        }
        .lab-dep-missing {
          color: #f87171;
          border-color: rgba(248,113,113,0.25);
          background: rgba(248,113,113,0.06);
        }
        .lab-dep-missing:hover {
          background: rgba(248,113,113,0.12);
          border-color: rgba(248,113,113,0.4);
          transform: translateY(-1px);
        }
        .lab-dep-installing {
          color: #a78bfa;
          border-color: rgba(167,139,250,0.25);
          background: rgba(167,139,250,0.06);
          cursor: wait;
        }
        .lab-dep-dot {
          width: 5px; height: 5px;
          border-radius: 50%;
          flex-shrink: 0;
        }
        .lab-dep-dot-ok { background: var(--primary); }
        .lab-dep-dot-missing {
          background: #f87171;
          animation: lab-dep-pulse 1.5s infinite;
        }
        @keyframes lab-dep-pulse {
          0%, 100% { opacity: 1; box-shadow: 0 0 4px rgba(248,113,113,0.4); }
          50% { opacity: 0.4; box-shadow: none; }
        }
        .lab-dep-install-icon {
          opacity: 0.4;
          transition: opacity 0.2s;
        }
        .lab-dep-missing:hover .lab-dep-install-icon {
          opacity: 1;
        }
        .lab-dep-empty {
          font-size: 9px;
          color: var(--text-muted);
          font-style: italic;
        }
        .lab-install-all-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          width: 100%;
          padding: 10px 16px;
          background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(236,72,153,0.08));
          border: 1px solid rgba(99,102,241,0.25);
          color: #a78bfa;
          font-size: 10px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          cursor: pointer;
          transition: all 0.2s;
          font-family: var(--font-sans);
        }
        .lab-install-all-btn:hover:not(:disabled) {
          background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(236,72,153,0.15));
          border-color: rgba(99,102,241,0.4);
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(99,102,241,0.15);
        }
        .lab-install-all-btn:disabled {
          opacity: 0.4;
          cursor: wait;
        }

        /* 🧪 Research Shell Terminal UI Styling */
        .lab-terminal-input-wrapper {
          padding: 12px 20px;
          background: rgba(0,0,0,0.4);
          border-top: 1px solid rgba(255,255,255,0.03);
          display: flex;
          flex-direction: column;
          gap: 4px;
        }
        .lab-terminal-prompt {
          display: flex;
          align-items: center;
          color: var(--primary);
        }
        .lab-terminal-textarea {
          width: 100%;
          background: transparent;
          border: none;
          outline: none;
          color: white;
          font-family: var(--font-mono);
          font-size: 11px;
          line-height: 1.5;
          resize: none;
          padding: 4px 0;
        }
        .lab-terminal-textarea::placeholder {
          color: #444;
          font-style: italic;
        }
        .lab-terminal-hint {
          font-size: 8px;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          color: #333;
          text-align: right;
          font-weight: 700;
        }

        /* 📱 Market-Ready Responsiveness */
        @media (max-width: 1024px) {
          .lab-workspace {
            flex-direction: column;
            overflow-y: auto;
          }
          .lab-sidebar {
            width: 100%;
            border-left: none;
            border-top: 1px solid var(--border);
            height: auto;
            flex-shrink: 0;
          }
          .lab-editor-panel {
            min-height: 500px;
            flex: none;
          }
          .lab-console-panel {
            min-height: 300px;
          }
        }
      `}</style>
    </div>
  );
}
