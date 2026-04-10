import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Code, Play, ShieldCheck, Terminal, Zap,
  AlertCircle, RefreshCw, Download, BarChart2,
  Activity, Settings, CheckCircle2, StopCircle, Cpu, Box,
  ChevronRight
} from 'lucide-react';
import { API_BASE_URL } from '../hooks/useSecureFederated';

const DEFAULT_MODEL_CODE = `# Federated Learning - Institutional Sandbox v1.5
# ----------------------------------------------
# Type '!pip install [package]' to add libraries.
# Type '!ls' to scout the filesystem.
# ----------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
  from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# -- - Standard Implementation-- -

  class SecureNodeModel(nn.Module):
    def __init__(self):
super(SecureNodeModel, self).__init__()
self.conv1 = nn.Conv2d(1, 32, 3, 1)
self.conv2 = nn.Conv2d(32, 64, 3, 1)
self.dropout1 = nn.Dropout(0.25)
self.fc1 = nn.Linear(9216, 128)
self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
x = torch.relu(self.conv1(x))
x = torch.relu(self.conv2(x))
x = torch.max_pool2d(x, 2)
x = self.dropout1(x)
x = torch.flatten(x, 1)
x = torch.relu(self.fc1(x))
return self.fc2(x)

def train(epochs = 5, lr = 0.01, batch_size = 32):
    # Dummy function for AST detection
    pass

print("🧪 Federated Sandbox Ready.")
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
      addLog(`Backend Error: ${ labState.error } `, 'error');
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

  useEffect(() => {
    fetchEnvironment();
    const interval = setInterval(fetchEnvironment, 45000); // 🧪 Reduced polling frequency as requested
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
        addLog(`[${ data.type || 'Error' }] Line ${ data.line || '?' }: ${ data.error } `, 'error');
      }
    } catch (err) {
      setStatus('ERROR');
      addLog('FATAL: Could not connect to Institutional Compiler Service.', 'error');
    }
  };

  const handleTrain = async () => {
    if (status !== 'READY' && status !== 'COMPLETE' && status !== 'IDLE') {
      onAction('Code must be verified before training.', 'error');
      return;
    }
    setStatus('TRAINING');
    addLog(`Starting training(Epochs: ${ epochs }, LR: ${ lr }, Batch: ${ batchSize })...`, 'info');
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/laboratory/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, hyperparams: { epochs, lr, batch_size: batchSize } })
      });
      const data = await response.json();
      if (!data.success) {
        setStatus('ERROR');
        addLog(`Failed to start training: ${ data.message } `, 'error');
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
    TRAINING: labState?.mode === 'SCRIPT' ? 'Executing standalone script...' : `Training — Epoch ${ labState?.epoch || 0 }/${epochs}`,
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

          {/* 🪄 Magic Command Suggester Hint */}
          {(() => {
            const lines = code.split('\n');
            const invalidMagic = lines.find(l => {
              const trimmed = l.trim();
              return (trimmed.startsWith('pip ') || trimmed.startsWith('conda ') || trimmed.startsWith('ls ') || trimmed.startsWith('python ')) && !l.startsWith('!');
            });
            if (invalidMagic) {
              return (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="absolute bottom-4 left-20 right-20 bg-primary/95 text-white p-3 border border-white/20 shadow-2xl z-50 flex items-center justify-between"
                >
                  <div className="flex items-center gap-3">
                    <Zap size={14} className="animate-pulse" />
                    <span className="text-[10px] font-bold tracking-widest uppercase font-mono">
                      Institutional Hint: Did you mean "!{invalidMagic.trim()}"?
                    </span>
                  </div>
                  <span className="text-[8px] opacity-60 uppercase font-bold tracking-tight">CLI Magic commands require prefix [!]</span>
                </motion.div>
              );
            }
            return null;
          })()}
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

        {/* 🧪 Environment Monitor (Stitch System) */}
        <div className="lab-panel lab-env-panel">
          <div className="lab-panel-header flex justify-between items-center group">
            <div className="flex items-center gap-2">
              <Box size={14} className="text-primary" />
              <span>Sandbox Environment</span>
            </div>
            <button
              onClick={fetchEnvironment}
              title="Refresh Environment"
              className="opacity-0 group-hover:opacity-100 transition-opacity"
            >
              <RefreshCw size={10} className={isEnvLoading ? 'lab-spin' : ''} />
            </button>
          </div>
          <div className="p-4 bg-dark-grounding/20 border-t border-border/5 space-y-3">
            <div className="flex justify-between items-center">
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-text-muted font-bold">Sandbox Health</span>
                <span className={`text-[11px] font-mono ${envData ? 'text-primary' : 'text-gray-500'}`}>
                  {envData ? 'INITIALIZED_STABLE' : 'NOT_INITIALIZED / PURGED'}
                </span>
              </div>
              {envData && (
                <button
                  onClick={handlePurgeSandbox}
                  className="px-2 py-1 bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/30 text-[9px] font-mono uppercase transition-all"
                >
                  Purge_Storage
                </button>
              )}
            </div>
            <div className="flex gap-2">
              <button
                onClick={fetchEnvironment}
                disabled={isEnvLoading}
                className="flex-1 py-1.5 bg-primary/5 hover:bg-primary/10 text-primary border border-primary/20 text-[10px] font-bold uppercase transition-all flex items-center justify-center gap-2"
              >
                <RefreshCw size={10} className={isEnvLoading ? 'lab-spin' : ''} />
                {isEnvLoading ? 'SCANNING...' : 'Refresh_Health'}
              </button>
            </div>
          </div>
          <div className="lab-panel-body">
            <div className="flex flex-col gap-4">
              <div className="flex justify-between items-center pb-3 border-b border-[#222]">
                <div className="flex flex-col">
                  <span className="text-[8px] text-primary font-bold tracking-widest uppercase">STATUS</span>
                  <span className="text-[11px] font-mono text-white">{envData?.status || 'AWAITING_INIT...'}</span>
                </div>
                <div className="flex flex-col text-right">
                  <span className="text-[8px] text-[#444] font-bold tracking-widest uppercase">PYTHON</span>
                  <span className="text-[11px] font-mono text-white">{envData?.python || '...'}</span>
                </div>
              </div>

              <div className="flex flex-col gap-2">
                <span className="text-[8px] text-[#444] font-bold tracking-widest mb-1">INSTALLED_PACKAGES</span>
                <div className="max-h-[240px] overflow-y-auto pr-2 space-y-1">
                  {(() => {
                    const all = isDeepScan ? (envData?.all_packages || []) : (envData?.root_packages || envData?.packages || []);

                    // Sort: Active imports at top
                    const sorted = [...all].sort((a, b) => {
                      const aActive = detectedDeps.includes(a.name);
                      const bActive = detectedDeps.includes(b.name);
                      if (aActive && !bActive) return -1;
                      if (!aActive && bActive) return 1;
                      return a.name.localeCompare(b.name);
                    });

                    return sorted.map((pkg, idx) => {
                      const isActive = detectedDeps.includes(pkg.name);
                      return (
                        <div key={idx} className={`flex justify-between items-center p-2 bg-[#050505] border ${isActive ? 'border-primary shadow-[0_0_5px_rgba(0,255,65,0.1)]' : 'border-[#111]'} transition-all`}>
                          <div className="flex items-center gap-2">
                            {isActive && <div className="w-1 h-1 rounded-full bg-primary animate-pulse"></div>}
                            <span className={`text-[10px] font-mono ${isActive ? 'text-primary font-bold' : 'text-gray-400'}`}>
                              {pkg.name}
                            </span>
                          </div>
                          <span className="text-[10px] text-[#444] font-mono">{pkg.version}</span>
                        </div>
                      );
                    });
                  })()}

                  {(!envData?.packages || envData.packages.length === 0) && (
                    <div className="text-[9px] text-[#333] italic py-4 text-center">INITIALIZING_SCOUT...</div>
                  )}
                </div>

                <button
                  onClick={() => setIsDeepScan(!isDeepScan)}
                  className="w-full mt-2 py-1 border border-border/30 hover:border-primary/50 text-[8px] font-mono text-gray-500 hover:text-primary transition-all uppercase tracking-widest text-center"
                >
                  {isDeepScan ? '[ DEACTIVATE_DEEP_SCAN ]' : `[ SHOW_${(envData?.all_packages?.length || 0) - (envData?.root_packages?.length || 0)}_DEPENDENCY_NODES ]`}
                </button>
              </div>

              <div className="flex flex-col gap-2">
                <span className="text-[8px] text-[#444] font-bold tracking-widest mb-1">DETECTED_DEPENDENCIES</span>
                <div className="flex flex-wrap gap-1 mb-2">
                  {detectedDeps.length > 0 ? (
                    detectedDeps.map((dep, idx) => {
                      const isInstalled = envData?.packages?.some(p => p.name.toLowerCase() === dep.toLowerCase());
                      return (
                        <div
                          key={idx}
                          onClick={!isInstalled ? () => onExecuteCommand(`!pip install ${dep}`) : undefined}
                          className={`flex items-center gap-1.5 px-2 py-1 border text-[9px] font-mono cursor-pointer transition-all ${isInstalled
                              ? 'bg-primary/10 border-primary/20 text-primary'
                              : 'bg-red-500/10 border-red-500/30 text-red-500 hover:bg-red-500/20'
                            }`}
                          title={isInstalled ? 'Library Available' : 'Click to Auto-Install'}
                        >
                          <div className={`w-1 h-1 rounded-full ${isInstalled ? 'bg-primary' : 'bg-red-500 animate-pulse'}`}></div>
                          {dep}
                        </div>
                      );
                    })
                  ) : (
                    <span className="text-[9px] text-[#333] italic">NO_IMPORTS_DETECTED</span>
                  )}
                </div>
                {detectedDeps.some(dep => !envData?.packages?.some(p => p.name.toLowerCase() === dep.toLowerCase())) && (
                  <button
                    onClick={() => {
                      const missing = detectedDeps.filter(dep => !envData?.packages?.some(p => p.name.toLowerCase() === dep.toLowerCase()));
                      onExecuteCommand(`!pip install ${missing.join(' ')}`);
                    }}
                    className="text-[8px] text-red-500 font-bold hover:underline text-left"
                  >
                    [→] INSTALL_ALL_MISSING_MODULES
                  </button>
                )}
              </div>

              <div className="p-3 bg-primary/5 border border-primary/20">
                <div className="flex items-center gap-2 mb-1">
                  <Zap size={10} className="text-primary" />
                  <span className="text-[9px] font-bold text-primary italic tracking-tight">VENV_IS_ACTIVE</span>
                </div>
                <p className="text-[9px] leading-relaxed text-[#666]">
                  Use <span className="text-primary font-bold">!pip install [package]</span> to install libraries directly into the sandbox.
                </p>
              </div>
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
        <div className="lab-panel lab-console-panel">
          <div className="lab-panel-header">
            <Terminal size={14} />
            <span>Compiler Console</span>
            <button onClick={() => setLogs([])} className="lab-clear-btn">Clear</button>
          </div>
          <div className="lab-console-body">
            {logs.map((log, i) => (
              <div key={i} className="lab-log-line">
                <span className="lab-log-time">[{log.time}]</span>
                <span className={`lab-log-msg ${log.type === 'error' ? 'lab-log-error' : log.type === 'success' ? 'lab-log-success' : log.type === 'warning' ? 'lab-log-warn' : ''}`}>
                  {log.msg}
                </span>
              </div>
            ))}
            {logs.length === 0 && (
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
          flex: 1;
          display: flex;
          flex-direction: column;
          min-height: 200px;
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
