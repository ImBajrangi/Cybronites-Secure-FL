import React, { useEffect, useState } from 'react';
import { 
  ShieldCheck, Lock, Database, Activity, CheckCircle2, AlertCircle, RefreshCcw,
  Zap, Fingerprint, LockKeyhole, Eye, EyeOff, Shield, Server, Cpu, Key
} from 'lucide-react';
import { useSecureTraining } from '../hooks/useSecureTraining';
import { motion, AnimatePresence } from 'framer-motion';
import { API_BASE_URL } from '../hooks/useSecureFederated';

export const PrivacyVault = () => {
    const { 
        datasets, jobs, models, loading, error, 
        fetchDatasets, fetchJobs, fetchModels, submitJob 
    } = useSecureTraining();

    const [selectedDataset, setSelectedDataset] = useState(null);
    const [isLaunching, setIsLaunching] = useState(false);
    const [initError, setInitError] = useState(false);
    const [showKey, setShowKey] = useState(false);
    const [fedStatus, setFedStatus] = useState(null);

    useEffect(() => {
        const init = async () => {
            try {
                await Promise.allSettled([fetchDatasets(), fetchJobs(), fetchModels()]);
            } catch {
                setInitError(true);
            }
        };
        init();
        // Also fetch federated status
        fetch(`${API_BASE_URL}/api/health`).then(r => r.json()).then(setFedStatus).catch(() => {});
    }, [fetchDatasets, fetchJobs, fetchModels]);

    const handleLaunch = async () => {
        if (!selectedDataset) return;
        setIsLaunching(true);
        try {
            await submitJob(selectedDataset.id, "SimpleCNN", {
                epochs: 5, batch_size: 32, learning_rate: 0.001
            });
            setSelectedDataset(null);
        } catch (err) {
            console.error(err);
        } finally {
            setIsLaunching(false);
        }
    };

    // Privacy specs used in the FL system
    const privacySpecs = [
        { label: 'Differential Privacy', value: 'ε-DP Enabled', status: 'active', icon: Shield },
        { label: 'L2 Norm Clip', value: '1.0', status: 'active', icon: Zap },
        { label: 'Noise Multiplier', value: '0.01', status: 'active', icon: Activity },
        { label: 'Encryption Standard', value: 'AES-256-GCM', status: 'active', icon: Lock },
        { label: 'Aggregation', value: 'Secure Median', status: 'active', icon: Server },
        { label: 'Memory Wipe', value: 'Post-Round Flush', status: 'active', icon: Cpu },
    ];

    return (
        <div className="pv-root">
            {/* Header */}
            <div className="pv-header">
                <div className="pv-header-left">
                    <div className="pv-header-icon">
                        <LockKeyhole size={18} />
                    </div>
                    <div className="pv-header-info">
                        <h2 className="pv-title">Institutional Privacy Vault</h2>
                        <div className="pv-subtitle">
                            <span className="pv-status-dot" />
                            <span>Encrypted & Synchronized</span>
                            <span className="pv-sep" />
                            <span className="pv-key-label">Master Key:</span>
                            <span className="pv-key-value" onClick={() => setShowKey(!showKey)}>
                                {showKey ? 'AES-256-GCM_STP_CORE' : '••••••••••••••••'}
                                {showKey ? <EyeOff size={10} /> : <Eye size={10} />}
                            </span>
                        </div>
                    </div>
                </div>
                <div className="pv-header-right">
                    <div className="pv-stat-block">
                        <span className="pv-stat-label">Federated Status</span>
                        <span className={`pv-stat-value ${fedStatus ? 'pv-val-ok' : ''}`}>
                            {fedStatus ? fedStatus.status : 'CHECKING...'}
                        </span>
                    </div>
                    <div className="pv-stat-block">
                        <span className="pv-stat-label">Connected Nodes</span>
                        <span className="pv-stat-value">{fedStatus?.clients || 0}</span>
                    </div>
                    <button 
                        onClick={() => { fetchDatasets(); fetchJobs(); fetchModels(); }}
                        className="pv-refresh-btn"
                    >
                        <RefreshCcw size={14} className={loading ? 'pv-spin' : ''} />
                    </button>
                </div>
            </div>

            <div className="pv-body">
                {/* Left Column */}
                <div className="pv-col-main">
                    {/* Privacy Shield Grid */}
                    <section className="pv-section">
                        <div className="pv-section-header">
                            <Shield size={14} />
                            <span>Active Privacy Shields</span>
                        </div>
                        <div className="pv-shield-grid">
                            {privacySpecs.map((spec, i) => (
                                <motion.div 
                                    key={i}
                                    className="pv-shield-card"
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: i * 0.05 }}
                                >
                                    <div className="pv-shield-icon">
                                        <spec.icon size={14} />
                                    </div>
                                    <div className="pv-shield-info">
                                        <span className="pv-shield-label">{spec.label}</span>
                                        <span className="pv-shield-value">{spec.value}</span>
                                    </div>
                                    <div className="pv-shield-status">
                                        <CheckCircle2 size={10} />
                                    </div>
                                </motion.div>
                            ))}
                        </div>
                    </section>

                    {/* Encrypted Dataset Pool */}
                    <section className="pv-section">
                        <div className="pv-section-header">
                            <Fingerprint size={14} />
                            <span>Encrypted Dataset Pool</span>
                            <span className="pv-badge">{Array.isArray(datasets) ? datasets.length : 0}</span>
                        </div>
                        
                        {(!Array.isArray(datasets) || datasets.length === 0) ? (
                            <div className="pv-empty-state">
                                <Database size={28} />
                                <span className="pv-empty-title">No Encrypted Datasets Available</span>
                                <span className="pv-empty-sub">
                                    {(error || initError) 
                                        ? 'Secure Training Engine (port 8100) is offline. Start it to browse encrypted datasets.'
                                        : 'Datasets will appear here once the Secure Training API seeds them.'}
                                </span>
                                {(error || initError) && (
                                    <button onClick={() => { setInitError(false); fetchDatasets(); }} className="pv-retry-btn">
                                        <RefreshCcw size={10} /> Retry Connection
                                    </button>
                                )}
                            </div>
                        ) : (
                            <div className="pv-dataset-grid">
                                {datasets.map(ds => (
                                    <motion.div 
                                        key={ds.id}
                                        whileHover={{ y: -2 }}
                                        className={`pv-dataset-card ${selectedDataset?.id === ds.id ? 'pv-dataset-selected' : ''}`}
                                        onClick={() => setSelectedDataset(ds)}
                                    >
                                        <div className="pv-dataset-top">
                                            <div className="pv-dataset-icon"><Database size={14} /></div>
                                            <div className="pv-dataset-badge">
                                                <ShieldCheck size={8} /> Protected
                                            </div>
                                        </div>
                                        <h4 className="pv-dataset-name">{ds.name}</h4>
                                        <p className="pv-dataset-desc">{ds.description}</p>
                                        <div className="pv-dataset-meta">
                                            <div className="pv-meta-item">
                                                <span className="pv-meta-label">Samples</span>
                                                <span className="pv-meta-value">{(ds.num_samples / 1000).toFixed(1)}k</span>
                                            </div>
                                            <div className="pv-meta-item">
                                                <span className="pv-meta-label">Dim</span>
                                                <span className="pv-meta-value">{ds.input_shape?.join('×') || 'N/A'}</span>
                                            </div>
                                            <div className="pv-meta-item pv-meta-right">
                                                <span className="pv-meta-label">Hash</span>
                                                <span className="pv-meta-hash">{ds.id.slice(0, 8)}...</span>
                                            </div>
                                        </div>
                                    </motion.div>
                                ))}
                            </div>
                        )}
                    </section>

                    {/* Active Training Jobs */}
                    <section className="pv-section">
                        <div className="pv-section-header">
                            <Activity size={14} />
                            <span>Training Synapses</span>
                        </div>
                        <div className="pv-jobs-container">
                            {(!Array.isArray(jobs) || jobs.length === 0) ? (
                                <div className="pv-jobs-empty">
                                    <Zap size={20} />
                                    <span>No Active Training Jobs</span>
                                    <span className="pv-jobs-empty-sub">Select a dataset and launch a secure training job</span>
                                </div>
                            ) : (
                                <div className="pv-jobs-list">
                                    {jobs.map(job => (
                                        <div key={job.id} className="pv-job-row">
                                            <div className="pv-job-left">
                                                <div className={`pv-job-indicator ${job.status === 'COMPLETED' ? 'pv-job-done' : 'pv-job-running'}`}>
                                                    {job.status === 'COMPLETED' 
                                                        ? <CheckCircle2 size={12} /> 
                                                        : <RefreshCcw size={12} className="pv-spin" />}
                                                </div>
                                                <div className="pv-job-info">
                                                    <span className="pv-job-name">Job_{job.id.slice(0, 6)}</span>
                                                    <span className="pv-job-model">{job.model_type}</span>
                                                </div>
                                            </div>
                                            <div className="pv-job-right">
                                                <span className="pv-job-progress">{(job.progress * 100).toFixed(1)}%</span>
                                                <span className="pv-job-status">{job.status}</span>
                                            </div>
                                            <div className="pv-job-bar">
                                                <motion.div 
                                                    className="pv-job-bar-fill"
                                                    initial={{ width: 0 }}
                                                    animate={{ width: `${job.progress * 100}%` }}
                                                />
                                            </div>
                                            {job.status === 'COMPLETED' && (
                                                <div className="pv-job-metrics">
                                                    <span>Acc: {job.accuracy}%</span>
                                                    <span>Loss: {job.loss}</span>
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </section>
                </div>

                {/* Right Column */}
                <div className="pv-col-side">
                    {/* Secure Orchestrator Launch */}
                    <div className="pv-launch-card">
                        <div className="pv-launch-glow" />
                        <div className="pv-launch-content">
                            <div className="pv-launch-header">
                                <Zap size={16} />
                                <h3>Secure Orchestrator</h3>
                            </div>
                            <p className="pv-launch-desc">
                                Execute in-memory training on encrypted assets. Data decrypted ONLY in RAM via AES-256-GCM.
                            </p>
                            <div className="pv-launch-target">
                                <span className="pv-launch-target-label">Target Dataset</span>
                                <span className="pv-launch-target-value">
                                    {selectedDataset ? selectedDataset.name : 'NONE_SELECTED'}
                                </span>
                            </div>
                            <button 
                                onClick={handleLaunch}
                                disabled={!selectedDataset || isLaunching}
                                className="pv-launch-btn"
                            >
                                {isLaunching ? <RefreshCcw size={12} className="pv-spin" /> : <Zap size={12} />}
                                Launch Secure Job
                            </button>
                        </div>
                    </div>

                    {/* Security Audit Ledger */}
                    <div className="pv-audit-card">
                        <div className="pv-audit-header">
                            <ShieldCheck size={12} />
                            <span>Security Audit Ledger</span>
                        </div>
                        <div className="pv-audit-rows">
                            <div className="pv-audit-row">
                                <span>RAM Decryption Cipher</span>
                                <span className="pv-audit-val">256-bit</span>
                            </div>
                            <div className="pv-audit-row">
                                <span>DP Noise Injection</span>
                                <span className="pv-audit-val pv-val-ok">
                                    <span className="pv-dot-ok" /> Active
                                </span>
                            </div>
                            <div className="pv-audit-row">
                                <span>Memory Wipe Signal</span>
                                <span className="pv-audit-val pv-val-ok">
                                    <span className="pv-dot-ok" /> Verified
                                </span>
                            </div>
                            <div className="pv-audit-row">
                                <span>Gradient Clipping</span>
                                <span className="pv-audit-val pv-val-ok">
                                    <span className="pv-dot-ok" /> L2=1.0
                                </span>
                            </div>
                            <div className="pv-audit-row">
                                <span>Model Aggregation</span>
                                <span className="pv-audit-val">Median</span>
                            </div>
                        </div>
                        <div className="pv-audit-warning">
                            <AlertCircle size={10} />
                            <span>Decrypted data never touches persistent storage. Ensure worker stable entropy.</span>
                        </div>
                    </div>

                    {/* Trained Weights */}
                    <div className="pv-weights-section">
                        <div className="pv-section-header">
                            <Key size={14} />
                            <span>Trained Weights Hub</span>
                        </div>
                        {(!Array.isArray(models) || models.length === 0) ? (
                            <div className="pv-weights-empty">Registry Empty</div>
                        ) : (
                            <div className="pv-weights-list">
                                {models.map(model => (
                                    <div key={model.id} className="pv-weight-item">
                                        <div className="pv-weight-info">
                                            <span className="pv-weight-name">Weights_v{model.id.slice(0, 4)}</span>
                                            <span className="pv-weight-meta">Acc: {model.accuracy}% | {model.model_type}</span>
                                        </div>
                                        <a 
                                            href={`http://localhost:8100/api/v1/models/download/${model.id}`}
                                            className="pv-weight-dl"
                                            title="Download Weights (.pt)"
                                        >
                                            <LockKeyhole size={12} />
                                        </a>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            <style>{`
                .pv-root {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    height: 100%;
                    overflow: hidden;
                    background: var(--bg-main);
                    font-family: var(--font-sans);
                }

                /* ─── Header ─── */
                .pv-header {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 16px 28px;
                    border-bottom: 1px solid var(--border);
                    background: var(--bg-surface);
                    flex-shrink: 0;
                }
                .pv-header-left {
                    display: flex;
                    align-items: center;
                    gap: 14px;
                }
                .pv-header-icon {
                    width: 36px;
                    height: 36px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: rgba(54,78,104,0.06);
                    border: 1px solid var(--border);
                    border-radius: 8px;
                    color: var(--primary);
                }
                .pv-header-info { display: flex; flex-direction: column; gap: 4px; }
                .pv-title {
                    font-size: 15px;
                    font-weight: 700;
                    color: var(--text-main);
                    letter-spacing: -0.01em;
                    margin: 0;
                }
                .pv-subtitle {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-size: 9px;
                    font-weight: 600;
                    color: var(--text-muted);
                    text-transform: uppercase;
                    letter-spacing: 0.08em;
                }
                .pv-status-dot {
                    width: 5px; height: 5px;
                    background: var(--success);
                    border-radius: 50%;
                    animation: pv-pulse 2s ease-in-out infinite;
                }
                @keyframes pv-pulse {
                    0%, 100% { box-shadow: 0 0 4px rgba(16,185,129,0.3); }
                    50% { box-shadow: 0 0 10px rgba(16,185,129,0.6); }
                }
                .pv-sep { width: 1px; height: 8px; background: var(--border); }
                .pv-key-label { color: var(--text-muted); opacity: 0.5; }
                .pv-key-value {
                    display: inline-flex;
                    align-items: center;
                    gap: 4px;
                    color: var(--text-main);
                    font-family: var(--font-mono);
                    cursor: pointer;
                    padding: 1px 6px;
                    border-radius: 3px;
                    transition: background 0.15s;
                }
                .pv-key-value:hover { background: rgba(54,78,104,0.06); }
                .pv-header-right {
                    display: flex;
                    align-items: center;
                    gap: 20px;
                }
                .pv-stat-block {
                    display: flex;
                    flex-direction: column;
                    gap: 2px;
                    text-align: right;
                }
                .pv-stat-label {
                    font-size: 8px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.15em;
                    color: var(--text-muted);
                    opacity: 0.5;
                }
                .pv-stat-value {
                    font-size: 12px;
                    font-weight: 700;
                    color: var(--text-main);
                    font-family: var(--font-mono);
                }
                .pv-val-ok { color: var(--success) !important; }
                .pv-refresh-btn {
                    width: 36px; height: 36px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border: 1px solid var(--border);
                    border-radius: 8px;
                    background: var(--bg-surface);
                    color: var(--text-muted);
                    cursor: pointer;
                    transition: all 0.2s;
                }
                .pv-refresh-btn:hover {
                    color: var(--primary);
                    border-color: var(--primary);
                }
                .pv-spin { animation: pv-spinning 1s linear infinite; }
                @keyframes pv-spinning { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

                /* ─── Body ─── */
                .pv-body {
                    flex: 1;
                    display: flex;
                    gap: 24px;
                    padding: 24px 28px;
                    overflow-y: auto;
                }
                .pv-col-main { flex: 2; display: flex; flex-direction: column; gap: 24px; min-width: 0; }
                .pv-col-side { flex: 1; display: flex; flex-direction: column; gap: 20px; min-width: 280px; max-width: 380px; }

                /* ─── Section ─── */
                .pv-section { display: flex; flex-direction: column; gap: 14px; }
                .pv-section-header {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-size: 10px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.15em;
                    color: var(--text-main);
                }
                .pv-section-header svg { color: var(--primary); }
                .pv-badge {
                    margin-left: auto;
                    font-size: 9px;
                    font-weight: 800;
                    color: var(--primary);
                    background: rgba(54,78,104,0.06);
                    padding: 2px 10px;
                    border-radius: 10px;
                }

                /* ─── Shield Grid ─── */
                .pv-shield-grid {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 10px;
                }
                .pv-shield-card {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    padding: 12px 14px;
                    background: var(--bg-surface);
                    border: 1px solid var(--border);
                    border-radius: 8px;
                    transition: all 0.2s;
                }
                .pv-shield-card:hover {
                    border-color: var(--primary);
                    box-shadow: 0 2px 12px rgba(54,78,104,0.06);
                }
                .pv-shield-icon {
                    width: 30px; height: 30px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: rgba(54,78,104,0.05);
                    border-radius: 6px;
                    color: var(--primary);
                    flex-shrink: 0;
                }
                .pv-shield-info { display: flex; flex-direction: column; gap: 1px; min-width: 0; }
                .pv-shield-label {
                    font-size: 8px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.1em;
                    color: var(--text-muted);
                }
                .pv-shield-value {
                    font-size: 11px;
                    font-weight: 700;
                    color: var(--text-main);
                    font-family: var(--font-mono);
                }
                .pv-shield-status {
                    margin-left: auto;
                    color: var(--success);
                    flex-shrink: 0;
                }

                /* ─── Empty State ─── */
                .pv-empty-state {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: 40px;
                    border: 1px dashed var(--border);
                    border-radius: 8px;
                    gap: 10px;
                    text-align: center;
                }
                .pv-empty-state svg { color: var(--text-muted); opacity: 0.2; }
                .pv-empty-title {
                    font-size: 11px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.1em;
                    color: var(--text-muted);
                }
                .pv-empty-sub {
                    font-size: 10px;
                    color: var(--text-muted);
                    opacity: 0.6;
                    max-width: 320px;
                    line-height: 1.6;
                }
                .pv-retry-btn {
                    display: inline-flex;
                    align-items: center;
                    gap: 6px;
                    margin-top: 8px;
                    padding: 6px 16px;
                    font-size: 9px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.1em;
                    color: var(--primary);
                    border: 1px solid var(--primary);
                    border-radius: 4px;
                    background: transparent;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                .pv-retry-btn:hover { background: var(--primary); color: white; }

                /* ─── Dataset Grid ─── */
                .pv-dataset-grid {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 12px;
                }
                .pv-dataset-card {
                    padding: 16px;
                    background: var(--bg-surface);
                    border: 1px solid var(--border);
                    border-radius: 8px;
                    cursor: pointer;
                    transition: all 0.25s;
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                }
                .pv-dataset-card:hover {
                    border-color: var(--primary);
                    box-shadow: 0 4px 16px rgba(54,78,104,0.08);
                    transform: translateY(-2px);
                }
                .pv-dataset-selected {
                    border-color: var(--primary) !important;
                    box-shadow: 0 0 0 2px var(--primary), 0 4px 16px rgba(54,78,104,0.1);
                }
                .pv-dataset-top { display: flex; justify-content: space-between; align-items: center; }
                .pv-dataset-icon {
                    width: 28px; height: 28px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: rgba(54,78,104,0.05);
                    border: 1px solid rgba(54,78,104,0.1);
                    border-radius: 6px;
                    color: var(--primary);
                }
                .pv-dataset-badge {
                    display: inline-flex;
                    align-items: center;
                    gap: 4px;
                    padding: 2px 8px;
                    font-size: 7px;
                    font-weight: 800;
                    text-transform: uppercase;
                    letter-spacing: 0.1em;
                    color: var(--success);
                    background: rgba(16,185,129,0.06);
                    border: 1px solid rgba(16,185,129,0.15);
                    border-radius: 20px;
                }
                .pv-dataset-name {
                    font-size: 11px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.08em;
                    color: var(--text-main);
                    margin: 0;
                }
                .pv-dataset-desc {
                    font-size: 9px;
                    color: var(--text-muted);
                    opacity: 0.7;
                    font-style: italic;
                    line-height: 1.5;
                    margin: 0;
                }
                .pv-dataset-meta {
                    display: flex;
                    align-items: center;
                    gap: 16px;
                    padding-top: 8px;
                    border-top: 1px solid rgba(226,230,236,0.5);
                }
                .pv-meta-item { display: flex; flex-direction: column; gap: 1px; }
                .pv-meta-right { margin-left: auto; text-align: right; }
                .pv-meta-label {
                    font-size: 7px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.1em;
                    color: var(--text-muted);
                    opacity: 0.5;
                }
                .pv-meta-value {
                    font-size: 10px;
                    font-weight: 700;
                    color: var(--text-main);
                    font-family: var(--font-mono);
                }
                .pv-meta-hash {
                    font-size: 9px;
                    font-family: var(--font-mono);
                    color: var(--text-muted);
                    opacity: 0.4;
                }

                /* ─── Jobs ─── */
                .pv-jobs-container {
                    background: var(--bg-surface);
                    border: 1px solid var(--border);
                    border-radius: 8px;
                    overflow: hidden;
                }
                .pv-jobs-empty {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: 32px;
                    gap: 6px;
                    color: var(--text-muted);
                    opacity: 0.4;
                    font-size: 10px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.15em;
                }
                .pv-jobs-empty-sub {
                    font-size: 9px;
                    font-weight: 400;
                    text-transform: none;
                    letter-spacing: 0;
                    opacity: 0.7;
                }
                .pv-jobs-list { display: flex; flex-direction: column; }
                .pv-job-row {
                    display: flex;
                    flex-wrap: wrap;
                    align-items: center;
                    padding: 12px 16px;
                    gap: 12px;
                    border-bottom: 1px solid rgba(226,230,236,0.5);
                }
                .pv-job-row:last-child { border-bottom: none; }
                .pv-job-left { display: flex; align-items: center; gap: 10px; }
                .pv-job-indicator {
                    width: 28px; height: 28px;
                    border-radius: 50%;
                    border: 1px solid var(--border);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .pv-job-done { color: var(--success); border-color: rgba(16,185,129,0.3); }
                .pv-job-running { color: var(--primary); }
                .pv-job-info { display: flex; flex-direction: column; gap: 1px; }
                .pv-job-name { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-main); }
                .pv-job-model { font-size: 8px; color: var(--text-muted); font-family: var(--font-mono); }
                .pv-job-right { margin-left: auto; text-align: right; }
                .pv-job-progress { font-size: 11px; font-weight: 700; color: var(--primary); font-family: var(--font-mono); }
                .pv-job-status { font-size: 8px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); }
                .pv-job-bar { width: 100%; height: 3px; background: rgba(226,230,236,0.5); border-radius: 2px; overflow: hidden; }
                .pv-job-bar-fill { height: 100%; background: linear-gradient(90deg, var(--primary), var(--success)); border-radius: 2px; }
                .pv-job-metrics {
                    width: 100%;
                    display: flex;
                    gap: 16px;
                    padding: 4px 8px;
                    background: rgba(16,185,129,0.04);
                    border: 1px solid rgba(16,185,129,0.1);
                    border-radius: 4px;
                    font-size: 9px;
                    font-weight: 700;
                    color: var(--success);
                    text-transform: uppercase;
                    letter-spacing: 0.08em;
                }

                /* ─── Launch Card ─── */
                .pv-launch-card {
                    background: var(--primary);
                    border-radius: 10px;
                    overflow: hidden;
                    position: relative;
                    box-shadow: 0 8px 24px rgba(54,78,104,0.15);
                }
                .pv-launch-glow {
                    position: absolute;
                    right: -30px;
                    top: -30px;
                    width: 120px;
                    height: 120px;
                    background: rgba(255,255,255,0.04);
                    border-radius: 50%;
                    filter: blur(30px);
                }
                .pv-launch-content {
                    position: relative;
                    padding: 24px;
                    display: flex;
                    flex-direction: column;
                    gap: 14px;
                    color: white;
                }
                .pv-launch-header {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                .pv-launch-header svg { opacity: 0.5; }
                .pv-launch-header h3 {
                    font-size: 14px;
                    font-weight: 700;
                    margin: 0;
                }
                .pv-launch-desc {
                    font-size: 10px;
                    opacity: 0.6;
                    line-height: 1.6;
                    margin: 0;
                    font-style: italic;
                }
                .pv-launch-target {
                    display: flex;
                    justify-content: space-between;
                    font-size: 9px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.1em;
                }
                .pv-launch-target-label { opacity: 0.4; }
                .pv-launch-target-value { opacity: 0.9; font-family: var(--font-mono); }
                .pv-launch-btn {
                    width: 100%;
                    padding: 12px;
                    background: white;
                    color: var(--primary);
                    border: none;
                    border-radius: 6px;
                    font-size: 10px;
                    font-weight: 800;
                    text-transform: uppercase;
                    letter-spacing: 0.2em;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                    transition: all 0.2s;
                }
                .pv-launch-btn:hover:not(:disabled) {
                    transform: translateY(-1px);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                }
                .pv-launch-btn:disabled {
                    opacity: 0.4;
                    cursor: not-allowed;
                    filter: grayscale(0.5);
                }

                /* ─── Audit Card ─── */
                .pv-audit-card {
                    background: var(--bg-surface);
                    border: 1px solid var(--border);
                    border-radius: 8px;
                    overflow: hidden;
                }
                .pv-audit-header {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 12px 16px;
                    font-size: 10px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.08em;
                    color: var(--text-main);
                    border-bottom: 1px solid var(--border);
                }
                .pv-audit-header svg { color: var(--success); }
                .pv-audit-rows { padding: 8px 0; }
                .pv-audit-row {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 8px 16px;
                    font-size: 9px;
                    font-weight: 600;
                    color: var(--text-muted);
                    text-transform: uppercase;
                    letter-spacing: 0.06em;
                }
                .pv-audit-val {
                    font-weight: 700;
                    color: var(--text-main);
                    font-family: var(--font-mono);
                    display: flex;
                    align-items: center;
                    gap: 4px;
                }
                .pv-dot-ok {
                    width: 4px; height: 4px;
                    background: var(--success);
                    border-radius: 50%;
                }
                .pv-audit-warning {
                    display: flex;
                    align-items: flex-start;
                    gap: 8px;
                    margin: 8px 12px 12px;
                    padding: 10px 12px;
                    background: rgba(239,68,68,0.04);
                    border: 1px solid rgba(239,68,68,0.1);
                    border-radius: 4px;
                    font-size: 8px;
                    font-weight: 600;
                    color: var(--error);
                    line-height: 1.5;
                }
                .pv-audit-warning svg { flex-shrink: 0; margin-top: 1px; }

                /* ─── Weights ─── */
                .pv-weights-section { display: flex; flex-direction: column; gap: 10px; }
                .pv-weights-empty {
                    padding: 24px;
                    border: 1px dashed var(--border);
                    border-radius: 6px;
                    font-size: 9px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.15em;
                    color: var(--text-muted);
                    opacity: 0.4;
                    text-align: center;
                }
                .pv-weights-list { display: flex; flex-direction: column; gap: 6px; }
                .pv-weight-item {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 10px 14px;
                    background: var(--bg-surface);
                    border: 1px solid var(--border);
                    border-radius: 6px;
                    transition: all 0.2s;
                }
                .pv-weight-item:hover { border-color: var(--primary); }
                .pv-weight-info { display: flex; flex-direction: column; gap: 2px; }
                .pv-weight-name {
                    font-size: 10px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.08em;
                    color: var(--text-main);
                }
                .pv-weight-meta {
                    font-size: 8px;
                    font-weight: 600;
                    color: var(--text-muted);
                    opacity: 0.6;
                    text-transform: uppercase;
                }
                .pv-weight-dl {
                    width: 30px; height: 30px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border: 1px solid var(--border);
                    border-radius: 4px;
                    color: var(--text-muted);
                    text-decoration: none;
                    transition: all 0.2s;
                }
                .pv-weight-dl:hover {
                    background: rgba(54,78,104,0.05);
                    color: var(--primary);
                    border-color: var(--primary);
                }

                /* ─── Responsive ─── */
                @media (max-width: 1200px) {
                    .pv-body { flex-direction: column; }
                    .pv-col-side { max-width: none; }
                    .pv-shield-grid { grid-template-columns: repeat(2, 1fr); }
                }
            `}</style>
        </div>
    );
};
