import React, { useState, useEffect } from 'react';
import {
  Database, ShieldCheck, Share2, Box, Lock,
  Activity, HardDrive, Search, Filter,
  FileText, Download, Server, Globe, Shield,
  RefreshCw, Cpu, Eye, Layers, Zap
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { API_BASE_URL } from '../hooks/useSecureFederated';

export const DatasetExplorer = ({ shards = [], clientsActive = 0, roundHistory = [], nodeRegistry = [] }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState('shards');
  const [vaultDatasets, setVaultDatasets] = useState([]);
  const [isVaultLoading, setIsVaultLoading] = useState(false);
  const [expandedCard, setExpandedCard] = useState(null);

  useEffect(() => {
    fetchVaultDatasets();
  }, []);

  const fetchVaultDatasets = async () => {
    setIsVaultLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/laboratory/vault-datasets`);
      const data = await res.json();
      if (data.success) setVaultDatasets(data.datasets || []);
    } catch (err) {
      console.error('Vault fetch error:', err);
    } finally {
      setIsVaultLoading(false);
    }
  };

  const realShards = shards.filter(s => s && typeof s === 'object');
  const totalSamples = realShards.reduce((sum, s) => sum + (s.size || 0), 0);
  const totalEncSize = vaultDatasets.reduce((sum, d) => sum + (d.encrypted_size || 0), 0);
  const latestRound = roundHistory?.length || 0;

  const filteredShards = realShards.filter(s => {
    const name = (s.name || s.org || s.id || '').toLowerCase();
    return name.includes(searchQuery.toLowerCase());
  });

  const filteredVault = vaultDatasets.filter(d => {
    return d.name.toLowerCase().includes(searchQuery.toLowerCase());
  });

  const regionColors = {
    'US-East': '#6366f1', 'EU-West': '#06b6d4', 'Global': '#8b5cf6',
    'US-West': '#f59e0b', 'AP-South': '#ef4444'
  };

  const statusColors = {
    'ACTIVE': '#22c55e', 'FULL': '#f59e0b', 'IDLE': '#6b7280', 'OFFLINE': '#ef4444'
  };

  return (
    <div className="ds-root section-fade">
      <header className="ds-header">
        <div className="ds-header-left">
          <div className="ds-module-badge">
            <span className="ds-badge-num">03</span>
            <span className="ds-badge-sep" />
            <span className="ds-badge-text">Data Registry</span>
          </div>
          <h2 className="ds-title">Distributed Data Inventory</h2>
          <p className="ds-subtitle">
            Live registry of federated training shards and encrypted vault datasets.
            All data is AES-256-GCM encrypted and verified against the node trust ledger.
          </p>
        </div>
        
        <div className="ds-header-stats">
          <div className="ds-stat-card">
            <Server size={14} className="ds-stat-icon" />
            <div className="ds-stat-info">
              <span className="ds-stat-label">Active Nodes</span>
              <span className="ds-stat-value">{clientsActive}</span>
            </div>
          </div>
          <div className="ds-stat-card">
            <Shield size={14} className="ds-stat-icon" />
            <div className="ds-stat-info">
              <span className="ds-stat-label">Vault Datasets</span>
              <span className="ds-stat-value">{vaultDatasets.length}</span>
            </div>
          </div>
          <div className="ds-stat-card">
            <Layers size={14} className="ds-stat-icon" />
            <div className="ds-stat-info">
              <span className="ds-stat-label">FL Shards</span>
              <span className="ds-stat-value">{realShards.length}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Tab Switcher */}
      <div className="ds-tab-bar">
        <button
          className={`ds-tab ${activeTab === 'shards' ? 'ds-tab-active' : ''}`}
          onClick={() => setActiveTab('shards')}
        >
          <HardDrive size={12} />
          <span>FL Training Shards</span>
          <span className="ds-tab-count">{realShards.length}</span>
        </button>
        <button
          className={`ds-tab ${activeTab === 'vault' ? 'ds-tab-active' : ''}`}
          onClick={() => setActiveTab('vault')}
        >
          <Lock size={12} />
          <span>Encrypted Vault</span>
          <span className="ds-tab-count">{vaultDatasets.length}</span>
        </button>

        <div className="ds-tab-spacer" />

        <div className="ds-search-group">
          <Search size={13} className={`ds-search-icon ${searchQuery ? 'ds-active' : ''}`} />
          <input
            type="text"
            placeholder="Search..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="ds-search-input"
          />
        </div>

        <button onClick={fetchVaultDatasets} className="ds-refresh-btn" title="Refresh Vault">
          <RefreshCw size={12} className={isVaultLoading ? 'ds-spin' : ''} />
        </button>
      </div>

      {/* FL Shards View */}
      <AnimatePresence mode="wait">
        {activeTab === 'shards' && (
          <motion.div
            key="shards"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="ds-grid"
          >
            {filteredShards.length > 0 ? filteredShards.map((shard, idx) => {
              const trustPct = Math.min(Math.round((shard.trust_score || 0) / 1.5), 100);
              const statusColor = statusColors[shard.status] || '#6b7280';
              const lastSeen = shard.last_seen ? new Date(shard.last_seen).toLocaleString() : '—';

              return (
                <motion.div
                  key={shard.id + idx}
                  layout
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.04 }}
                  className="ds-card"
                  onClick={() => setExpandedCard(expandedCard === shard.id ? null : shard.id)}
                >
                  <div className="ds-card-head">
                    <div className="ds-card-tag-row">
                      <span className="ds-card-tag">{shard.name || 'FL_Node'}</span>
                      <span className="ds-card-status" style={{ color: statusColor, borderColor: statusColor + '33' }}>
                        <span className="ds-status-dot" style={{ background: statusColor }} />
                        {shard.status || 'UNKNOWN'}
                      </span>
                    </div>
                    <h3 className="ds-card-name" style={{ fontSize: '13px', fontFamily: 'var(--font-mono)' }}>{shard.ip_address || '—'}</h3>
                    <div className="ds-card-meta">
                      <span className="ds-meta-item">
                        <Lock size={9} />
                        {shard.encryption || 'AES-256-GCM'}
                      </span>
                      <span className="ds-meta-item">
                        <Activity size={9} />
                        Trust: {shard.trust_score || 0}
                      </span>
                    </div>
                  </div>

                  <div className="ds-card-body">
                    <div className="ds-load-section">
                      <div className="ds-load-header">
                        <span className="ds-load-label">Trust Score</span>
                        <span className="ds-load-value">{shard.trust_score || 0}</span>
                      </div>
                      <div className="ds-load-track">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${trustPct}%` }}
                          transition={{ duration: 0.8, ease: 'easeOut' }}
                          className="ds-load-fill"
                          style={{ background: trustPct >= 80 ? '#22c55e' : trustPct >= 50 ? '#f59e0b' : '#ef4444' }}
                        />
                      </div>
                    </div>
                  </div>

                  <div className="ds-card-foot">
                    <span className="ds-foot-hint" style={{ fontSize: '8px', opacity: 0.5 }}>{lastSeen}</span>
                    <span className="ds-foot-badge" style={{ color: 'var(--text-muted)' }}>
                      <Cpu size={9} />
                      <span>{shard.id}</span>
                    </span>
                  </div>

                  <AnimatePresence>
                    {expandedCard === shard.id && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="ds-card-expanded"
                      >
                        <div className="ds-exp-row">
                          <span className="ds-exp-label">Node ID</span>
                          <span className="ds-exp-value ds-mono">{shard.id}</span>
                        </div>
                        <div className="ds-exp-row">
                          <span className="ds-exp-label">IP Address</span>
                          <span className="ds-exp-value ds-mono">{shard.ip_address}</span>
                        </div>
                        <div className="ds-exp-row">
                          <span className="ds-exp-label">Trust Score</span>
                          <span className="ds-exp-value">{shard.trust_score}</span>
                        </div>
                        <div className="ds-exp-row">
                          <span className="ds-exp-label">Last Active</span>
                          <span className="ds-exp-value">{lastSeen}</span>
                        </div>
                        <div className="ds-exp-row">
                          <span className="ds-exp-label">Encryption</span>
                          <span className="ds-exp-value">{shard.encryption}</span>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              );
            }) : (
              <div className="ds-empty-state">
                <HardDrive size={28} style={{ opacity: 0.15 }} />
                <span>No FL shards registered yet. Start training to generate shards.</span>
              </div>
            )}
          </motion.div>
        )}

        {/* Vault Datasets View */}
        {activeTab === 'vault' && (
          <motion.div
            key="vault"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="ds-grid ds-vault-grid"
          >
            {filteredVault.length > 0 ? filteredVault.map((ds, idx) => {
              const isRegression = (ds.classes || 0) === 0;
              return (
                <motion.div
                  key={ds.name}
                  layout
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.06 }}
                  className="ds-card ds-vault-card"
                  onClick={() => setExpandedCard(expandedCard === ds.name ? null : ds.name)}
                >
                  <div className="ds-card-head">
                    <div className="ds-card-tag-row">
                      <span className="ds-card-tag ds-vault-tag">
                        <Lock size={8} />
                        ENCRYPTED
                      </span>
                      <span className="ds-card-status ds-vault-status">
                        <ShieldCheck size={9} />
                        AES-256-GCM
                      </span>
                    </div>
                    <h3 className="ds-card-name">{ds.name}</h3>
                    <p className="ds-card-desc">{ds.description}</p>
                  </div>

                  <div className="ds-card-body">
                    <div className="ds-vault-stats">
                      <div className="ds-vault-stat">
                        <span className="ds-vault-stat-num">{ds.samples?.toLocaleString()}</span>
                        <span className="ds-vault-stat-label">Samples</span>
                      </div>
                      <div className="ds-vault-stat-divider" />
                      <div className="ds-vault-stat">
                        <span className="ds-vault-stat-num">{isRegression ? 'REG' : ds.classes}</span>
                        <span className="ds-vault-stat-label">{isRegression ? 'Task' : 'Classes'}</span>
                      </div>
                      <div className="ds-vault-stat-divider" />
                      <div className="ds-vault-stat">
                        <span className="ds-vault-stat-num">{JSON.stringify(ds.shape)}</span>
                        <span className="ds-vault-stat-label">Shape</span>
                      </div>
                    </div>
                  </div>

                  <div className="ds-card-foot">
                    <div className="ds-foot-badge">
                      <Shield size={9} />
                      <span>Privacy Vault</span>
                    </div>
                    <span className="ds-foot-hint">vault.load("{ds.name}")</span>
                  </div>

                  <AnimatePresence>
                    {expandedCard === ds.name && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="ds-card-expanded"
                      >
                        <div className="ds-exp-row">
                          <span className="ds-exp-label">Access</span>
                          <span className="ds-exp-value">RAM-only decryption</span>
                        </div>
                        <div className="ds-exp-row">
                          <span className="ds-exp-label">Wipe Policy</span>
                          <span className="ds-exp-value">Auto-wipe after training</span>
                        </div>
                        <div className="ds-exp-row">
                          <span className="ds-exp-label">Lab Usage</span>
                          <code className="ds-exp-code">data, labels, info = vault.load("{ds.name}")</code>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              );
            }) : (
              <div className="ds-empty-state">
                <Lock size={28} style={{ opacity: 0.15 }} />
                <span>{isVaultLoading ? 'Loading encrypted datasets...' : 'No datasets in vault. Start the Secure Training Platform.'}</span>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      <style>{`
        .ds-root {
          padding: 40px;
          display: flex; flex-direction: column; gap: 32px;
          background: var(--bg-main);
          min-height: 100%;
          font-family: var(--font-sans);
          padding-bottom: 120px;
        }

        .ds-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 24px; flex-wrap: wrap; }
        .ds-header-left { flex: 1; min-width: 300px; }
        .ds-module-badge { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
        .ds-badge-num { font-size: 10px; font-weight: 800; color: var(--primary); letter-spacing: 0.1em; }
        .ds-badge-sep { width: 24px; height: 1px; background: var(--primary); opacity: 0.2; }
        .ds-badge-text { font-size: 10px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.2em; }
        .ds-title { font-family: var(--font-serif); font-size: 32px; font-weight: 500; color: var(--text-main); margin: 0 0 12px 0; letter-spacing: -0.01em; }
        .ds-subtitle { font-size: 13px; color: var(--text-muted); line-height: 1.6; max-width: 600px; margin: 0; }

        .ds-header-stats { display: flex; gap: 12px; }
        .ds-stat-card {
          display: flex; align-items: center; gap: 12px;
          padding: 14px 20px;
          background: var(--bg-surface);
          border: 1px solid var(--border);
          border-radius: 2px;
        }
        .ds-stat-icon { color: var(--primary); opacity: 0.5; }
        .ds-stat-info { display: flex; flex-direction: column; gap: 2px; }
        .ds-stat-label { font-size: 8px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.15em; }
        .ds-stat-value { font-size: 18px; font-weight: 800; color: var(--text-main); font-family: var(--font-mono); }

        /* Tabs */
        .ds-tab-bar {
          display: flex; align-items: center; gap: 4px;
          padding-bottom: 16px;
          border-bottom: 1px solid var(--border);
        }
        .ds-tab {
          display: flex; align-items: center; gap: 6px;
          padding: 8px 16px;
          background: none; border: 1px solid transparent;
          font-size: 10px; font-weight: 700; text-transform: uppercase;
          letter-spacing: 0.08em; color: var(--text-muted);
          cursor: pointer; transition: all 0.2s;
          border-radius: 0;
        }
        .ds-tab:hover { color: var(--text-main); background: rgba(255,255,255,0.02); }
        .ds-tab-active {
          color: var(--primary); border-color: var(--primary);
          background: rgba(99,102,241,0.04);
        }
        .ds-tab-count {
          font-size: 9px; font-weight: 800;
          padding: 1px 6px; border-radius: 10px;
          background: rgba(99,102,241,0.08); color: var(--primary);
        }
        .ds-tab-spacer { flex: 1; }
        .ds-search-group { display: flex; align-items: center; gap: 8px; }
        .ds-search-icon { color: var(--text-muted); opacity: 0.3; transition: all 0.2s; }
        .ds-search-icon.ds-active { opacity: 1; color: var(--primary); }
        .ds-search-input {
          background: none; border: none; outline: none;
          font-size: 11px; font-weight: 600; color: var(--text-main);
          width: 140px; padding: 4px 0;
        }
        .ds-search-input::placeholder { color: var(--text-muted); opacity: 0.3; }
        .ds-refresh-btn {
          background: none; border: 1px solid var(--border);
          padding: 6px; cursor: pointer; color: var(--text-muted);
          transition: all 0.2s;
        }
        .ds-refresh-btn:hover { border-color: var(--primary); color: var(--primary); }
        @keyframes dsSpin { to { transform: rotate(360deg); } }
        .ds-spin { animation: dsSpin 1s linear infinite; }

        /* Grid */
        .ds-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .ds-vault-grid { grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); }

        /* Cards (shared) */
        .ds-card {
          background: var(--bg-surface);
          border: 1px solid var(--border);
          display: flex; flex-direction: column;
          transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
          cursor: pointer; overflow: hidden;
        }
        .ds-card:hover { border-color: var(--primary); transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.06); }

        .ds-card-head { padding: 20px 24px 16px; }
        .ds-card-tag-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .ds-card-tag {
          font-size: 8px; font-weight: 800; color: var(--primary);
          text-transform: uppercase; letter-spacing: 0.2em;
          display: flex; align-items: center; gap: 4px;
        }
        .ds-vault-tag { color: #8b5cf6; }
        .ds-card-status {
          font-size: 8px; font-weight: 700; text-transform: uppercase;
          letter-spacing: 0.08em;
          display: flex; align-items: center; gap: 4px;
          padding: 2px 8px; border: 1px solid;
        }
        .ds-vault-status { color: #22c55e; border-color: rgba(34,197,94,0.2); }
        .ds-status-dot { width: 5px; height: 5px; border-radius: 50%; }
        .ds-card-name {
          font-family: var(--font-serif); font-size: 17px; font-weight: 500;
          color: var(--text-main); margin: 0;
        }
        .ds-card-desc {
          font-size: 11px; color: var(--text-muted); line-height: 1.5;
          margin: 6px 0 0; display: -webkit-box;
          -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
        }
        .ds-card-meta { display: flex; gap: 14px; margin-top: 8px; }
        .ds-meta-item {
          font-size: 9px; font-weight: 600; color: var(--text-muted);
          display: flex; align-items: center; gap: 4px;
          text-transform: uppercase; letter-spacing: 0.06em;
        }

        .ds-card-body { padding: 0 24px 20px; }

        /* Load bar */
        .ds-load-section { display: flex; flex-direction: column; gap: 8px; }
        .ds-load-header { display: flex; justify-content: space-between; }
        .ds-load-label { font-size: 9px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; }
        .ds-load-value { font-size: 11px; font-weight: 800; color: var(--text-main); font-family: var(--font-mono); }
        .ds-load-track { height: 4px; background: var(--bg-main); overflow: hidden; }
        .ds-load-fill { height: 100%; border-radius: 2px; }

        /* Vault stats */
        .ds-vault-stats {
          display: flex; align-items: center; justify-content: center;
          background: var(--bg-main);
          border: 1px solid var(--border);
          padding: 12px 0;
        }
        .ds-vault-stat { flex: 1; text-align: center; padding: 4px 12px; }
        .ds-vault-stat-num {
          display: block; font-size: 13px; font-weight: 800;
          color: var(--text-main); font-family: var(--font-mono);
        }
        .ds-vault-stat-label {
          display: block; font-size: 8px; font-weight: 700;
          color: var(--text-muted); text-transform: uppercase;
          letter-spacing: 0.1em; margin-top: 2px;
        }
        .ds-vault-stat-divider { width: 1px; height: 28px; background: var(--border); }

        /* Footer */
        .ds-card-foot {
          padding: 10px 24px;
          border-top: 1px solid var(--border);
          display: flex; justify-content: space-between; align-items: center;
          opacity: 0.5; transition: opacity 0.2s;
        }
        .ds-card:hover .ds-card-foot { opacity: 1; }
        .ds-foot-badge {
          display: flex; align-items: center; gap: 5px;
          font-size: 8px; font-weight: 700; color: #8b5cf6;
          text-transform: uppercase; letter-spacing: 0.08em;
        }
        .ds-foot-hint {
          font-size: 9px; font-family: var(--font-mono);
          color: var(--text-muted); opacity: 0.6;
        }

        /* Expanded */
        .ds-card-expanded {
          border-top: 1px solid var(--border);
          overflow: hidden;
        }
        .ds-exp-row {
          display: flex; justify-content: space-between; align-items: center;
          padding: 8px 24px;
          border-bottom: 1px solid rgba(255,255,255,0.02);
        }
        .ds-exp-row:last-child { border-bottom: none; }
        .ds-exp-label { font-size: 9px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; }
        .ds-exp-value { font-size: 10px; font-weight: 600; color: var(--text-main); }
        .ds-mono { font-family: var(--font-mono); font-size: 9px; }
        .ds-exp-code {
          font-size: 9px; font-family: var(--font-mono);
          background: rgba(99,102,241,0.06); padding: 3px 8px;
          color: var(--primary); border-radius: 2px;
        }

        /* Empty */
        .ds-empty-state {
          grid-column: 1 / -1;
          display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px;
          padding: 80px 40px;
          border: 2px dashed var(--border);
          background: var(--bg-surface);
          text-align: center;
        }
        .ds-empty-state span { font-size: 12px; color: var(--text-muted); max-width: 400px; line-height: 1.6; }
      `}</style>
    </div>
  );
};
