import React, { useRef, useEffect } from 'react';
import { Box, Layers, History, ShieldCheck, Activity, Archive, Landmark, Database } from 'lucide-react';
import { motion } from 'framer-motion';

export const BlockchainRibbon = ({ blockchain }) => {
  const ribbonRef = useRef(null);

  useEffect(() => {
    if (ribbonRef.current) {
      ribbonRef.current.scrollTop = ribbonRef.current.scrollHeight;
    }
  }, [blockchain]);

  const hasBlocks = blockchain && blockchain.length > 0;

  return (
    <div className="bl-root">
      {/* Ribbon Header */}
      <div className="bl-header">
        <div className="bl-header-left">
          <Landmark size={13} className="bl-header-icon" />
          <span className="bl-header-title">Audit Ledger</span>
        </div>
        <div className="bl-header-right">
          <div className="bl-header-label-group">
            <Database size={13} className="bl-db-icon" />
            <h3 className="bl-header-name">Blockchain Ledger</h3>
          </div>
          <div className="bl-sync-badge">
            <div className="bl-sync-dot" />
            <span>Ledger Synchronized</span>
          </div>
        </div>
      </div>

      {/* Ledger Column Headers */}
      <div className="bl-col-headers">
        <div>Block ID</div>
        <div className="bl-col-hash">Checksum (Signature)</div>
        <div className="bl-col-center">Node Load</div>
        <div className="bl-col-right">Integrity Status</div>
      </div>

      {/* Ledger Rows Viewport */}
      <div ref={ribbonRef} className="bl-rows">
        {!hasBlocks ? (
          <div className="bl-empty">
            <Database size={20} className="bl-empty-icon" />
            <span>Awaiting genesis block synchronization...</span>
          </div>
        ) : (
          blockchain.map((block) => (
            <motion.div
              key={`${block.index}-${block.hash}`}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className={`bl-row ${block.index === 0 ? 'bl-row-genesis' : ''}`}
            >
              {/* Block ID */}
              <div className="bl-cell-id">
                <div className={`bl-block-dot ${block.index === 0 ? 'bl-dot-genesis' : ''}`} />
                <span className="bl-block-name">
                  {block.index === 0 ? 'Genesis' : `Block ${block.index.toString().padStart(4, '0')}`}
                </span>
              </div>

              {/* Signature Hash */}
              <div className="bl-cell-hash">
                <span className="bl-hash-text">
                  {block.hash !== 'null' && block.hash.length > 16
                    ? `${block.hash.substring(0, 10)}...${block.hash.substring(block.hash.length - 8)}`
                    : block.hash === 'null' ? '0x0000_GENESIS' : block.hash}
                </span>
              </div>

              {/* Metrics/Load */}
              <div className="bl-cell-load">
                <span className="bl-load-text">{block.transactions.length} LRD</span>
                <div className="bl-load-bars">
                  {[...Array(Math.min(block.transactions.length, 5))].map((_, idx) => (
                    <div key={idx} className="bl-load-bar" />
                  ))}
                </div>
              </div>

              {/* Integrity Status */}
              <div className="bl-cell-status">
                <span className={`bl-status-tag ${block.index === 0 ? 'bl-tag-auth' : 'bl-tag-verified'}`}>
                  {block.index === 0 ? 'Authorized' : 'Verified'}
                </span>
              </div>
            </motion.div>
          ))
        )}
      </div>

      <style>{`
        .bl-root {
          display: flex;
          flex-direction: column;
          height: 100%;
          overflow: hidden;
          background: #0a0e17;
          border-radius: 2px;
          position: relative;
        }
        .bl-root::before {
          content: '';
          position: absolute;
          top: 0; left: 0; right: 0;
          height: 1px;
          background: linear-gradient(90deg, transparent, var(--primary), transparent);
          opacity: 0.3;
          z-index: 5;
        }

        /* ── Header ── */
        .bl-header {
          padding: 12px 24px;
          border-bottom: 1px solid rgba(255,255,255,0.08);
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-shrink: 0;
          background: linear-gradient(180deg, rgba(255,255,255,0.03), transparent);
        }
        .bl-header-left {
          display: flex;
          align-items: center;
          gap: 10px;
        }
        .bl-header-icon {
          color: var(--primary);
          opacity: 0.8;
        }
        .bl-header-title {
          font-size: 10px;
          font-weight: 800;
          color: rgba(255,255,255,0.5);
          text-transform: uppercase;
          letter-spacing: 0.25em;
          font-family: var(--font-mono);
        }
        .bl-header-right {
          display: flex;
          align-items: center;
          gap: 16px;
        }
        .bl-header-label-group {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .bl-db-icon {
          color: var(--primary);
          opacity: 0.6;
        }
        .bl-header-name {
          font-size: 12px;
          font-weight: 600;
          color: rgba(255,255,255,0.85);
          margin: 0;
        }
        .bl-sync-badge {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 4px 12px;
          background: rgba(16,185,129,0.08);
          border: 1px solid rgba(16,185,129,0.15);
        }
        .bl-sync-dot {
          width: 6px; height: 6px;
          border-radius: 50%;
          background: #10b981;
          box-shadow: 0 0 8px rgba(16,185,129,0.5);
          animation: bl-pulse 2s infinite;
        }
        @keyframes bl-pulse {
          0% { opacity: 1; box-shadow: 0 0 8px rgba(16,185,129,0.5); }
          50% { opacity: 0.5; box-shadow: 0 0 4px rgba(16,185,129,0.2); }
          100% { opacity: 1; box-shadow: 0 0 8px rgba(16,185,129,0.5); }
        }
        .bl-sync-badge span {
          font-size: 9px;
          font-weight: 700;
          color: #34d399;
          text-transform: uppercase;
          letter-spacing: 0.08em;
        }

        /* ── Column Headers ── */
        .bl-col-headers {
          display: grid;
          grid-template-columns: 130px 1fr 110px 130px;
          align-items: center;
          padding: 10px 24px;
          border-bottom: 1px solid rgba(255,255,255,0.06);
          background: rgba(0,0,0,0.4);
          flex-shrink: 0;
          font-size: 9px;
          font-weight: 800;
          color: rgba(255,255,255,0.35);
          text-transform: uppercase;
          letter-spacing: 0.2em;
          font-family: var(--font-mono);
          position: sticky;
          top: 0;
          z-index: 10;
        }
        .bl-col-hash { padding-left: 16px; }
        .bl-col-center { text-align: center; }
        .bl-col-right { text-align: right; }

        /* ── Rows ── */
        .bl-rows {
          flex: 1;
          overflow-y: auto;
          background: #060a12;
        }
        .bl-rows::-webkit-scrollbar { width: 5px; }
        .bl-rows::-webkit-scrollbar-track { background: #060a12; }
        .bl-rows::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
        .bl-rows::-webkit-scrollbar-thumb:hover { background: #334155; }

        .bl-row {
          display: grid;
          grid-template-columns: 130px 1fr 110px 130px;
          align-items: center;
          padding: 12px 24px;
          border-bottom: 1px solid rgba(255,255,255,0.04);
          transition: background 0.2s;
        }
        .bl-row:hover {
          background: rgba(255,255,255,0.03);
        }
        .bl-row-genesis {
          background: rgba(var(--primary-rgb), 0.04);
          border-left: 2px solid var(--primary);
        }

        /* ── Cells ── */
        .bl-cell-id {
          display: flex;
          align-items: center;
          gap: 10px;
        }
        .bl-block-dot {
          width: 6px; height: 6px;
          border-radius: 50%;
          background: rgba(255,255,255,0.25);
          flex-shrink: 0;
        }
        .bl-dot-genesis {
          background: var(--primary);
          box-shadow: 0 0 10px rgba(var(--primary-rgb), 0.5);
        }
        .bl-block-name {
          font-size: 11px;
          font-weight: 700;
          color: rgba(255,255,255,0.9);
          text-transform: uppercase;
          letter-spacing: 0.12em;
          font-family: var(--font-serif);
        }

        .bl-cell-hash {
          padding-left: 16px;
          overflow: hidden;
          min-width: 0;
        }
        .bl-hash-text {
          font-family: var(--font-mono);
          font-size: 10px;
          color: rgba(255,255,255,0.45);
          letter-spacing: 0.02em;
          transition: color 0.2s;
        }
        .bl-row:hover .bl-hash-text {
          color: rgba(255,255,255,0.7);
        }

        .bl-cell-load {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
        }
        .bl-load-text {
          font-size: 10px;
          font-weight: 700;
          color: rgba(255,255,255,0.4);
          text-transform: uppercase;
          font-family: var(--font-mono);
          white-space: nowrap;
        }
        .bl-load-bars {
          display: flex;
          gap: 2px;
          align-items: flex-end;
        }
        .bl-load-bar {
          width: 3px;
          height: 10px;
          background: var(--primary);
          opacity: 0.35;
          transition: opacity 0.2s;
        }
        .bl-row:hover .bl-load-bar {
          opacity: 0.7;
        }

        .bl-cell-status {
          display: flex;
          justify-content: flex-end;
        }
        .bl-status-tag {
          font-size: 9px;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          padding: 4px 12px;
          border: 1px solid;
          white-space: nowrap;
        }
        .bl-tag-auth {
          color: var(--primary);
          border-color: rgba(var(--primary-rgb), 0.25);
          background: rgba(var(--primary-rgb), 0.08);
        }
        .bl-tag-verified {
          color: #10b981;
          border-color: rgba(16,185,129,0.15);
          background: rgba(16,185,129,0.06);
        }

        /* ── Empty State ── */
        .bl-empty {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 12px;
          padding: 40px;
          height: 100%;
        }
        .bl-empty-icon {
          color: rgba(255,255,255,0.08);
        }
        .bl-empty span {
          font-size: 10px;
          font-weight: 700;
          color: rgba(255,255,255,0.2);
          text-transform: uppercase;
          letter-spacing: 0.15em;
        }
      `}</style>
    </div>
  );
};
