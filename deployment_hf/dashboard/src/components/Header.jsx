import React from 'react';
import { Network, ShieldCheck, Activity, GraduationCap } from 'lucide-react';

export const Header = ({ status }) => {
  return (
    <header className="shell-header glass-header flex items-center justify-between transition-all duration-300">
      <div className="flex items-center gap-4">
        <div className="w-10 h-10 bg-primary flex items-center justify-center text-white font-serif text-xl shadow-lg">FL</div>
        <div className="flex flex-col">
          <h1 className="type-l2 serif text-text-main leading-tight tracking-tight font-medium">
            AI Guardian <span className="text-accent underline decoration-accent/30 decoration-2 underline-offset-4 font-serif italic italic">Professional</span>
          </h1>
          <span className="type-label-bold opacity-30">INSTITUTIONAL FEDERATED LAB</span>
        </div>
      </div>

      <div className="flex items-center gap-6">
        <div className="flex items-center gap-3 px-4 py-1.5 bg-success/5 border border-success/20 rounded-sm">
          <div className="w-2 h-2 rounded-full bg-success shadow-[0_0_12px_var(--success)] animate-pulse" />
          <span className="type-label-bold text-success">{status}</span>
        </div>
        <div className="flex items-center gap-3 px-4 py-1.5 bg-primary/5 border border-primary/20 rounded-sm group hover:border-accent/50 transition-colors">
          <ShieldCheck size={14} className="text-primary group-hover:text-accent transition-colors" />
          <span className="type-label-bold text-primary italic">Node Secured</span>
        </div>
      </div>
    </header>
  );
};
