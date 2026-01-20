import React from 'react';

const TerminalMockup = () => {
  return (
    <div className="w-full rounded-xl bg-background border border-background-tertiary p-4 font-mono text-xs overflow-hidden shadow-inner">
      <div className="flex items-center gap-2 mb-4 border-b border-background-tertiary pb-2 opacity-50">
        <div className="w-2 h-2 rounded-full bg-text-dim" />
        <span className="text-[10px] uppercase tracking-widest text-text-dim">databoard-cli v1.0.2</span>
      </div>
      <div className="space-y-1.5">
        <div className="flex gap-2">
          <span className="text-primary">$</span>
          <span className="text-text">databoard upload data.csv</span>
        </div>
        <div className="text-text-dim">
          [1/3] Validating schema... <span className="text-primary">OK</span>
        </div>
        <div className="text-text-dim">
          [2/3] Uploading 1.2M rows... <span className="text-primary">DONE</span>
        </div>
        <div className="text-text-dim">
          [3/3] Generating insights...
        </div>
        <div className="flex gap-2 pt-2">
          <span className="text-primary">$</span>
          <span className="text-text">databoard analyze --model enterprise</span>
        </div>
        <div className="text-blue-400">
          ✓ Analysis complete. 4 new insights found.
        </div>
        <div className="text-text-dim italic">
          ➜ Open https://databoard.ai/d/x9f2 to view dashboard
        </div>
        <div className="flex gap-2 animate-pulse">
          <span className="text-primary">$</span>
          <div className="w-2 h-4 bg-primary" />
        </div>
      </div>
    </div>
  );
};

export default TerminalMockup;
