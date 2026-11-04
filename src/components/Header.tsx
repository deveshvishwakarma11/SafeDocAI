import React from 'react';
import { BookOpen, Activity, Menu } from 'lucide-react';
import { NavTab } from './Sidebar';
import { MascotRobot } from './MascotArt';

interface HeaderProps {
  activeTab: NavTab;
  onOpenReadiness: () => void;
  onOpenOnboarding: () => void;
  documentCount: number;
  onToggleSidebar: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onOpenReadiness,
  onOpenOnboarding,
  documentCount,
  onToggleSidebar
}) => {
  const getTabTitle = () => {
    switch (activeTab) {
      case 'dashboard':
        return 'Personal Knowledge Workspace';
      case 'chat':
        return 'Grounded Document Intelligence & Search';
      case 'documents':
        return 'Document Inventory & Evidence Store';
      case 'upload':
        return 'Multi-Stage Ingestion & Indexer';
      case 'settings':
        return 'System Settings & Privacy Audits';
    }
  };

  return (
    <header className="h-16 bg-white/90 backdrop-blur-md border-b border-[#D9CCFF]/40 px-4 sm:px-6 flex items-center justify-between sticky top-0 z-20 select-none">
      <div className="flex items-center gap-2 sm:gap-3 min-w-0">
        {/* Toggle Sidebar Button */}
        <button
          onClick={onToggleSidebar}
          className="p-2 rounded-xl text-[#7541DA] bg-[#F5EFFF] hover:bg-[#EBDDFF] border border-[#D9CCFF] transition-all cursor-pointer flex items-center justify-center shrink-0 shadow-xs"
          title="Open / Toggle Navigation Sidebar"
          aria-label="Toggle Sidebar Navigation"
        >
          <Menu size={19} />
        </button>

        {/* Brand logo for mobile screens */}
        <div className="flex md:hidden items-center gap-1.5 shrink-0">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-[#7541DA] to-[#8B42E8] flex items-center justify-center text-white shadow-xs">
            <MascotRobot size={22} mood="happy" />
          </div>
        </div>

        <div className="min-w-0">
          <h2 className="text-sm font-extrabold text-[#281238] tracking-tight truncate">
            {getTabTitle()}
          </h2>
          <p className="text-[11px] text-[#2A1E39]/60 flex items-center gap-1.5 truncate">
            <span>Local SQLite · offline</span>
            <span aria-hidden="true">·</span>
            <span>{documentCount} {documentCount === 1 ? 'doc' : 'docs'}</span>
          </p>
        </div>
      </div>

      <div className="flex items-center gap-1.5 sm:gap-2.5 shrink-0">
        {/* Onboarding Guide */}
        <button
          onClick={onOpenOnboarding}
          className="inline-flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-xl text-xs font-semibold bg-white text-[#281238] border border-[#D9CCFF]/70 hover:bg-[#F5EFFF] transition-colors cursor-pointer"
        >
          <BookOpen size={12} className="text-[#7541DA]" />
          <span className="hidden sm:inline">Guide</span>
        </button>

        {/* Runtime Diagnostics */}
        <button
          onClick={onOpenReadiness}
          className="inline-flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-xl text-xs font-semibold bg-white text-[#281238] border border-[#D9CCFF]/70 hover:border-[#7541DA] transition-colors cursor-pointer"
        >
          <Activity size={12} className="text-emerald-600" />
          <span className="hidden sm:inline">Diagnostics</span>
        </button>

        {/* Privacy Offline Badge */}
        <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-emerald-50 text-emerald-700 border border-emerald-200 text-xs font-semibold">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span>Local Only</span>
        </div>
      </div>
    </header>
  );
};
