import React from 'react';
import {
  Home,
  MessageSquare,
  FolderArchive,
  UploadCloud,
  Sliders,
  ShieldCheck,
  HardDrive,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  X,
  Menu
} from 'lucide-react';
import { MascotRobot } from './MascotArt';

export type NavTab = 'dashboard' | 'chat' | 'documents' | 'upload' | 'settings';

interface SidebarProps {
  activeTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  documentCount: number;
  totalBytes: number;
  onOpenReadiness: () => void;
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
  isMobileOpen?: boolean;
  onCloseMobile?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onSelectTab,
  documentCount,
  totalBytes,
  onOpenReadiness,
  isCollapsed = false,
  onToggleCollapse,
  isMobileOpen = false,
  onCloseMobile
}) => {
  const mbUsed = (totalBytes / (1024 * 1024)).toFixed(1);
  const percentUsed = Math.max(2, Math.min(100, Math.round((totalBytes / (10 * 1024 * 1024 * 1024)) * 100)));

  const navItems = [
    { id: 'dashboard' as NavTab, label: 'Home Dashboard', icon: Home, count: null },
    { id: 'chat' as NavTab, label: 'AI Chat & Ask', icon: MessageSquare, count: null },
    { id: 'documents' as NavTab, label: 'Document Library', icon: FolderArchive, count: documentCount },
    { id: 'upload' as NavTab, label: 'Upload & Index', icon: UploadCloud, count: null },
    { id: 'settings' as NavTab, label: 'Settings & Privacy', icon: Sliders, count: null }
  ];

  const handleItemClick = (id: NavTab) => {
    onSelectTab(id);
    if (onCloseMobile) {
      onCloseMobile();
    }
  };

  const sidebarContent = (isDrawer = false) => (
    <div className="flex flex-col h-full bg-white select-none">
      {/* Brand Header */}
      <div className={`p-4 border-b border-[#D9CCFF]/30 flex items-center justify-between ${isCollapsed && !isDrawer ? 'px-2' : 'px-4'}`}>
        <div
          className="flex items-center gap-3 cursor-pointer overflow-hidden"
          onClick={() => handleItemClick('dashboard')}
          title="SafeDocAI Home"
        >
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-[#7541DA] to-[#8B42E8] flex items-center justify-center text-white shadow-sm shadow-[#7541DA]/20 shrink-0">
            <MascotRobot size={30} mood="happy" />
          </div>
          {(!isCollapsed || isDrawer) && (
            <div className="min-w-0">
              <h1 className="font-extrabold text-base text-[#281238] tracking-tight flex items-center gap-1.5 truncate">
                SafeDocAI
                <span className="text-[10px] font-semibold bg-[#F5EFFF] text-[#7541DA] px-1.5 py-0.5 rounded border border-[#D9CCFF]">
                  v1.0
                </span>
              </h1>
              <p className="text-[10px] uppercase font-bold tracking-wider text-[#7541DA]/70 truncate">
                Personal Document Intel
              </p>
            </div>
          )}
        </div>

        {/* Close button for drawer or collapse toggle for desktop */}
        {isDrawer ? (
          <button
            onClick={onCloseMobile}
            className="p-1.5 rounded-lg text-[#2A1E39]/60 hover:bg-[#F5EFFF] hover:text-[#281238] transition-colors cursor-pointer"
            title="Close navigation drawer"
          >
            <X size={18} />
          </button>
        ) : onToggleCollapse && (
          <button
            onClick={onToggleCollapse}
            className="p-1.5 rounded-lg text-[#2A1E39]/50 hover:bg-[#F5EFFF] hover:text-[#7541DA] transition-colors cursor-pointer hidden lg:flex items-center justify-center"
            title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {isCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
          </button>
        )}
      </div>

      {/* Navigation Links */}
      <nav className="p-3 space-y-1.5 flex-1 overflow-y-auto">
        {(!isCollapsed || isDrawer) && (
          <p className="px-3 pt-2 pb-1 text-[11px] font-bold tracking-wider uppercase text-[#2A1E39]/50">
            Navigation
          </p>
        )}
        {navItems.map(item => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => handleItemClick(item.id)}
              title={item.label}
              className={`w-full flex items-center rounded-xl font-semibold transition-all duration-150 cursor-pointer ${
                isCollapsed && !isDrawer
                  ? 'justify-center p-3 text-sm'
                  : 'justify-between px-3.5 py-2.5 text-sm'
              } ${
                isActive
                  ? 'bg-[#7541DA] text-white shadow-sm shadow-[#7541DA]/25'
                  : 'text-[#2A1E39]/80 hover:bg-[#F5EFFF] hover:text-[#7541DA]'
              }`}
            >
              <div className="flex items-center gap-3 min-w-0">
                <Icon size={19} className={isActive ? 'text-white' : 'text-[#7541DA] shrink-0'} />
                {(!isCollapsed || isDrawer) && <span className="truncate">{item.label}</span>}
              </div>
              {(!isCollapsed || isDrawer) && item.count !== null && (
                <span
                  className={`text-xs px-2 py-0.5 rounded-full font-bold shrink-0 ${
                    isActive ? 'bg-white/20 text-white' : 'bg-[#F5EFFF] text-[#7541DA]'
                  }`}
                >
                  {item.count}
                </span>
              )}
            </button>
          );
        })}

        {/* Runtime Readiness Card */}
        {(!isCollapsed || isDrawer) && (
          <div className="pt-4">
            <div
              onClick={() => {
                onOpenReadiness();
                if (isDrawer && onCloseMobile) onCloseMobile();
              }}
              className="p-3 rounded-xl bg-gradient-to-b from-[#F5EFFF] to-white border border-[#D9CCFF]/60 hover:border-[#7541DA] transition-colors cursor-pointer group"
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[11px] font-bold text-[#281238] flex items-center gap-1.5">
                  <ShieldCheck size={14} className="text-emerald-600" />
                  Local Runtime Status
                </span>
                <span className="inline-flex items-center gap-1 text-[10px] text-emerald-600 font-semibold">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                  100% Local
                </span>
              </div>
              <p className="text-[11px] text-[#2A1E39]/70 leading-relaxed">
                SQLite + Vector Embeddings offline. Zero cloud document transfer.
              </p>
              <div className="mt-2 flex items-center gap-1 text-[11px] font-bold text-[#7541DA] group-hover:underline">
                <span>View Diagnostics</span>
                <span>→</span>
              </div>
            </div>
          </div>
        )}
      </nav>

      {/* Storage and Profile Footer */}
      {(!isCollapsed || isDrawer) ? (
        <div className="p-4 border-t border-[#D9CCFF]/40 bg-[#FBF8FF]/50 space-y-3 shrink-0">
          {/* Storage Bar */}
          <div>
            <div className="flex items-center justify-between text-xs mb-1.5">
              <span className="text-[#2A1E39]/70 font-medium flex items-center gap-1">
                <HardDrive size={13} className="text-[#7541DA]" /> Storage
              </span>
              <span className="font-semibold text-[#281238]">
                {mbUsed} MB <span className="text-[#2A1E39]/50 font-normal">used · managed store</span>
              </span>
            </div>
            <div className="w-full bg-[#E8DEFF] h-2 rounded-full overflow-hidden">
              <div
                className="bg-gradient-to-r from-[#7541DA] to-[#8B42E8] h-full rounded-full transition-all duration-300"
                style={{ width: `${percentUsed}%` }}
              />
            </div>
          </div>

          {/* Profile Info */}
          <div className="flex items-center gap-2.5 p-2 rounded-xl bg-white border border-[#D9CCFF]/50 shadow-xs">
            <div className="w-8 h-8 rounded-full bg-[#7541DA]/15 flex items-center justify-center text-[#7541DA] font-bold text-xs">
              LU
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs font-bold text-[#281238] truncate">
                Local User
              </p>
              <p className="text-[10px] text-[#7541DA] font-medium truncate flex items-center gap-1">
                <CheckCircle2 size={10} className="text-emerald-500" /> Local Profile
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="p-2 border-t border-[#D9CCFF]/40 flex flex-col items-center gap-2 py-3">
          <div
            onClick={onToggleCollapse}
            className="w-10 h-10 rounded-xl hover:bg-[#F5EFFF] text-[#7541DA] flex items-center justify-center cursor-pointer"
            title="Expand Sidebar"
          >
            <ChevronRight size={18} />
          </div>
        </div>
      )}
    </div>
  );

  return (
    <>
      {/* 1. Desktop / Normal View Sidebar */}
      <aside
        className={`hidden md:flex flex-col h-screen border-r border-[#D9CCFF]/40 shrink-0 sticky top-0 z-30 transition-all duration-200 ${
          isCollapsed ? 'w-20' : 'w-64'
        }`}
      >
        {sidebarContent(false)}
      </aside>

      {/* 2. Mobile / Narrow Screen Drawer Backdrop Overlay */}
      {isMobileOpen && (
        <div
          className="fixed inset-0 bg-[#281238]/60 backdrop-blur-xs z-50 md:hidden animate-in fade-in duration-200"
          onClick={onCloseMobile}
        />
      )}

      {/* 3. Mobile / Narrow Screen Slide-Over Drawer */}
      <div
        className={`fixed inset-y-0 left-0 w-72 max-w-[85vw] bg-white z-50 md:hidden shadow-2xl transition-transform duration-300 ease-out border-r border-[#D9CCFF] ${
          isMobileOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {sidebarContent(true)}
      </div>
    </>
  );
};
