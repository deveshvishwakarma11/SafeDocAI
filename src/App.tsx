import React, { useState, useEffect } from 'react';
import { Sidebar, NavTab } from './components/Sidebar';
import { Header } from './components/Header';
import { LandingHero } from './components/LandingHero';
import { DashboardView } from './components/DashboardView';
import { ChatView } from './components/ChatView';
import { DocumentsView } from './components/DocumentsView';
import { SettingsView } from './components/SettingsView';
import { RuntimeReadinessModal } from './components/RuntimeReadinessModal';
import { OnboardingModal } from './components/OnboardingModal';
import { db } from './utils/storage';
import { StoredDocument, ChatMessage } from './types';
import { Home, MessageSquare, FolderArchive, UploadCloud, Sliders } from 'lucide-react';

// Lazy-load DocumentViewerModal to improve initial bundle size and perceived performance
const DocumentViewerModal = React.lazy(() =>
  import('./components/DocumentViewerModal').then(module => ({ default: module.DocumentViewerModal }))
);

export default function App() {
  const [activeTab, setActiveTab] = useState<NavTab>('dashboard');
  const [documents, setDocuments] = useState<StoredDocument[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [selectedDocForViewer, setSelectedDocForViewer] = useState<StoredDocument | null>(null);
  // The answer scope outlives the viewer modal: closing the viewer keeps the
  // chat pinned to that file until another document (or "All") is chosen.
  const [activeDocumentId, setActiveDocumentId] = useState<number | undefined>(undefined);
  const [isReadinessOpen, setIsReadinessOpen] = useState(false);
  const [isOnboardingOpen, setIsOnboardingOpen] = useState(false);
  const [showHero, setShowHero] = useState(true);
  const [pendingQuery, setPendingQuery] = useState<string | undefined>(undefined);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  // Load the SQLite-backed inventory from the local server on mount
  useEffect(() => {
    db.getAllDocuments()
      .then(setDocuments)
      .catch(err => console.error('Failed to load documents from local engine:', err));
  }, []);

  const refreshDocuments = async () => {
    try {
      setDocuments(await db.getAllDocuments());
    } catch (err) {
      console.error('Failed to refresh documents:', err);
    }
  };

  const handleQueryFromHome = (query: string) => {
    setPendingQuery(query);
    setActiveTab('chat');
  };

  const handleOpenViewer = (doc: StoredDocument) => {
    setSelectedDocForViewer(doc);
    setActiveDocumentId(doc.id);
  };

  const handleDocumentRemoved = async (id: number) => {
    await db.removeDocument(id);
    refreshDocuments();
    if (selectedDocForViewer?.id === id) {
      setSelectedDocForViewer(null);
    }
    if (activeDocumentId === id) {
      setActiveDocumentId(undefined);
    }
  };

  const handleToggleSidebar = () => {
    if (window.innerWidth < 768) {
      setIsMobileSidebarOpen(prev => !prev);
    } else {
      setIsSidebarCollapsed(prev => !prev);
    }
  };

  const totalBytes = documents.reduce((acc, d) => acc + (d.file_size_bytes || 0), 0);

  return (
    <div className="flex h-screen bg-[#FBF8FF] overflow-hidden">
      {/* Navigation Sidebar (Desktop + Mobile Drawer) */}
      <Sidebar
        activeTab={activeTab}
        onSelectTab={tab => {
          setActiveTab(tab);
          if (tab === 'chat') {
            setPendingQuery(undefined);
          }
        }}
        documentCount={documents.length}
        totalBytes={totalBytes}
        onOpenReadiness={() => setIsReadinessOpen(true)}
        isCollapsed={isSidebarCollapsed}
        onToggleCollapse={() => setIsSidebarCollapsed(prev => !prev)}
        isMobileOpen={isMobileSidebarOpen}
        onCloseMobile={() => setIsMobileSidebarOpen(false)}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-screen overflow-y-auto min-w-0 pb-16 md:pb-0">
        <Header
          activeTab={activeTab}
          onOpenReadiness={() => setIsReadinessOpen(true)}
          onOpenOnboarding={() => setIsOnboardingOpen(true)}
          documentCount={documents.length}
          onToggleSidebar={handleToggleSidebar}
        />

        <main className="p-4 sm:p-6 max-w-7xl w-full mx-auto space-y-6">
          {/* Landing Hero (can be toggled or viewed on Dashboard) */}
          {activeTab === 'dashboard' && showHero && (
            <div className="relative">
              <LandingHero
                onGetStarted={() => {
                  setShowHero(false);
                }}
                onOpenGuide={() => setIsOnboardingOpen(true)}
                onOpenReadiness={() => setIsReadinessOpen(true)}
                documentCount={documents.length}
              />
              <button
                onClick={() => setShowHero(false)}
                className="absolute top-4 right-4 text-white/60 hover:text-white text-xs px-2.5 py-1 rounded-lg bg-black/20 hover:bg-black/40 backdrop-blur-xs transition-colors cursor-pointer"
              >
                Hide Hero
              </button>
            </div>
          )}

          {/* Active Tab Screen */}
          {activeTab === 'dashboard' && (
            <DashboardView
              documents={documents}
              onSelectTab={setActiveTab}
              onOpenViewer={handleOpenViewer}
              onSubmitQuery={handleQueryFromHome}
            />
          )}

          {activeTab === 'chat' && (
            <ChatView
              messages={messages}
              setMessages={setMessages}
              documents={documents}
              onOpenViewer={handleOpenViewer}
              initialQuery={pendingQuery}
              scopedDocumentId={activeDocumentId}
              onScopeChange={setActiveDocumentId}
            />
          )}

          {(activeTab === 'documents' || activeTab === 'upload') && (
            <DocumentsView
              documents={documents}
              onRefreshDocs={refreshDocuments}
              onOpenViewer={handleOpenViewer}
            />
          )}

          {activeTab === 'settings' && (
            <SettingsView
              documents={documents}
              onRefreshDocs={refreshDocuments}
              onOpenReadiness={() => setIsReadinessOpen(true)}
            />
          )}
        </main>
      </div>

      {/* Mobile Bottom Navigation Bar (< 768px screens) */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 z-40 bg-white/95 backdrop-blur-md border-t border-[#D9CCFF]/60 px-3 py-2 flex items-center justify-around shadow-lg">
        {[
          { id: 'dashboard' as NavTab, label: 'Home', icon: Home },
          { id: 'chat' as NavTab, label: 'Chat', icon: MessageSquare },
          { id: 'documents' as NavTab, label: 'Docs', icon: FolderArchive, badge: documents.length },
          { id: 'upload' as NavTab, label: 'Upload', icon: UploadCloud },
          { id: 'settings' as NavTab, label: 'Settings', icon: Sliders }
        ].map(item => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => {
                setActiveTab(item.id);
                if (item.id === 'chat') setPendingQuery(undefined);
              }}
              className={`flex flex-col items-center justify-center py-1 px-2.5 rounded-xl transition-all cursor-pointer relative ${
                isActive ? 'text-[#7541DA]' : 'text-[#2A1E39]/60 hover:text-[#7541DA]'
              }`}
            >
              <div className="relative">
                <Icon size={19} className={isActive ? 'stroke-[2.5]' : ''} />
                {item.badge !== undefined && item.badge > 0 && (
                  <span className="absolute -top-1 -right-2 bg-[#7541DA] text-white text-[9px] font-extrabold w-4 h-4 rounded-full flex items-center justify-center">
                    {item.badge}
                  </span>
                )}
              </div>
              <span className={`text-[10px] mt-0.5 ${isActive ? 'font-extrabold' : 'font-medium'}`}>
                {item.label}
              </span>
            </button>
          );
        })}
      </nav>

      {/* Modals with Lazy-Loading */}
      {selectedDocForViewer && (
        <React.Suspense
          fallback={
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#281238]/60 backdrop-blur-xs p-4">
              <div className="bg-white rounded-3xl p-8 max-w-sm w-full text-center space-y-3 border border-[#D9CCFF] shadow-2xl">
                <div className="w-12 h-12 rounded-2xl bg-[#F5EFFF] text-[#7541DA] flex items-center justify-center mx-auto">
                  <div className="w-6 h-6 border-3 border-[#7541DA] border-t-transparent rounded-full animate-spin" />
                </div>
                <h4 className="text-sm font-extrabold text-[#281238]">
                  Loading Document Viewer...
                </h4>
                <p className="text-xs text-[#2A1E39]/60">
                  Initializing local rendering engine and sandboxed viewer
                </p>
              </div>
            </div>
          }
        >
          <DocumentViewerModal
            document={selectedDocForViewer}
            onClose={() => setSelectedDocForViewer(null)}
            onRemove={handleDocumentRemoved}
            onDocumentUpdated={refreshDocuments}
          />
        </React.Suspense>
      )}

      <RuntimeReadinessModal
        isOpen={isReadinessOpen}
        onClose={() => setIsReadinessOpen(false)}
      />

      <OnboardingModal
        isOpen={isOnboardingOpen}
        onClose={() => setIsOnboardingOpen(false)}
        onGetStarted={() => {
          setActiveTab('dashboard');
        }}
      />
    </div>
  );
}
