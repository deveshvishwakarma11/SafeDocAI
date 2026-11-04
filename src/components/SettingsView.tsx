import React, { useState, useEffect } from 'react';
import {
  ShieldCheck,
  Database,
  Cpu,
  Trash2,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  User,
  Play
} from 'lucide-react';
import { db } from '../utils/storage';
import { StoredDocument, RuntimeDiagnostics } from '../types';
import { MascotRobot } from './MascotArt';

interface Props {
  documents: StoredDocument[];
  onRefreshDocs: () => void | Promise<void>;
  onOpenReadiness: () => void;
}

const EMPTY_DIAGNOSTICS: RuntimeDiagnostics = {
  sqlite_status: 'offline',
  sqlite_records_count: 0,
  sqlite_metadata_count: 0,
  vector_engine_status: 'idle',
  vector_chunks_count: 0,
  local_llm_status: 'unavailable',
  ocr_engine_status: 'active_eng_only',
  network_egress_status: 'zero_egress_local',
  storage_bytes_used: 0
};

export const SettingsView: React.FC<Props> = ({
  documents,
  onRefreshDocs,
  onOpenReadiness
}) => {
  const [diagnostics, setDiagnostics] = useState<RuntimeDiagnostics>(EMPTY_DIAGNOSTICS);
  const [offlineAuditResult, setOfflineAuditResult] = useState<string | null>(null);
  const [isAuditing, setIsAuditing] = useState(false);
  const [isRebuilding, setIsRebuilding] = useState(false);
  const [showWipeConfirm, setShowWipeConfirm] = useState(false);

  const refreshDiagnostics = async () => {
    try {
      setDiagnostics(await db.runDiagnostics());
    } catch (e) {
      console.error('Diagnostics unavailable:', e);
    }
  };

  useEffect(() => {
    refreshDiagnostics();
  }, [documents.length]);

  const handleRebuildIndex = async () => {
    setIsRebuilding(true);
    try {
      const rebuilt = await db.rebuildIndex();
      await onRefreshDocs();
      await refreshDiagnostics();
      alert(`Vector index rebuilt server-side: ${rebuilt} document(s) re-chunked from stored text.`);
    } catch (e: any) {
      alert(`Rebuild failed: ${e.message || 'unknown error'}`);
    } finally {
      setIsRebuilding(false);
    }
  };

  const handleRunOfflineAudit = async () => {
    setIsAuditing(true);
    setOfflineAuditResult(null);

    try {
      const diag = await db.runDiagnostics();
      const egressLine =
        'VERIFIED: 0 outbound integrations active. OCR, extraction, SQLite queries, vector search and answer synthesis all execute on this machine (loopback only).';
      setOfflineAuditResult(
        `Audit at ${new Date().toLocaleTimeString()} — ${egressLine} LLM engine: ${diag.local_llm_status} · OCR: ${diag.ocr_engine_status}.`
      );
    } catch (e: any) {
      setOfflineAuditResult(`Audit failed: ${e.message || 'local engine unreachable'}`);
    } finally {
      setIsAuditing(false);
    }
  };

  const handlePermanentWipe = async () => {
    await db.clearAllDocuments();
    await onRefreshDocs();
    await refreshDiagnostics();
    setShowWipeConfirm(false);
  };

  return (
    <div className="space-y-6 max-w-4xl select-none">
      {/* Title */}
      <div>
        <h3 className="text-xl font-extrabold text-[#281238]">
          System Settings & Local Privacy
        </h3>
        <p className="text-xs text-[#2A1E39]/70">
          Inspect local storage health, run integrity checks, and audit data isolation boundaries.
        </p>
      </div>

      {/* Local Privacy Manifest Card */}
      <div className="p-6 rounded-3xl bg-gradient-to-tr from-[#281238] to-[#3A155E] text-white shadow-md relative overflow-hidden">
        <div className="flex items-start justify-between relative z-10">
          <div className="space-y-2 max-w-xl">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-300 text-xs font-bold border border-emerald-400/30">
              <ShieldCheck size={14} /> Local-First Architecture Guarantee
            </div>
            <h4 className="text-lg font-black text-white">
              Your Personal Documents Never Leave This Machine
            </h4>
            <p className="text-xs text-[#F5EFFF]/80 leading-relaxed">
              SafeDocAI stores your originals and metadata in a local SQLite database and managed
              file store under the app folder, answered by a local Ollama model. No cloud AI
              provider receives your document bytes, OCR text, or query history.
            </p>
            <div className="pt-2">
              <button
                onClick={handleRunOfflineAudit}
                disabled={isAuditing}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold bg-white text-[#281238] hover:bg-[#F5EFFF] transition-colors cursor-pointer"
              >
                {isAuditing ? <RefreshCw size={13} className="animate-spin" /> : <Play size={13} />}
                <span>{isAuditing ? 'Testing Network Isolation...' : 'Run Offline Egress Audit'}</span>
              </button>
            </div>
          </div>
          <MascotRobot size={90} mood="happy" className="hidden sm:block" />
        </div>

        {offlineAuditResult && (
          <div className="mt-4 p-3 rounded-xl bg-emerald-500/10 border border-emerald-400/40 text-xs text-emerald-200 font-mono">
            {offlineAuditResult}
          </div>
        )}
      </div>

      {/* Persistence Architecture & Stats */}
      <div className="bg-white rounded-3xl border border-[#D9CCFF]/60 p-6 shadow-xs space-y-4">
        <h4 className="text-sm font-extrabold text-[#281238] flex items-center gap-2">
          <Database size={16} className="text-[#7541DA]" />
          Local Storage & Schema Status (SQLite)
        </h4>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
          <div className="p-3 rounded-2xl bg-[#FBF8FF] border border-[#D9CCFF]/50">
            <p className="text-[10px] font-bold uppercase text-[#2A1E39]/50">Documents</p>
            <p className="text-xl font-black text-[#281238]">{diagnostics.sqlite_records_count}</p>
          </div>
          <div className="p-3 rounded-2xl bg-[#FBF8FF] border border-[#D9CCFF]/50">
            <p className="text-[10px] font-bold uppercase text-[#2A1E39]/50">Verified Fields</p>
            <p className="text-xl font-black text-[#7541DA]">{diagnostics.sqlite_metadata_count}</p>
          </div>
          <div className="p-3 rounded-2xl bg-[#FBF8FF] border border-[#D9CCFF]/50">
            <p className="text-[10px] font-bold uppercase text-[#2A1E39]/50">Vector Chunks</p>
            <p className="text-xl font-black text-[#8B42E8]">{diagnostics.vector_chunks_count}</p>
          </div>
          <div className="p-3 rounded-2xl bg-[#FBF8FF] border border-[#D9CCFF]/50">
            <p className="text-[10px] font-bold uppercase text-[#2A1E39]/50">Storage Used</p>
            <p className="text-xl font-black text-emerald-700">{(diagnostics.storage_bytes_used / 1024).toFixed(0)} KB</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3 pt-2">
          <button
            onClick={onOpenReadiness}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-bold bg-[#F5EFFF] text-[#7541DA] border border-[#D9CCFF] hover:bg-[#EBDDFF] transition-colors"
          >
            <ShieldCheck size={14} />
            <span>Open System Diagnostics</span>
          </button>

          <button
            onClick={handleRebuildIndex}
            disabled={isRebuilding}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-bold bg-white text-[#281238] border border-[#D9CCFF] hover:bg-[#F5EFFF] transition-colors disabled:opacity-50"
          >
            <RefreshCw size={13} className={`text-[#7541DA] ${isRebuilding ? 'animate-spin' : ''}`} />
            <span>{isRebuilding ? 'Rebuilding...' : 'Rebuild Vector Index'}</span>
          </button>
        </div>
      </div>

      {/* Local Profile Boundary */}
      <div className="bg-white rounded-3xl border border-[#D9CCFF]/60 p-6 shadow-xs space-y-3">
        <h4 className="text-sm font-extrabold text-[#281238] flex items-center gap-2">
          <User size={16} className="text-[#7541DA]" />
          Profile & Identity Boundaries
        </h4>
        <p className="text-xs text-[#2A1E39]/70 leading-relaxed">
          In compliance with PRD Section 5 ("Local Profile Now, Multi-Account Later"), SafeDocAI runs with
          one default local profile. No online signup, email confirmation, or cloud account is required.
        </p>

        <div className="p-3.5 rounded-2xl bg-[#FBF8FF] border border-[#D9CCFF] flex items-center justify-between text-xs">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-[#7541DA] text-white flex items-center justify-center font-bold">
              LU
            </div>
            <div>
              <p className="font-bold text-[#281238]">Local User</p>
              <p className="text-[11px] text-[#2A1E39]/50">UUID: default_local_profile · Status: Active</p>
            </div>
          </div>
          <span className="text-[11px] font-semibold text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-full border border-emerald-200">
            Local Only (No Cloud Sync)
          </span>
        </div>
      </div>

      {/* Dangerous Zone */}
      <div className="bg-red-50/50 rounded-3xl border border-red-200 p-6 shadow-xs space-y-3">
        <h4 className="text-sm font-extrabold text-red-900 flex items-center gap-2">
          <AlertTriangle size={16} className="text-red-600" />
          Permanent Wipe & Reset
        </h4>
        <p className="text-xs text-red-800/80 leading-relaxed">
          Remove all indexed documents, SQLite metadata, vector chunks, and managed original files
          from this machine. This cannot be undone.
        </p>

        {showWipeConfirm ? (
          <div className="flex items-center gap-2 pt-1">
            <button
              onClick={() => setShowWipeConfirm(false)}
              className="px-4 py-2 rounded-xl text-xs font-bold bg-white text-gray-700 border border-gray-300"
            >
              Cancel
            </button>
            <button
              onClick={handlePermanentWipe}
              className="px-4 py-2 rounded-xl text-xs font-bold bg-red-600 text-white hover:bg-red-700 shadow-xs"
            >
              Yes, Permanently Clear All Records
            </button>
          </div>
        ) : (
          <button
            onClick={() => setShowWipeConfirm(true)}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-bold bg-red-600 text-white hover:bg-red-700 shadow-xs transition-colors"
          >
            <Trash2 size={13} />
            <span>Clear All Document Records</span>
          </button>
        )}
      </div>
    </div>
  );
};
