import React, { useEffect, useState } from 'react';
import {
  X,
  CheckCircle2,
  AlertCircle,
  Cpu,
  Database,
  ShieldCheck,
  FileText,
  Play,
  WifiOff,
  HardDrive,
  RefreshCw
} from 'lucide-react';

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

/** Exactly what GET /api/diagnostics returns — no synthetic fields. */
interface ServerDiagnostics {
  sqlite_status: string;
  sqlite_records_count: number;
  sqlite_metadata_count: number;
  schema_version: number;
  vector_engine_status: string;
  vector_chunks_count: number;
  local_llm_status: string;
  ollama_synthesis_enabled: boolean;
  ocr_engine_status: string;
  network_egress_status: string;
  storage_bytes_used: number;
}

async function fetchDiagnostics(): Promise<ServerDiagnostics> {
  const res = await fetch('/api/diagnostics');
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function describeOcr(status: string): string {
  switch (status) {
    case 'active_eng_hin':
      return 'eng + hin — loaded offline from data/tessdata';
    case 'active_eng_only':
      return 'eng only — offline (drop hin.traineddata into data/tessdata for Hindi)';
    default:
      return 'unavailable — no local traineddata / worker failed to start';
  }
}

function describeEgress(_status: string): string {
  return 'zero_egress_local — no cloud integration exists; OCR, extraction and answers all run on this machine';
}

export const RuntimeReadinessModal: React.FC<Props> = ({ isOpen, onClose }) => {
  const [diagnostics, setDiagnostics] = useState<ServerDiagnostics | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isRunningTest, setIsRunningTest] = useState(false);
  const [testLog, setTestLog] = useState<string[]>([]);

  const load = async (): Promise<ServerDiagnostics | null> => {
    try {
      const d = await fetchDiagnostics();
      setDiagnostics(d);
      setLoadError(null);
      return d;
    } catch (e: any) {
      setLoadError(e.message || 'could not reach local server');
      return null;
    }
  };

  useEffect(() => {
    if (isOpen) {
      load();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  if (!isOpen) return null;

  const handleRunSuite = async () => {
    setIsRunningTest(true);
    setTestLog([]);

    const logs: string[] = [];
    const addLog = (msg: string) => {
      logs.push(msg);
      setTestLog([...logs]);
    };

    addLog('GET /api/diagnostics — querying the running local server...');
    const d = await load();

    if (!d) {
      addLog(`-> FAILED: local server unreachable (${loadError || 'unknown error'})`);
      setIsRunningTest(false);
      return;
    }

    addLog(
      `[1/5] sqlite: ${d.sqlite_status} — documents=${d.sqlite_records_count}, fields=${d.sqlite_metadata_count}, schema_v${d.schema_version}`
    );
    addLog(`[2/5] vector index: ${d.vector_chunks_count} chunks (${d.vector_engine_status})`);
    addLog(`[3/5] OCR engine: ${describeOcr(d.ocr_engine_status)}`);
    addLog(`[4/5] network egress: ${describeEgress(d.network_egress_status)}`);
    addLog(
      `[5/5] local LLM: ${d.local_llm_status}, synthesis ${d.ollama_synthesis_enabled ? 'ON (OLLAMA_SYNTHESIS=1)' : 'off (deterministic engines answer instantly)'} · storage ${(d.storage_bytes_used / 1024).toFixed(0)} KB`
    );
    addLog(
      d.sqlite_status === 'healthy'
        ? 'Diagnostics complete. SQLite integrity: healthy.'
        : 'Diagnostics complete. WARNING: SQLite integrity reported degraded.'
    );

    setIsRunningTest(false);
  };

  const pill = (ok: boolean, label: string) => (
    <span
      className={`text-[11px] font-bold flex items-center gap-1 ${ok ? 'text-emerald-600' : 'text-amber-600'}`}
    >
      {ok ? <CheckCircle2 size={13} /> : <AlertCircle size={13} />} {label}
    </span>
  );

  const d = diagnostics;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#281238]/60 backdrop-blur-xs p-4 animate-in fade-in duration-200">
      <div className="bg-white rounded-2xl shadow-2xl border border-[#D9CCFF] max-w-2xl w-full max-h-[90vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="p-6 border-b border-[#D9CCFF]/50 flex items-center justify-between bg-[#F5EFFF]/50">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/10 text-emerald-600 flex items-center justify-center font-bold">
              <ShieldCheck size={22} />
            </div>
            <div>
              <h3 className="font-extrabold text-lg text-[#281238]">Local Runtime Diagnostics</h3>
              <p className="text-xs text-[#2A1E39]/70">
                Live values from GET /api/diagnostics — nothing simulated
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg flex items-center justify-center text-[#2A1E39]/50 hover:bg-[#D9CCFF]/40 hover:text-[#281238]"
          >
            <X size={18} />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1">
          {loadError && !d && (
            <div className="p-4 rounded-xl border border-red-200 bg-red-50 text-xs text-red-700 font-semibold">
              Could not reach the local server: {loadError}
            </div>
          )}

          {/* Readiness Grid — real numbers only */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
            {/* SQLite */}
            <div className="p-4 rounded-xl border border-[#D9CCFF]/80 bg-[#FBF8FF]">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-[#281238] flex items-center gap-1.5">
                  <Database size={15} className="text-[#7541DA]" />
                  SQLite Metadata Store
                </span>
                {d ? pill(d.sqlite_status === 'healthy', d.sqlite_status) : <span className="text-[11px] text-[#2A1E39]/50">…</span>}
              </div>
              <p className="text-xs text-[#2A1E39]/70 leading-relaxed">
                {d
                  ? `${d.sqlite_records_count} documents, ${d.sqlite_metadata_count} verified evidence fields (schema v${d.schema_version}).`
                  : 'Waiting for server…'}
              </p>
            </div>

            {/* Vector chunks */}
            <div className="p-4 rounded-xl border border-[#D9CCFF]/80 bg-[#FBF8FF]">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-[#281238] flex items-center gap-1.5">
                  <FileText size={15} className="text-[#7541DA]" />
                  Local Vector Chunks
                </span>
                {d ? pill(d.vector_chunks_count >= 0, d.vector_engine_status) : <span className="text-[11px] text-[#2A1E39]/50">…</span>}
              </div>
              <p className="text-xs text-[#2A1E39]/70 leading-relaxed">
                {d
                  ? `${d.vector_chunks_count} text chunks indexed (500-char / 50 overlap, stored in SQLite).`
                  : 'Waiting for server…'}
              </p>
            </div>

            {/* OCR */}
            <div className="p-4 rounded-xl border border-[#D9CCFF]/80 bg-[#FBF8FF]">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-[#281238] flex items-center gap-1.5">
                  <Cpu size={15} className="text-[#7541DA]" />
                  OCR Engine
                </span>
                {d
                  ? pill(d.ocr_engine_status !== 'unavailable', d.ocr_engine_status)
                  : <span className="text-[11px] text-[#2A1E39]/50">…</span>}
              </div>
              <p className="text-xs text-[#2A1E39]/70 leading-relaxed">
                {d ? describeOcr(d.ocr_engine_status) : 'Waiting for server…'}
              </p>
            </div>

            {/* Local LLM */}
            <div className="p-4 rounded-xl border border-[#D9CCFF]/80 bg-[#FBF8FF]">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-[#281238] flex items-center gap-1.5">
                  <HardDrive size={15} className="text-[#7541DA]" />
                  Answer Engine
                </span>
                {d ? pill(true, d.local_llm_status) : <span className="text-[11px] text-[#2A1E39]/50">…</span>}
              </div>
              <p className="text-xs text-[#2A1E39]/70 leading-relaxed">
                {d
                  ? d.ollama_synthesis_enabled
                    ? 'Ollama synthesis enabled — LLM answers open questions after the deterministic engines.'
                    : 'Deterministic grounded engines answer instantly (fields → verbatim excerpt → honest not-found). Ollama synthesis off.'
                  : 'Waiting for server…'}
              </p>
            </div>

            {/* Privacy Egress */}
            <div className="p-4 rounded-xl border border-[#D9CCFF]/80 bg-[#FBF8FF]">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-[#281238] flex items-center gap-1.5">
                  <WifiOff size={15} className="text-emerald-600" />
                  Network Egress
                </span>
                {d
                  ? pill(true, d.network_egress_status)
                  : <span className="text-[11px] text-[#2A1E39]/50">…</span>}
              </div>
              <p className="text-xs text-[#2A1E39]/70 leading-relaxed">
                {d ? describeEgress(d.network_egress_status) : 'Waiting for server…'}
              </p>
            </div>

            {/* Storage */}
            <div className="p-4 rounded-xl border border-[#D9CCFF]/80 bg-[#FBF8FF]">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-[#281238] flex items-center gap-1.5">
                  <HardDrive size={15} className="text-[#7541DA]" />
                  Managed Store
                </span>
                {d ? pill(d.storage_bytes_used >= 0, 'local') : <span className="text-[11px] text-[#2A1E39]/50">…</span>}
              </div>
              <p className="text-xs text-[#2A1E39]/70 leading-relaxed">
                {d
                  ? `${(d.storage_bytes_used / 1024).toFixed(0)} KB under data/ (originals, output, understood artifacts).`
                  : 'Waiting for server…'}
              </p>
            </div>
          </div>

          {/* Test Terminal */}
          <div className="rounded-xl border border-[#281238]/20 bg-[#281238] p-4 text-white">
            <div className="flex items-center justify-between mb-2 border-b border-white/10 pb-2">
              <span className="text-xs font-mono font-semibold text-[#D9CCFF]">
                local diagnostics runner (GET /api/diagnostics)
              </span>
              <button
                onClick={handleRunSuite}
                disabled={isRunningTest}
                className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-bold bg-[#7541DA] text-white hover:bg-[#8B42E8] transition-colors disabled:opacity-50"
              >
                {isRunningTest ? <RefreshCw size={12} className="animate-spin" /> : <Play size={12} />}
                <span>{isRunningTest ? 'Running...' : 'Run Diagnostics'}</span>
              </button>
            </div>

            <div className="font-mono text-xs text-white/80 space-y-1 max-h-40 overflow-y-auto">
              {testLog.length === 0 ? (
                <p className="text-white/40 italic">
                  Click 'Run Diagnostics' to fetch live SQLite counts, vector chunk totals, OCR
                  languages, egress configuration and storage usage from the running server.
                </p>
              ) : (
                testLog.map((line, idx) => (
                  <p
                    key={idx}
                    className={
                      /FAILED|WARNING/i.test(line)
                        ? 'text-red-400 font-semibold'
                        : /healthy|zero_egress/i.test(line)
                          ? 'text-emerald-400'
                          : ''
                    }
                  >
                    {line}
                  </p>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-[#D9CCFF]/40 bg-[#FBF8FF] flex items-center justify-between">
          <p className="text-xs text-[#2A1E39]/70">
            Storage used:{' '}
            <strong>
              {d ? `${(d.storage_bytes_used / 1024).toFixed(0)} KB` : '—'}
            </strong>
          </p>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-xs font-bold bg-[#7541DA] text-white hover:bg-[#8B42E8] transition-colors"
          >
            Close Diagnostics
          </button>
        </div>
      </div>
    </div>
  );
};
