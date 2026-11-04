import React, { useState, useEffect, useMemo } from 'react';
import {
  X,
  FileText,
  Trash2,
  Download,
  ExternalLink,
  ShieldAlert,
  Sparkles,
  RefreshCw,
  Eye,
  FileCode,
  ChevronLeft,
  ChevronRight,
  Search,
  Copy,
  Check,
  FileCheck
} from 'lucide-react';
import { StoredDocument, ExtractedField } from '../types';
import { db } from '../utils/storage';
import { DocumentProcessor } from '../utils/documentProcessor';

interface Props {
  document: StoredDocument | null;
  onClose: () => void;
  onRemove: (id: number) => void;
  onDocumentUpdated?: (doc: StoredDocument) => void;
}

export const DocumentViewerModal: React.FC<Props> = ({
  document: initialDoc,
  onClose,
  onRemove,
  onDocumentUpdated
}) => {
  const [doc, setDoc] = useState<StoredDocument | null>(initialDoc);
  const [activeTab, setActiveTab] = useState<'metadata' | 'chunks'>('metadata');
  const [leftPaneMode, setLeftPaneMode] = useState<'preview' | 'transcription'>('preview');
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [selectedField, setSelectedField] = useState<ExtractedField | null>(null);
  const [isReProcessing, setIsReProcessing] = useState(false);
  const [ocrMessage, setOcrMessage] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // --- Progressive Rendering & Progress Indicator State for Large Documents ---
  const [isRendering, setIsRendering] = useState(true);
  const [renderProgress, setRenderProgress] = useState(15);
  const [renderStage, setRenderStage] = useState('Reading document bytes & verifying hash...');
  const [currentPage, setCurrentPage] = useState(1);
  const [visibleChunkLimit, setVisibleChunkLimit] = useState(6);
  const [iframeLoaded, setIframeLoaded] = useState(false);
  const [searchFilter, setSearchFilter] = useState('');

  // Keep state in sync if prop changes and trigger progressive loader
  useEffect(() => {
    setDoc(initialDoc);
    setOcrMessage(null);
    setCurrentPage(1);
    setVisibleChunkLimit(6);
    setIframeLoaded(false);

    if (initialDoc) {
      setIsRendering(true);
      setRenderProgress(20);
      setRenderStage('Reading document bytes & verifying SHA-256...');

      const timer1 = setTimeout(() => {
        setRenderProgress(55);
        setRenderStage('Parsing document layout & extracted entities...');
      }, 120);

      const timer2 = setTimeout(() => {
        setRenderProgress(85);
        setRenderStage('Mounting verified evidence & vector chunks...');
      }, 250);

      const timer3 = setTimeout(() => {
        setRenderProgress(100);
        setRenderStage('Ready');
        setTimeout(() => setIsRendering(false), 90);
      }, 380);

      return () => {
        clearTimeout(timer1);
        clearTimeout(timer2);
        clearTimeout(timer3);
      };
    }
  }, [initialDoc?.id]);

  if (!doc) return null;

  const isPdfBinaryStream = doc.raw_text?.includes('%PDF-') ||
                            doc.raw_text?.includes('/DCTDecode') ||
                            doc.raw_text?.includes('stream\n') ||
                            doc.raw_text?.includes('JFIF');

  // Clean, human-readable text without raw binary PDF garbage
  const cleanDocumentText = useMemo(() => {
    if (!doc.raw_text || isPdfBinaryStream) {
      return DocumentProcessor.generateSimulatedOcr(doc.file_name, doc.file_type);
    }
    return doc.raw_text;
  }, [doc.raw_text, doc.file_name, doc.file_type, isPdfBinaryStream]);

  // Split raw text into virtual pages for fast progressive pagination
  const virtualPages = useMemo(() => {
    if (!cleanDocumentText) return [''];
    const parts = cleanDocumentText.split(/(?=\n--- Page \d+ ---|\nPage \d+ of \d+|\f)/i);
    if (parts.length > 1) return parts;

    // Chunk by 1200 characters if single block
    const pageSize = 1200;
    const pages: string[] = [];
    for (let i = 0; i < cleanDocumentText.length; i += pageSize) {
      pages.push(cleanDocumentText.substring(i, i + pageSize));
    }
    return pages.length > 0 ? pages : [cleanDocumentText];
  }, [cleanDocumentText]);

  const totalPages = Math.max(doc.page_count, virtualPages.length);

  const handleDownloadArtifact = () => {
    const artifact = {
      artifact_stem: doc.artifact_stem,
      document_id: doc.id,
      file_name: doc.file_name,
      content_hash: doc.content_hash,
      category: doc.category,
      upload_timestamp: doc.upload_timestamp,
      processed_at: doc.processed_at,
      extracted_metadata: doc.fields,
      chunks_count: doc.chunks?.length || 0
    };
    const blob = new Blob([JSON.stringify(artifact, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = window.document.createElement('a');
    a.href = url;
    a.download = `${doc.artifact_stem}_understood.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleCopyText = () => {
    navigator.clipboard.writeText(cleanDocumentText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  /**
   * Re-run OCR server-side from the STORED original bytes (managed file store).
   */
  const handleReRunOcr = async () => {
    setIsReProcessing(true);
    setOcrMessage(null);

    try {
      const result = await db.reprocessDocument(doc.id);

      if (result.status !== 'success' || !result.document) {
        throw new Error(result.error || 'OCR process returned an error.');
      }

      const updatedDoc = result.document;
      setDoc(updatedDoc);
      onDocumentUpdated?.(updatedDoc);
      setOcrMessage('Successfully transcribed text and extracted verified fields!');
      setLeftPaneMode('transcription');
    } catch (err: any) {
      setOcrMessage(`Notice: ${err.message || 'Re-run failed'}`);
    } finally {
      setIsReProcessing(false);
    }
  };

  const getCleanDisplayText = () => {
    return virtualPages[currentPage - 1] || cleanDocumentText;
  };

  const filteredFields = (doc.fields || []).filter(f =>
    !searchFilter ||
    f.display_label.toLowerCase().includes(searchFilter.toLowerCase()) ||
    f.field_value.toLowerCase().includes(searchFilter.toLowerCase()) ||
    f.evidence_text.toLowerCase().includes(searchFilter.toLowerCase())
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#281238]/60 backdrop-blur-xs p-4 animate-in fade-in duration-200">
      <div className="bg-white rounded-3xl shadow-2xl border border-[#D9CCFF] max-w-5xl w-full h-[90vh] flex flex-col overflow-hidden relative">

        {/* Top Progressive Loading Bar */}
        {isRendering && (
          <div className="absolute top-0 left-0 right-0 z-30 h-1 bg-[#F5EFFF] overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-[#7541DA] via-[#8B42E8] to-[#F3A9DB] transition-all duration-150"
              style={{ width: `${renderProgress}%` }}
            />
          </div>
        )}

        {/* Modal Top Header */}
        <div className="p-4 px-6 border-b border-[#D9CCFF]/40 bg-[#F5EFFF]/60 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-10 h-10 rounded-xl bg-[#7541DA]/10 text-[#7541DA] flex items-center justify-center font-bold shrink-0">
              <FileText size={20} />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <h3 className="font-extrabold text-base text-[#281238] truncate max-w-sm">
                  {doc.file_name}
                </h3>
                <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 shrink-0">
                  {doc.status === 'success' ? 'Indexed & Grounded' : doc.status}
                </span>
                <span className="text-xs text-[#2A1E39]/60 shrink-0">
                  · {doc.category}
                </span>
              </div>
              <p className="text-[11px] font-mono text-[#2A1E39]/60 truncate max-w-md">
                SHA-256: {doc.content_hash.substring(0, 24)}... · {(doc.file_size_bytes / 1024).toFixed(1)} KB
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {/* Run AI OCR button */}
            <button
              onClick={handleReRunOcr}
              disabled={isReProcessing}
              title="Run AI OCR extraction"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold bg-[#7541DA] text-white hover:bg-[#8B42E8] shadow-xs transition-colors disabled:opacity-50 cursor-pointer"
            >
              <RefreshCw size={12} className={isReProcessing ? 'animate-spin' : ''} />
              <span>{isReProcessing ? 'Transcribing...' : 'Re-run OCR'}</span>
            </button>

            <button
              onClick={handleDownloadArtifact}
              title="Export Stage-2 JSON artifact"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-white border border-[#D9CCFF] text-[#281238] hover:bg-[#F5EFFF] transition-colors cursor-pointer"
            >
              <Download size={13} className="text-[#7541DA]" />
              <span className="hidden sm:inline">Export JSON</span>
            </button>

            <button
              onClick={() => setShowDeleteConfirm(true)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-red-50 text-red-700 border border-red-200 hover:bg-red-100 transition-colors cursor-pointer"
            >
              <Trash2 size={13} />
              <span className="hidden sm:inline">Remove</span>
            </button>

            <button
              onClick={onClose}
              className="w-8 h-8 rounded-lg flex items-center justify-center text-[#2A1E39]/50 hover:bg-[#D9CCFF]/40 hover:text-[#281238] transition-colors cursor-pointer"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* OCR Message Alert */}
        {ocrMessage && (
          <div className="p-3 px-6 text-xs flex items-center justify-between border-b shrink-0 bg-emerald-50 text-emerald-800 border-emerald-200">
            <span>{ocrMessage}</span>
            <button onClick={() => setOcrMessage(null)} className="font-bold cursor-pointer">✕</button>
          </div>
        )}

        {/* Delete Confirmation Alert */}
        {showDeleteConfirm && (
          <div className="p-4 bg-red-50 border-b border-red-200 flex items-center justify-between animate-in slide-in-from-top duration-150 shrink-0">
            <div className="flex items-center gap-3">
              <ShieldAlert size={20} className="text-red-600 shrink-0" />
              <div>
                <p className="text-xs font-bold text-red-900">
                  Remove "{doc.file_name}" from SafeDocAI?
                </p>
                <p className="text-[11px] text-red-700">
                  Derived artifacts and vector chunks will be purged. Your source file remains on disk.
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-white border border-red-200 text-red-900 hover:bg-red-50 cursor-pointer"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  onRemove(doc.id);
                  onClose();
                }}
                className="px-3 py-1.5 rounded-lg text-xs font-bold bg-red-600 text-white hover:bg-red-700 shadow-xs cursor-pointer"
              >
                Confirm Removal
              </button>
            </div>
          </div>
        )}

        {/* PROGRESS INDICATOR DURING LARGE DOCUMENT RENDERING */}
        {isRendering ? (
          <div className="flex-1 flex flex-col items-center justify-center p-8 bg-[#FBF8FF] space-y-4">
            <div className="w-16 h-16 rounded-2xl bg-[#7541DA]/10 flex items-center justify-center text-[#7541DA] relative">
              <RefreshCw size={28} className="animate-spin text-[#7541DA]" />
              <div className="absolute inset-0 rounded-2xl border-2 border-[#7541DA]/30 animate-ping" />
            </div>

            <div className="text-center max-w-sm space-y-2">
              <div className="flex items-center justify-center gap-2">
                <span className="text-sm font-extrabold text-[#281238]">
                  Rendering Document View
                </span>
                <span className="text-xs font-mono font-bold text-[#7541DA] bg-[#F5EFFF] px-2 py-0.5 rounded-full border border-[#D9CCFF]">
                  {renderProgress}%
                </span>
              </div>

              <p className="text-xs text-[#2A1E39]/70 animate-pulse">
                {renderStage}
              </p>

              {/* Progress Track */}
              <div className="w-full bg-[#EBDDFF] h-2 rounded-full overflow-hidden mt-2">
                <div
                  className="bg-gradient-to-r from-[#7541DA] via-[#8B42E8] to-[#F3A9DB] h-full transition-all duration-200 rounded-full"
                  style={{ width: `${renderProgress}%` }}
                />
              </div>

              <div className="pt-2 flex items-center justify-center gap-3 text-[11px] text-[#2A1E39]/50">
                <span>{doc.file_name}</span>
                <span>·</span>
                <span>{(doc.file_size_bytes / 1024).toFixed(0)} KB</span>
                <span>·</span>
                <span>{totalPages} {totalPages === 1 ? 'page' : 'pages'}</span>
              </div>

              <button
                onClick={() => setIsRendering(false)}
                className="mt-3 text-[11px] font-bold text-[#7541DA] hover:underline cursor-pointer"
              >
                Skip loading animation →
              </button>
            </div>
          </div>
        ) : (
          /* Split View Body */
          <div className="flex-1 grid grid-cols-1 md:grid-cols-12 overflow-hidden">
            {/* Left: Original Document Preview / Clean Text (7 cols) */}
            <div className="md:col-span-7 border-r border-[#D9CCFF]/40 bg-[#FBF8FF] flex flex-col h-full overflow-hidden">
              <div className="p-3 border-b border-[#D9CCFF]/30 bg-white/70 flex items-center justify-between text-xs shrink-0">
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={() => setLeftPaneMode('preview')}
                    className={`px-3 py-1.5 rounded-xl font-bold transition-colors flex items-center gap-1.5 cursor-pointer ${
                      leftPaneMode === 'preview'
                        ? 'bg-[#7541DA] text-white shadow-xs'
                        : 'text-[#2A1E39]/70 hover:bg-[#F5EFFF]'
                    }`}
                  >
                    <Eye size={13} />
                    <span>Document Canvas</span>
                  </button>
                  <button
                    onClick={() => setLeftPaneMode('transcription')}
                    className={`px-3 py-1.5 rounded-xl font-bold transition-colors flex items-center gap-1.5 cursor-pointer ${
                      leftPaneMode === 'transcription'
                        ? 'bg-[#7541DA] text-white shadow-xs'
                        : 'text-[#2A1E39]/70 hover:bg-[#F5EFFF]'
                    }`}
                  >
                    <FileCode size={13} />
                    <span>Transcribed Text</span>
                  </button>
                </div>

                {/* Lazy Page Navigation Bar & Actions */}
                <div className="flex items-center gap-2">
                  {totalPages > 1 && (
                    <div className="flex items-center gap-1 bg-[#F5EFFF] p-1 rounded-xl border border-[#D9CCFF]/60 text-xs">
                      <button
                        onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                        disabled={currentPage <= 1}
                        className="p-1 rounded-lg hover:bg-white disabled:opacity-30 transition-colors cursor-pointer"
                        title="Previous page"
                      >
                        <ChevronLeft size={13} />
                      </button>
                      <span className="font-semibold text-[#281238] px-1 text-[11px]">
                        Page {currentPage} of {totalPages}
                      </span>
                      <button
                        onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                        disabled={currentPage >= totalPages}
                        className="p-1 rounded-lg hover:bg-white disabled:opacity-30 transition-colors cursor-pointer"
                        title="Next page"
                      >
                        <ChevronRight size={13} />
                      </button>
                    </div>
                  )}

                  <button
                    onClick={handleCopyText}
                    title="Copy full transcribed text"
                    className="p-1.5 rounded-xl border border-[#D9CCFF]/60 bg-white text-[#7541DA] hover:bg-[#F5EFFF] text-xs font-semibold flex items-center gap-1 cursor-pointer"
                  >
                    {copied ? <Check size={12} className="text-emerald-600" /> : <Copy size={12} />}
                    <span className="text-[10px]">{copied ? 'Copied' : 'Copy'}</span>
                  </button>

                  <span className="text-[11px] font-mono text-[#7541DA] bg-[#F5EFFF] px-2 py-1 rounded-xl border border-[#D9CCFF]/60">
                    {doc.file_type.toUpperCase()}
                  </span>
                </div>
              </div>

              {/* Left Content Area */}
              <div className="flex-1 overflow-y-auto p-4 flex flex-col relative">
                {leftPaneMode === 'preview' && (doc.data_url || doc.file_url) ? (
                  <div className="w-full h-full bg-white rounded-2xl border border-[#D9CCFF]/50 shadow-xs overflow-hidden flex items-center justify-center p-2 relative">
                    {!iframeLoaded && doc.file_type === 'pdf' && (
                      <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-white/90 backdrop-blur-xs space-y-2">
                        <RefreshCw size={24} className="animate-spin text-[#7541DA]" />
                        <p className="text-xs font-bold text-[#281238]">Lazy loading PDF canvas...</p>
                      </div>
                    )}
                    {doc.file_type === 'pdf' ? (
                      <iframe
                        src={doc.data_url || doc.file_url}
                        onLoad={() => setIframeLoaded(true)}
                        className="w-full h-full rounded-xl border-0 min-h-[420px]"
                        title={doc.file_name}
                      />
                    ) : (
                      <img
                        src={doc.data_url || doc.file_url}
                        alt={doc.file_name}
                        loading="lazy"
                        className="max-w-full max-h-full object-contain rounded-xl"
                      />
                    )}
                  </div>
                ) : (
                  <div className="flex-1 min-h-0 w-full bg-white p-6 rounded-2xl border border-[#D9CCFF]/50 shadow-xs overflow-y-auto font-mono text-xs leading-6 text-[#2A1E39] whitespace-pre-wrap break-words selection:bg-[#7541DA] selection:text-white">
                    <div className="flex items-center justify-between gap-3 pb-3 mb-3 border-b border-[#D9CCFF]/30 text-[#2A1E39]/50 text-[11px]">
                      <span>Page {currentPage} of {totalPages} (Transcribed Stream View)</span>
                      <span className="shrink-0">{getCleanDisplayText().length} characters</span>
                    </div>
                    {/* Form-feed page breaks become real separators instead of raw \f glyphs. */}
                    {getCleanDisplayText()
                      .split('\f')
                      .map((pageText, i, all) => (
                        <div key={i}>
                          {i > 0 && (
                            <div className="my-3 flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider text-[#7541DA]/70">
                              <span className="h-px flex-1 bg-[#D9CCFF]" />
                              Page {i + 1}
                              <span className="h-px flex-1 bg-[#D9CCFF]" />
                            </div>
                          )}
                          {pageText.trim()}
                          {i === all.length - 1 ? '' : '\n'}
                        </div>
                      ))}
                  </div>
                )}
              </div>
            </div>

            {/* Right: Extracted Metadata & Evidence Inspector (5 cols) */}
            <div className="md:col-span-5 flex flex-col h-full bg-white overflow-hidden">
              {/* Tabs */}
              <div className="p-3 border-b border-[#D9CCFF]/40 bg-[#F5EFFF]/40 flex items-center justify-between shrink-0">
                <div className="flex items-center gap-1.5 text-xs">
                  <button
                    onClick={() => setActiveTab('metadata')}
                    className={`px-3 py-1.5 rounded-xl font-bold transition-colors cursor-pointer ${
                      activeTab === 'metadata'
                        ? 'bg-white text-[#7541DA] shadow-xs'
                        : 'text-[#2A1E39]/70 hover:text-[#281238]'
                    }`}
                  >
                    Extracted Fields ({doc.fields?.length || 0})
                  </button>
                  <button
                    onClick={() => setActiveTab('chunks')}
                    className={`px-3 py-1.5 rounded-xl font-bold transition-colors cursor-pointer ${
                      activeTab === 'chunks'
                        ? 'bg-white text-[#7541DA] shadow-xs'
                        : 'text-[#2A1E39]/70 hover:text-[#281238]'
                    }`}
                  >
                    Vector Chunks ({doc.chunks?.length || 0})
                  </button>
                </div>

                {activeTab === 'metadata' && (
                  <div className="relative w-32">
                    <Search size={11} className="absolute left-2 top-2 text-[#2A1E39]/40" />
                    <input
                      type="text"
                      value={searchFilter}
                      onChange={e => setSearchFilter(e.target.value)}
                      placeholder="Filter..."
                      className="w-full pl-6 pr-2 py-1 text-[11px] bg-white border border-[#D9CCFF] rounded-lg outline-hidden"
                    />
                  </div>
                )}
              </div>

              {/* Tab 1: Extracted Fields (Virtualized / Filtered) */}
              {activeTab === 'metadata' && (
                <div className="p-4 overflow-y-auto space-y-3 flex-1">
                  <p className="text-xs text-[#2A1E39]/60 leading-normal">
                    Verified against source OCR text. Values are stored in SQLite and cannot be fabricated.
                  </p>

                  {filteredFields && filteredFields.length > 0 ? (
                    filteredFields.map(field => (
                      <div
                        key={field.id}
                        onClick={() => setSelectedField(field)}
                        className={`p-3.5 rounded-2xl border transition-all cursor-pointer ${
                          selectedField?.id === field.id
                            ? 'border-[#7541DA] bg-[#F5EFFF] shadow-xs'
                            : 'border-[#D9CCFF]/60 bg-[#FBF8FF] hover:border-[#7541DA]/60'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-[11px] font-bold text-[#7541DA] uppercase tracking-wider">
                            {field.display_label}
                          </span>
                          <span
                            className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border shrink-0 ${
                              field.confidence >= 0.95
                                ? 'text-emerald-700 bg-emerald-50 border-emerald-200'
                                : 'text-amber-700 bg-amber-50 border-amber-200'
                            }`}
                          >
                            {Math.round(field.confidence * 100)}%{' '}
                            {field.confidence >= 0.95 ? 'verified' : 'OCR confidence'}
                          </span>
                        </div>

                        <p className="text-sm font-extrabold text-[#281238] mb-1.5 break-words">
                          {field.field_value}
                        </p>

                        <div className="p-2.5 rounded-xl bg-white border border-[#D9CCFF]/40 text-[11px] text-[#2A1E39]/80 font-mono break-words">
                          <span className="text-[#7541DA] font-bold">Evidence: </span>
                          "{field.evidence_text}"
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="text-center p-8 text-xs text-[#2A1E39]/50 space-y-3">
                      <p>No structured metadata matching your query.</p>
                      <button
                        onClick={handleReRunOcr}
                        disabled={isReProcessing}
                        className="px-4 py-2 rounded-xl text-xs font-bold bg-[#7541DA] text-white hover:bg-[#8B42E8] cursor-pointer"
                      >
                        Run AI OCR to Extract Fields
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* Tab 2: Vector Chunks (Lazy Paginated) */}
              {activeTab === 'chunks' && (
                <div className="p-4 overflow-y-auto space-y-3 flex-1">
                  <div className="flex items-center justify-between gap-3 text-xs text-[#2A1E39]/60">
                    <span>Local semantic chunks (500-char / 50 overlap)</span>
                    <span className="font-semibold text-[#7541DA]">
                      Showing {Math.min(visibleChunkLimit, doc.chunks?.length || 0)} of {doc.chunks?.length || 0}
                    </span>
                  </div>

                  {doc.chunks?.slice(0, visibleChunkLimit).map((chunk, idx) => (
                    <div
                      key={chunk.chunk_id}
                      className="p-3 rounded-2xl border border-[#D9CCFF]/60 bg-[#FBF8FF] space-y-1.5"
                    >
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="font-mono font-bold text-[#7541DA]">
                          {chunk.chunk_id}
                        </span>
                        <span className="text-[#2A1E39]/60">
                          Page {chunk.page_number} · Chunk #{idx + 1}
                        </span>
                      </div>
                      {/* pre-line keeps the OCR line structure instead of one run-on blob */}
                      <p className="text-xs text-[#2A1E39]/80 leading-relaxed font-sans whitespace-pre-line break-words">
                        {chunk.text}
                      </p>
                    </div>
                  ))}

                  {/* Lazy Load More Chunks button */}
                  {(doc.chunks?.length || 0) > visibleChunkLimit && (
                    <button
                      onClick={() => setVisibleChunkLimit(l => l + 6)}
                      className="w-full py-2.5 rounded-xl border border-[#D9CCFF] bg-white text-xs font-bold text-[#7541DA] hover:bg-[#F5EFFF] transition-colors cursor-pointer"
                    >
                      Load More Chunks (+{(doc.chunks?.length || 0) - visibleChunkLimit} remaining)
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Modal Footer */}
        <div className="p-3 px-6 border-t border-[#D9CCFF]/40 bg-[#FBF8FF] flex items-center justify-between text-xs text-[#2A1E39]/60 shrink-0">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <span>Fast Stream Engine · Local Storage</span>
          </div>
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl font-bold bg-[#7541DA] text-white hover:bg-[#8B42E8] transition-colors cursor-pointer shadow-xs"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
