import React, { useState, useRef } from 'react';
import {
  Upload,
  FileText,
  Search,
  Trash2,
  Eye,
  CheckCircle2,
  AlertCircle,
  Clock,
  X,
  Download,
  Plus,
  RefreshCw,
  FolderArchive,
  Layers,
  Sparkles
} from 'lucide-react';
import { StoredDocument, IngestionJob } from '../types';
import { DocumentProcessor } from '../utils/documentProcessor';
import { db } from '../utils/storage';
import { MascotRobot } from './MascotArt';

interface Props {
  documents: StoredDocument[];
  onRefreshDocs: () => void | Promise<void>;
  onOpenViewer: (doc: StoredDocument) => void;
}

export const DocumentsView: React.FC<Props> = ({
  documents,
  onRefreshDocs,
  onOpenViewer
}) => {
  const [searchFilter, setSearchFilter] = useState('');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [activeJob, setActiveJob] = useState<IngestionJob | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const isCancelledRef = useRef<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const filteredDocs = documents.filter(doc => {
    const matchesSearch = doc.file_name.toLowerCase().includes(searchFilter.toLowerCase()) ||
                          doc.category.toLowerCase().includes(searchFilter.toLowerCase());
    const matchesCategory = categoryFilter === 'all' || doc.category === categoryFilter;
    return matchesSearch && matchesCategory;
  });

  const categories = Array.from(new Set(documents.map(d => d.category)));

  const handleFileUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    setUploadError(null);
    setUploadSuccess(null);
    isCancelledRef.current = false;

    const newJob: IngestionJob = {
      job_id: `job_${Date.now()}`,
      file_name: file.name,
      file_size: file.size,
      stage: 'uploading',
      progress: 5
    };
    setActiveJob(newJob);

    try {
      const res = await DocumentProcessor.processDocument(file, {
        onProgress: (stage, progress) => {
          setActiveJob(prev => prev ? { ...prev, stage: stage as any, progress } : null);
        },
        isCancelled: () => isCancelledRef.current
      });

      if (res.status === 'cancelled') {
        setActiveJob(null);
        setUploadError('Document upload was cancelled. Staging files cleared.');
      } else if (res.status === 'duplicate') {
        setActiveJob(null);
        setUploadError(res.error || 'Exact duplicate detected.');
        if (res.document) {
          onOpenViewer(res.document);
        }
      } else if (res.status === 'failed') {
        setActiveJob(null);
        setUploadError(res.error || 'Ingestion failed during processing.');
      } else if (res.status === 'success' && res.document) {
        setActiveJob(null);
        setUploadSuccess(`Successfully indexed "${res.document.file_name}" with ${res.document.fields.length} verified fields.`);
        onRefreshDocs();
      }
    } catch (err: any) {
      setActiveJob(null);
      setUploadError(err.message || 'Error occurred during ingestion.');
    }
  };

  const handleCancelJob = () => {
    isCancelledRef.current = true;
    setActiveJob(null);
    setUploadError('Ingestion cancelled by user.');
  };

  const handleDeleteDocument = async (id: number) => {
    const res = await db.removeDocument(id);
    if (res.success) {
      onRefreshDocs();
    }
  };

  return (
    <div className="space-y-6 select-none">
      {/* Upload Dropzone */}
      <div className="bg-white rounded-3xl border-2 border-dashed border-[#D9CCFF] p-8 text-center hover:border-[#7541DA] transition-all relative overflow-hidden group">
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.jpg,.jpeg,.png"
          onChange={e => handleFileUpload(e.target.files)}
          className="hidden"
        />

        <div className="max-w-md mx-auto space-y-3">
          <div className="w-14 h-14 rounded-2xl bg-[#F5EFFF] text-[#7541DA] flex items-center justify-center mx-auto group-hover:scale-110 transition-transform">
            <Upload size={28} />
          </div>

          <h3 className="font-black text-lg text-[#281238]">
            Drag & Drop your documents here or click to browse
          </h3>

          <p className="text-xs text-[#2A1E39]/60">
            Supported formats: <strong className="text-[#7541DA]">PDF, JPG, JPEG, PNG</strong> (Max 50 MB)
          </p>

          <div className="flex items-center justify-center gap-3 pt-2">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-xs font-bold bg-[#7541DA] text-white hover:bg-[#8B42E8] shadow-sm shadow-[#7541DA]/20 transition-all cursor-pointer"
            >
              <Plus size={15} />
              <span>Select File</span>
            </button>
          </div>
        </div>
      </div>

      {/* Active Ingestion Progress Card */}
      {activeJob && (
        <div className="p-4 bg-white rounded-2xl border border-[#7541DA] shadow-md space-y-2 animate-in fade-in duration-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <RefreshCw size={16} className="text-[#7541DA] animate-spin" />
              <div>
                <p className="text-xs font-bold text-[#281238]">
                  {activeJob.stage}: <span className="font-normal">{activeJob.file_name}</span>
                </p>
                <p className="text-[10px] text-[#2A1E39]/60">
                  Managed store ingestion · Evidence grounding check
                </p>
              </div>
            </div>
            <button
              onClick={handleCancelJob}
              className="px-2.5 py-1 rounded-lg text-xs font-bold bg-red-50 text-red-700 border border-red-200 hover:bg-red-100"
            >
              Cancel
            </button>
          </div>

          <div className="w-full bg-[#F5EFFF] h-2.5 rounded-full overflow-hidden">
            <div
              className="bg-gradient-to-r from-[#7541DA] to-[#8B42E8] h-full transition-all duration-200"
              style={{ width: `${activeJob.progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Alerts */}
      {uploadError && (
        <div className="p-3.5 rounded-2xl bg-amber-50 border border-amber-200 text-xs text-amber-900 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertCircle size={16} className="text-amber-600 shrink-0" />
            <span>{uploadError}</span>
          </div>
          <button onClick={() => setUploadError(null)} className="text-amber-600 hover:text-amber-900">
            <X size={14} />
          </button>
        </div>
      )}

      {uploadSuccess && (
        <div className="p-3.5 rounded-2xl bg-emerald-50 border border-emerald-200 text-xs text-emerald-900 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CheckCircle2 size={16} className="text-emerald-600 shrink-0" />
            <span>{uploadSuccess}</span>
          </div>
          <button onClick={() => setUploadSuccess(null)} className="text-emerald-600 hover:text-emerald-900">
            <X size={14} />
          </button>
        </div>
      )}

      {/* Document Library Filter Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 bg-white p-4 rounded-2xl border border-[#D9CCFF]/50 shadow-xs">
        <div className="relative w-full sm:w-72">
          <Search size={16} className="absolute left-3 top-2.5 text-[#2A1E39]/40" />
          <input
            type="text"
            value={searchFilter}
            onChange={e => setSearchFilter(e.target.value)}
            placeholder="Search by filename or category..."
            className="w-full pl-9 pr-3 py-1.5 text-xs bg-[#FBF8FF] border border-[#D9CCFF] rounded-xl text-[#281238] outline-hidden focus:border-[#7541DA]"
          />
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          <span className="text-xs text-[#2A1E39]/60 shrink-0">Category:</span>
          <select
            value={categoryFilter}
            onChange={e => setCategoryFilter(e.target.value)}
            className="text-xs bg-[#FBF8FF] border border-[#D9CCFF] text-[#281238] font-semibold rounded-xl px-3 py-1.5 outline-hidden"
          >
            <option value="all">All Categories ({documents.length})</option>
            {categories.map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Documents Grid */}
      {filteredDocs.length === 0 ? (
        <div className="bg-white rounded-3xl border border-[#D9CCFF]/60 p-12 text-center flex flex-col items-center">
          <MascotRobot size={90} mood="search" className="mb-2" />
          <h4 className="font-extrabold text-base text-[#281238]">
            No matching documents found
          </h4>
          <p className="text-xs text-[#2A1E39]/60 mt-1 mb-4">
            Try adjusting your search filter or upload a new file.
          </p>
          <button
            onClick={() => { setSearchFilter(''); setCategoryFilter('all'); }}
            className="text-xs font-bold text-[#7541DA] hover:underline"
          >
            Clear Filters
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredDocs.map(doc => (
            <div
              key={doc.id}
              onClick={() => onOpenViewer(doc)}
              className="bg-white rounded-2xl border border-[#D9CCFF]/70 hover:border-[#7541DA] p-5 shadow-xs hover:shadow-md transition-all cursor-pointer flex flex-col justify-between group"
            >
              <div>
                <div className="flex items-start justify-between gap-2 mb-3">
                  <div className="w-10 h-10 rounded-xl bg-[#F5EFFF] text-[#7541DA] flex items-center justify-center shrink-0 border border-[#D9CCFF]/50 group-hover:bg-[#7541DA] group-hover:text-white transition-colors">
                    <FileText size={20} />
                  </div>
                  <span className="text-[10px] font-semibold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-200 flex items-center gap-1">
                    <CheckCircle2 size={10} /> Indexed
                  </span>
                </div>

                <h4 className="font-bold text-sm text-[#281238] truncate group-hover:text-[#7541DA] transition-colors mb-1">
                  {doc.file_name}
                </h4>

                <p className="text-xs text-[#2A1E39]/60 mb-3">
                  {doc.category}
                </p>

                {/* Extracted Fields Preview */}
                <div className="space-y-1 mb-3">
                  {doc.fields?.slice(0, 2).map(f => (
                    <div key={f.id} className="text-[11px] flex items-center justify-between bg-[#FBF8FF] px-2 py-1 rounded-lg border border-[#D9CCFF]/40">
                      <span className="text-[#2A1E39]/60">{f.display_label}:</span>
                      <span className="font-bold text-[#281238] truncate max-w-[120px]">{f.field_value}</span>
                    </div>
                  ))}
                  {(doc.fields?.length || 0) > 2 && (
                    <p className="text-[10px] text-[#7541DA] font-semibold text-right">
                      +{(doc.fields?.length || 0) - 2} more fields
                    </p>
                  )}
                </div>
              </div>

              {/* Card Footer */}
              <div className="pt-3 border-t border-[#D9CCFF]/40 flex items-center justify-between text-[11px] text-[#2A1E39]/50">
                <span>{(doc.file_size_bytes / 1024).toFixed(0)} KB · {doc.page_count} pg</span>
                <div className="flex items-center gap-1">
                  <button
                    onClick={e => {
                      e.stopPropagation();
                      onOpenViewer(doc);
                    }}
                    className="p-1.5 rounded-lg text-[#7541DA] hover:bg-[#F5EFFF]"
                    title="Inspect document & evidence"
                  >
                    <Eye size={15} />
                  </button>
                  <button
                    onClick={e => {
                      e.stopPropagation();
                      handleDeleteDocument(doc.id);
                    }}
                    className="p-1.5 rounded-lg text-[#2A1E39]/40 hover:text-red-600 hover:bg-red-50"
                    title="Remove from SafeDocAI"
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
