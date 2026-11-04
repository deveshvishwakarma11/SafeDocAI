import React, { useState } from 'react';
import {
  Search,
  ArrowRight,
  Upload,
  MessageSquare,
  FolderArchive,
  Sparkles,
  FileText,
  CheckCircle2,
  Clock,
  Eye,
  MoreVertical,
  Plus,
  RefreshCw
} from 'lucide-react';
import { StoredDocument } from '../types';
import { MascotRobot } from './MascotArt';
import { NavTab } from './Sidebar';

interface Props {
  documents: StoredDocument[];
  onSelectTab: (tab: NavTab) => void;
  onOpenViewer: (doc: StoredDocument) => void;
  onSubmitQuery: (query: string) => void;
}

export const DashboardView: React.FC<Props> = ({
  documents,
  onSelectTab,
  onOpenViewer,
  onSubmitQuery
}) => {
  const [searchQuery, setSearchQuery] = useState('');

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    onSubmitQuery(searchQuery.trim());
  };

  // Generate deterministic suggestions strictly from CURRENT STORED DOCUMENTS as mandated by PRD Section 6.2 & TRD 11.3!
  const generateDynamicSuggestions = () => {
    const suggestions: string[] = [];
    if (documents.length === 0) return [];

    for (const doc of documents) {
      if (doc.status !== 'success') continue;

      const fileName = doc.file_name;
      const lower = fileName.toLowerCase();

      if (lower.includes('marksheet')) {
        suggestions.push(`What is my roll number in ${fileName}?`);
        suggestions.push(`What is my CGPA or marks in ${fileName}?`);
      } else if (lower.includes('bill')) {
        suggestions.push(`What is the total due amount for ${fileName}?`);
        suggestions.push(`What is the due date on ${fileName}?`);
      } else if (lower.includes('passbook') || lower.includes('statement')) {
        suggestions.push(`What is my account number in ${fileName}?`);
        suggestions.push(`What is the available balance in ${fileName}?`);
      } else if (lower.includes('aadhaar') || lower.includes('id')) {
        suggestions.push(`What is my date of birth in ${fileName}?`);
        suggestions.push(`Show my address in ${fileName}`);
      } else if (lower.includes('policy') || lower.includes('insurance')) {
        suggestions.push(`What is my sum insured in ${fileName}?`);
        suggestions.push(`When does my ${fileName} expire?`);
      } else if (doc.fields && doc.fields.length > 0) {
        const topField = doc.fields[0];
        suggestions.push(`What is the ${topField.display_label} in ${fileName}?`);
      }
    }

    // Limit to 4 relevant prompts
    return suggestions.slice(0, 4);
  };

  const dynamicSuggestions = generateDynamicSuggestions();

  return (
    <div className="space-y-8 select-none">
      {/* Search / Ask Composer */}
      <div className="relative">
        <form onSubmit={handleSearchSubmit} className="relative">
          <div className="relative flex items-center bg-white rounded-2xl shadow-md border border-[#D9CCFF] p-2 pr-3 hover:border-[#7541DA] transition-all">
            <div className="pl-3 pr-2 text-[#7541DA]">
              <Search size={20} />
            </div>
            <input
              type="text"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="Ask anything about your documents in English, Hindi, or Hinglish..."
              className="w-full bg-transparent text-sm sm:text-base text-[#281238] placeholder-[#2A1E39]/40 outline-hidden py-2"
            />
            <button
              type="submit"
              disabled={!searchQuery.trim()}
              className="inline-flex items-center justify-center w-10 h-10 rounded-xl bg-[#7541DA] text-white hover:bg-[#8B42E8] disabled:opacity-30 disabled:hover:bg-[#7541DA] transition-all cursor-pointer shrink-0"
            >
              <ArrowRight size={18} />
            </button>
          </div>
        </form>

        {/* Dynamic Inventory-driven Suggestions */}
        {documents.length > 0 && dynamicSuggestions.length > 0 && (
          <div className="mt-3 flex items-center gap-2 flex-wrap">
            <span className="text-xs font-bold text-[#7541DA] flex items-center gap-1">
              <Sparkles size={12} /> Suggestions:
            </span>
            {dynamicSuggestions.map((prompt, i) => (
              <button
                key={i}
                onClick={() => onSubmitQuery(prompt)}
                className="text-xs px-3 py-1.5 rounded-lg bg-[#F5EFFF] text-[#281238] border border-[#D9CCFF] hover:bg-[#EAE0FF] hover:border-[#7541DA] transition-colors"
              >
                {prompt}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Quick Actions Bar */}
      <div>
        <p className="text-xs font-bold uppercase tracking-wider text-[#2A1E39]/50 mb-3">
          Quick Actions
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5">
          <div
            onClick={() => onSelectTab('upload')}
            className="p-4 rounded-2xl bg-white border border-[#D9CCFF]/60 hover:border-[#7541DA] shadow-xs hover:shadow-md transition-all cursor-pointer group"
          >
            <div className="w-10 h-10 rounded-xl bg-[#7541DA]/10 text-[#7541DA] flex items-center justify-center mb-3 group-hover:scale-105 transition-transform">
              <Upload size={20} />
            </div>
            <h4 className="font-extrabold text-sm text-[#281238] group-hover:text-[#7541DA] transition-colors">
              Upload Document
            </h4>
            <p className="text-xs text-[#2A1E39]/60 mt-0.5">
              PDF, JPG, JPEG, PNG
            </p>
          </div>

          <div
            onClick={() => onSelectTab('chat')}
            className="p-4 rounded-2xl bg-white border border-[#D9CCFF]/60 hover:border-[#7541DA] shadow-xs hover:shadow-md transition-all cursor-pointer group"
          >
            <div className="w-10 h-10 rounded-xl bg-[#8B42E8]/10 text-[#8B42E8] flex items-center justify-center mb-3 group-hover:scale-105 transition-transform">
              <MessageSquare size={20} />
            </div>
            <h4 className="font-extrabold text-sm text-[#281238] group-hover:text-[#7541DA] transition-colors">
              Ask a Question
            </h4>
            <p className="text-xs text-[#2A1E39]/60 mt-0.5">
              Get instant answers
            </p>
          </div>

          <div
            onClick={() => onSelectTab('documents')}
            className="p-4 rounded-2xl bg-white border border-[#D9CCFF]/60 hover:border-[#7541DA] shadow-xs hover:shadow-md transition-all cursor-pointer group"
          >
            <div className="w-10 h-10 rounded-xl bg-[#B768F2]/10 text-[#B768F2] flex items-center justify-center mb-3 group-hover:scale-105 transition-transform">
              <FolderArchive size={20} />
            </div>
            <h4 className="font-extrabold text-sm text-[#281238] group-hover:text-[#7541DA] transition-colors">
              View Documents
            </h4>
            <p className="text-xs text-[#2A1E39]/60 mt-0.5">
              Browse your {documents.length} files
            </p>
          </div>

          <div
            onClick={() => onSelectTab('documents')}
            className="p-4 rounded-2xl bg-white border border-[#D9CCFF]/60 hover:border-[#7541DA] shadow-xs hover:shadow-md transition-all cursor-pointer group"
          >
            <div className="w-10 h-10 rounded-xl bg-[#F3A9DB]/20 text-[#8B42E8] flex items-center justify-center mb-3 group-hover:scale-105 transition-transform">
              <Sparkles size={20} />
            </div>
            <h4 className="font-extrabold text-sm text-[#281238] group-hover:text-[#7541DA] transition-colors">
              Extract Metadata
            </h4>
            <p className="text-xs text-[#2A1E39]/60 mt-0.5">
              Verified key-value facts
            </p>
          </div>
        </div>
      </div>

      {/* Main Grid: Recent Documents + Helper Card */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Recent Documents Table (8 cols) */}
        <div className="lg:col-span-8 bg-white rounded-2xl border border-[#D9CCFF]/60 shadow-xs p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="font-extrabold text-base text-[#281238]">
                Recent Documents
              </h3>
              <p className="text-xs text-[#2A1E39]/60">
                Managed local records stored under SafeDocAI
              </p>
            </div>
            {documents.length > 0 && (
              <button
                onClick={() => onSelectTab('documents')}
                className="text-xs font-bold text-[#7541DA] hover:underline"
              >
                View All ({documents.length}) →
              </button>
            )}
          </div>

          {documents.length === 0 ? (
            /* Empty State matching PRD Rule: No phantom files! */
            <div className="py-12 px-4 text-center flex flex-col items-center">
              <MascotRobot size={100} mood="search" className="mb-3" />
              <h4 className="text-base font-bold text-[#281238]">
                No Documents Indexed Yet
              </h4>
              <p className="text-xs text-[#2A1E39]/60 max-w-sm mt-1 mb-5">
                Upload your marksheet, bills, receipts, or identity cards. SafeDocAI will parse and extract verified evidence locally.
              </p>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => onSelectTab('upload')}
                  className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-bold bg-[#7541DA] text-white hover:bg-[#8B42E8] transition-colors"
                >
                  <Plus size={14} /> Upload First File
                </button>
              </div>
            </div>
          ) : (
            <div className="divide-y divide-[#D9CCFF]/30">
              {documents.slice(0, 5).map(doc => (
                <div
                  key={doc.id}
                  onClick={() => onOpenViewer(doc)}
                  className="py-3 px-2 flex items-center justify-between hover:bg-[#F5EFFF]/40 rounded-xl transition-colors cursor-pointer group"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <div className="w-9 h-9 rounded-xl bg-[#F5EFFF] text-[#7541DA] flex items-center justify-center shrink-0 border border-[#D9CCFF]/50 group-hover:border-[#7541DA] transition-colors">
                      <FileText size={18} />
                    </div>
                    <div className="min-w-0">
                      <p className="text-sm font-bold text-[#281238] truncate group-hover:text-[#7541DA] transition-colors">
                        {doc.file_name}
                      </p>
                      <p className="text-xs text-[#2A1E39]/50 flex items-center gap-2">
                        <span>{doc.category}</span>
                        <span aria-hidden="true">·</span>
                        <span>{(doc.file_size_bytes / 1024).toFixed(0)} KB</span>
                        <span aria-hidden="true">·</span>
                        <span>{new Date(doc.upload_timestamp).toLocaleDateString()}</span>
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    <span className="text-[11px] font-semibold text-emerald-700 bg-emerald-50 px-2.5 py-0.5 rounded-full border border-emerald-200 flex items-center gap-1">
                      <CheckCircle2 size={11} /> Indexed
                    </span>
                    <button
                      onClick={e => {
                        e.stopPropagation();
                        onOpenViewer(doc);
                      }}
                      className="p-1.5 rounded-lg text-[#2A1E39]/40 hover:text-[#7541DA] hover:bg-[#F5EFFF] transition-colors"
                    >
                      <Eye size={16} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Mascot Helper Card (4 cols) */}
        <div className="lg:col-span-4 bg-gradient-to-b from-[#F5EFFF] to-white rounded-2xl border border-[#D9CCFF] p-6 shadow-xs flex flex-col items-center text-center">
          <MascotRobot size={90} mood="happy" className="mb-2" />
          <h4 className="font-extrabold text-sm text-[#281238]">
            Your Document Assistant
          </h4>
          <p className="text-xs text-[#2A1E39]/70 mt-1 leading-relaxed">
            Find, understand, and extract key values from your personal documents with 100% privacy and zero hallucination.
          </p>

          <div className="w-full mt-4 pt-4 border-t border-[#D9CCFF]/40 text-left space-y-2 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-[#2A1E39]/70">Verified Facts:</span>
              <span className="font-bold text-[#7541DA]">
                {documents.reduce((acc, d) => acc + (d.fields?.length || 0), 0)} fields
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[#2A1E39]/70">Indexed Chunks:</span>
              <span className="font-bold text-[#7541DA]">
                {documents.reduce((acc, d) => acc + (d.chunks?.length || 0), 0)} chunks
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[#2A1E39]/70">Language Mode:</span>
              <span className="font-bold text-emerald-700">En / Hi / Hinglish</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
