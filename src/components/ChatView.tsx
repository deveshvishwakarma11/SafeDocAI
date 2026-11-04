import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Square,
  Trash2,
  FileText,
  ExternalLink,
  CheckCircle2,
  AlertCircle,
  Sparkles,
  Layers,
  HelpCircle,
  Database,
  Cpu
} from 'lucide-react';
import { ChatMessage, QueryResult, StoredDocument } from '../types';
import { QueryRouter } from '../utils/queryRouter';
import { renderChatMarkdown } from '../utils/markdown';
import { MascotRobot } from './MascotArt';

/**
 * Dedupe evidence rows: the same document coming back twice (once per field)
 * is ONE source, not two — duplicates made the evidence list look broken.
 */
const evidenceRows = (msg: ChatMessage) => {
  const seen = new Set<string>();
  return (msg.evidence || []).filter(ev => {
    const key = `${ev.document_id}|${ev.field_name || ''}|${(ev.snippet || '').slice(0, 80)}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
};

interface Props {
  messages: ChatMessage[];
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  documents: StoredDocument[];
  onOpenViewer: (doc: StoredDocument) => void;
  initialQuery?: string;
  /** Document id the chat is pinned to (survives closing the viewer). */
  scopedDocumentId?: number;
  onScopeChange: (id: number | undefined) => void;
}

export const ChatView: React.FC<Props> = ({
  messages,
  setMessages,
  documents,
  onOpenViewer,
  initialQuery,
  scopedDocumentId,
  onScopeChange
}) => {
  const [inputText, setInputText] = useState(initialQuery || '');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortCtlRef = useRef<AbortController | null>(null);

  const activeDocFilter = scopedDocumentId;
  const scopedDoc = activeDocFilter
    ? documents.find(d => d.id === activeDocFilter) || null
    : null;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  useEffect(() => {
    if (initialQuery && initialQuery.trim()) {
      handleSendMessage(initialQuery);
    }
  }, [initialQuery]);

  const handleSendMessage = async (queryText: string) => {
    const text = queryText.trim();
    if (!text || isLoading) return;

    // Each request owns its own AbortController: Stop aborts the in-flight
    // fetch, and a late reply from a stopped query can never append itself.
    abortCtlRef.current?.abort();
    const ctl = new AbortController();
    abortCtlRef.current = ctl;

    const userMessage: ChatMessage = {
      id: `msg_${Date.now()}_u`,
      role: 'user',
      content: text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      const result: QueryResult = await QueryRouter.executeQuery(text, {
        documents,
        activeDocFilter,
        signal: ctl.signal,
        isCancelled: () => ctl.signal.aborted
      });

      if (ctl.signal.aborted) {
        return;
      }

      const assistantMessage: ChatMessage = {
        id: `msg_${Date.now()}_a`,
        role: 'assistant',
        content: result.answer,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        route: result.route,
        intent: result.intent,
        evidence: result.evidence,
        clarification_options: result.clarification_options,
        is_insufficient_evidence: result.is_insufficient_evidence
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err: any) {
      if (!ctl.signal.aborted && err?.name !== 'AbortError') {
        setMessages(prev => [
          ...prev,
          {
            id: `msg_${Date.now()}_err`,
            role: 'assistant',
            content: `Error retrieving evidence: ${err.message || 'Unknown processing error'}`,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            is_insufficient_evidence: true
          }
        ]);
      }
    } finally {
      if (abortCtlRef.current === ctl) {
        abortCtlRef.current = null;
        setIsLoading(false);
      }
    }
  };

  const handleStopQuery = () => {
    const ctl = abortCtlRef.current;
    abortCtlRef.current = null;
    ctl?.abort();
    setIsLoading(false);
    setMessages(prev => [
      ...prev,
      {
        id: `msg_${Date.now()}_stop`,
        role: 'assistant',
        content: '⏹ Generation stopped.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        is_cancelled: true
      }
    ]);
  };

  const handleClearChat = () => {
    setMessages([]);
  };

  const getRouteBadge = (msg: ChatMessage) => {
    // Honest labelling: an insufficient-evidence answer is NOT an out-of-scope
    // question — show what actually happened.
    if (msg.is_insufficient_evidence && msg.intent !== 'unrelated_request') {
      return (
        <span className="inline-flex items-center gap-1 text-[10px] font-bold text-amber-700 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
          <AlertCircle size={10} /> Insufficient Evidence
        </span>
      );
    }

    // Intent-specific labels: an all-fields dump is a SQLite field summary, not
    // a vector search, and a copied line is a quote — say what actually ran.
    const byIntent: Record<string, { text: string; cls: string; icon: React.ReactNode }> = {
      field_summary: {
        text: 'SQLite Field Summary',
        cls: 'text-[#7541DA] bg-[#F5EFFF] border-[#D9CCFF]',
        icon: <Database size={10} />
      },
      document_details: {
        text: 'SQLite Field Summary',
        cls: 'text-[#7541DA] bg-[#F5EFFF] border-[#D9CCFF]',
        icon: <Database size={10} />
      },
      document_summary: {
        text: 'SQLite Field Summary',
        cls: 'text-[#7541DA] bg-[#F5EFFF] border-[#D9CCFF]',
        icon: <Database size={10} />
      },
      document_type_profile: {
        text: 'Document Type',
        cls: 'text-blue-700 bg-blue-50 border-blue-200',
        icon: <FileText size={10} />
      },
      raw_excerpt_answer: {
        text: 'Verbatim Excerpt',
        cls: 'text-emerald-700 bg-emerald-50 border-emerald-200',
        icon: <CheckCircle2 size={10} />
      }
    };

    if (msg.intent) {
      const direct = byIntent[msg.intent];
      if (direct) {
        return (
          <span className={`inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded border ${direct.cls}`}>
            {direct.icon} {direct.text}
          </span>
        );
      }
      if (msg.intent.startsWith('raw_line_lookup_') || msg.intent.startsWith('exact_lookup_')) {
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-bold text-[#7541DA] bg-[#F5EFFF] px-2 py-0.5 rounded border border-[#D9CCFF]">
            <Database size={10} /> SQLite Exact Field
          </span>
        );
      }
    }

    switch (msg.route) {
      case 'exact_field':
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-bold text-[#7541DA] bg-[#F5EFFF] px-2 py-0.5 rounded border border-[#D9CCFF]">
            <Database size={10} /> SQLite Exact Field
          </span>
        );
      case 'semantic_rag':
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-bold text-violet-700 bg-violet-50 px-2 py-0.5 rounded border border-violet-200">
            <Cpu size={10} /> Local Vector Search
          </span>
        );
      case 'document_retrieval':
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-bold text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-200">
            <FileText size={10} /> Document Retrieval
          </span>
        );
      case 'clarification':
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-bold text-amber-700 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
            <HelpCircle size={10} /> Clarification Needed
          </span>
        );
      case 'greeting':
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-bold text-[#7541DA] bg-[#F5EFFF] px-2 py-0.5 rounded border border-[#D9CCFF]">
            <Sparkles size={10} /> SafeDocAI Assistant
          </span>
        );
      case 'out_of_scope':
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-bold text-slate-700 bg-slate-50 px-2 py-0.5 rounded border border-slate-200">
            <AlertCircle size={10} /> Scope Guard
          </span>
        );
      default:
        return null;
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)] bg-white rounded-3xl border border-[#D9CCFF]/60 shadow-sm overflow-hidden select-none">
      {/* Chat Sub-Header */}
      <div className="p-4 px-6 border-b border-[#D9CCFF]/40 bg-[#F5EFFF]/40 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <MascotRobot size={36} mood="happy" />
          <div>
            <h3 className="font-extrabold text-sm text-[#281238] flex items-center gap-2">
              SafeDocAI Evidence Chat
              <span className="text-[10px] font-semibold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-200">
                Grounding Active
              </span>
            </h3>
            <p className="text-xs text-[#2A1E39]/60">
              Answers are 100% tied to stored documents with zero guesswork.
            </p>

            {/* Scope chip: always visible, so it is obvious WHICH file answers */}
            <div className="mt-1.5 flex items-center gap-1.5 flex-wrap">
              <span
                className={`inline-flex items-center gap-1.5 text-[11px] font-bold px-2.5 py-1 rounded-full border max-w-[260px] ${
                  scopedDoc
                    ? 'text-[#7541DA] bg-white border-[#D9CCFF]'
                    : 'text-[#2A1E39]/60 bg-white/70 border-[#D9CCFF]/60'
                }`}
                title={scopedDoc ? `Answering only from ${scopedDoc.file_name}` : 'Answering from all documents'}
              >
                <FileText size={11} className="shrink-0" />
                <span className="shrink-0">Answering from:</span>
                <span className="truncate">
                  {scopedDoc ? scopedDoc.file_name : `All documents (${documents.length})`}
                </span>
              </span>

              <select
                value={activeDocFilter ?? ''}
                onChange={e => onScopeChange(e.target.value ? Number(e.target.value) : undefined)}
                className="text-[11px] bg-white border border-[#D9CCFF] text-[#281238] font-semibold rounded-full px-2 py-1 outline-hidden cursor-pointer"
                title="Change which document answers"
              >
                <option value="">All documents ({documents.length})</option>
                {documents.map(d => (
                  <option key={d.id} value={d.id}>
                    {d.file_name}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {messages.length > 0 && (
            <button
              onClick={handleClearChat}
              title="Clear session messages"
              className="p-1.5 rounded-lg text-[#2A1E39]/50 hover:text-red-600 hover:bg-red-50 transition-colors"
            >
              <Trash2 size={16} />
            </button>
          )}
        </div>
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-5 bg-[#FBF8FF]/50">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 max-w-md mx-auto">
            <MascotRobot size={110} mood="waving" className="mb-3" />
            <h4 className="font-extrabold text-base text-[#281238]">
              {scopedDoc ? `Ask Anything About ${scopedDoc.file_name}` : 'Ask Anything About Your Documents'}
            </h4>
            <p className="text-xs text-[#2A1E39]/70 mt-1 mb-6 leading-relaxed">
              {scopedDoc
                ? 'Every answer is quoted from this file only — its extracted fields first, then its transcribed lines. Nothing is guessed.'
                : 'SafeDocAI understands natural language in English, Hindi, and Hinglish. Open a document to pin the chat to it, or ask across all files.'}
            </p>

            {/* Starter Pills — derived from what THIS document actually holds */}
            <div className="w-full space-y-2 text-left">
              <p className="text-[11px] font-bold uppercase tracking-wider text-[#7541DA]">
                {scopedDoc ? 'Ask about this file:' : 'Try one of these real queries:'}
              </p>

              {scopedDoc
                ? [
                    ...(scopedDoc.fields || [])
                      .filter(f => f.field_name !== 'candidate_name')
                      .slice(0, 3)
                      .map(f => ({ label: `What is the ${f.display_label.toLowerCase()}?`, text: `What is the ${f.display_label.toLowerCase()}?` })),
                    { label: 'Show me all details', text: 'show all details' },
                    { label: 'What kind of document is this?', text: 'what kind of document is this?' }
                  ].map(item => (
                    <button
                      key={item.label}
                      onClick={() => handleSendMessage(item.text)}
                      className="w-full p-2.5 text-xs rounded-xl bg-white border border-[#D9CCFF] text-[#281238] hover:border-[#7541DA] hover:bg-[#F5EFFF] text-left transition-all"
                    >
                      → {item.label}
                    </button>
                  ))
                : [
                    { label: 'What documents do you have?', text: 'What documents do you have?' },
                    { label: 'Mera roll number kya hai?', text: 'Mera roll number kya hai?' },
                    { label: 'What is my electricity bill amount?', text: 'What is my electricity bill amount?' },
                    { label: 'What is the capital of France?', text: 'What is the capital of France?' }
                  ].map(item => (
                    <button
                      key={item.label}
                      onClick={() => handleSendMessage(item.text)}
                      className="w-full p-2.5 text-xs rounded-xl bg-white border border-[#D9CCFF] text-[#281238] hover:border-[#7541DA] hover:bg-[#F5EFFF] text-left transition-all"
                    >
                      → {item.label}
                    </button>
                  ))}
            </div>
          </div>
        ) : (
          messages
            .filter(msg => (msg.content || '').trim().length > 0)
            .map(msg => (
            <div
              key={msg.id}
              className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
            >
              {/* Message Bubble */}
              <div
                className={`rounded-2xl shadow-xs ${
                  msg.role === 'user'
                    ? 'max-w-[80%] p-3 px-4 bg-[#F5EFFF] text-[#281238] border border-[#D9CCFF]'
                    : 'w-full max-w-3xl p-4 bg-white text-[#281238] border border-[#D9CCFF] shadow-sm'
                }`}
              >
                {/* Assistant Metadata Header */}
                {msg.role === 'assistant' && (
                  <div className="flex items-center gap-2 flex-wrap pb-2 mb-2 border-b border-[#D9CCFF]/40">
                    <MascotRobot size={22} mood="happy" />
                    <span className="text-xs font-extrabold text-[#281238]">SafeDocAI</span>
                    {/* Badge sits right after the name, not pinned to the card edge */}
                    {getRouteBadge(msg)}
                  </div>
                )}

                {/* Message Content (grounded answer markdown: bold/code/headings) */}
                <div
                  className="text-sm leading-relaxed whitespace-pre-wrap break-words text-[#281238]"
                  dangerouslySetInnerHTML={{ __html: renderChatMarkdown(msg.content) }}
                />

                {/* Clarification Options Buttons */}
                {msg.clarification_options && msg.clarification_options.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-[#D9CCFF]/40 space-y-1.5">
                    <p className="text-[11px] font-bold text-[#7541DA]">
                      Select an option to proceed:
                    </p>
                    <div className="flex flex-col gap-1.5">
                      {msg.clarification_options.map((opt, idx) => (
                        <button
                          key={idx}
                          onClick={() => handleSendMessage(opt.value)}
                          className="text-left text-xs p-2 rounded-xl bg-[#F5EFFF] hover:bg-[#EAE0FF] text-[#281238] font-semibold border border-[#D9CCFF] transition-colors"
                        >
                          → {opt.label}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Source Evidence Card */}
                {msg.evidence && msg.evidence.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-[#D9CCFF]/40 space-y-2">
                    <div className="flex items-center gap-1 text-[11px] text-[#7541DA] font-semibold">
                      <CheckCircle2 size={12} className="text-emerald-600" />
                      Grounded Source Evidence ({evidenceRows(msg).length})
                    </div>

                    <div className="space-y-1.5">
                      {evidenceRows(msg).map((ev, i) => {
                        const targetDoc = documents.find(d => d.id === ev.document_id);
                        const snippet = (ev.snippet || '').replace(/\s+/g, ' ').trim();
                        return (
                          <div
                            key={i}
                            className="p-2.5 rounded-xl bg-[#FBF8FF] border border-[#D9CCFF]/60 text-xs flex items-start gap-3 hover:border-[#7541DA] transition-colors"
                          >
                            <div className="min-w-0 flex-1">
                              <div className="flex items-center gap-2 flex-wrap">
                                <span className="font-bold text-[#281238] truncate max-w-[60%]">
                                  {ev.file_name}
                                </span>
                                {ev.field_name && (
                                  <span className="text-[10px] font-semibold text-[#7541DA] bg-[#F5EFFF] px-1.5 py-0.5 rounded">
                                    {ev.field_name}
                                  </span>
                                )}
                                {ev.page_number && (
                                  <span className="text-[10px] text-[#2A1E39]/50">Page {ev.page_number}</span>
                                )}
                              </div>
                              {snippet && (
                                <p className="mt-1 text-[11px] font-mono text-[#2A1E39]/75 break-words line-clamp-2">
                                  “{snippet}”
                                </p>
                              )}
                            </div>

                            {targetDoc && (
                              <button
                                onClick={() => onOpenViewer(targetDoc)}
                                className="inline-flex items-center gap-1 text-[11px] font-bold text-[#7541DA] px-2 py-1 rounded-lg bg-white border border-[#D9CCFF] hover:bg-[#F5EFFF] shrink-0 mt-0.5"
                              >
                                <span>Inspect</span>
                                <ExternalLink size={11} />
                              </button>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Timestamp — own row so it cannot collide with evidence */}
                <div className="mt-2.5 pt-1.5 text-[10px] text-[#2A1E39]/45 text-right tabular-nums">
                  {msg.timestamp}
                </div>
              </div>
            </div>
          ))
        )}

        {/* Loading Indicator with Stop Action */}
        {isLoading && (
          <div className="flex flex-col items-start space-y-2">
            <div className="p-4 rounded-2xl bg-white border border-[#D9CCFF] shadow-sm flex items-center gap-3">
              <MascotRobot size={30} mood="thinking" className="animate-spin" />
              <div className="text-xs">
                <p className="font-bold text-[#281238]">
                  Retrieving local evidence & verifying ground truth...
                </p>
                <p className="text-[10px] text-[#2A1E39]/60">
                  SQLite exact lookup + local vector search
                </p>
              </div>
            </div>

            <button
              onClick={handleStopQuery}
              className="inline-flex items-center gap-1 px-3 py-1 rounded-lg text-xs font-semibold bg-red-50 text-red-700 border border-red-200 hover:bg-red-100 transition-colors"
            >
              <Square size={11} fill="currentColor" />
              <span>Stop Generating</span>
            </button>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Composer */}
      <div className="p-4 border-t border-[#D9CCFF]/40 bg-white">
        <form
          onSubmit={e => {
            e.preventDefault();
            handleSendMessage(inputText);
          }}
          className="flex items-center gap-2"
        >
          <input
            type="text"
            value={inputText}
            onChange={e => setInputText(e.target.value)}
            disabled={isLoading}
            placeholder="Type your question in English, Hindi, or Hinglish..."
            className="flex-1 bg-[#FBF8FF] border border-[#D9CCFF] rounded-2xl px-4 py-3 text-sm text-[#281238] placeholder-[#2A1E39]/40 outline-hidden focus:border-[#7541DA] transition-colors"
          />

          <button
            type="submit"
            disabled={!inputText.trim() || isLoading}
            className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-[#7541DA] text-white hover:bg-[#8B42E8] disabled:opacity-40 transition-all shrink-0 cursor-pointer shadow-sm shadow-[#7541DA]/20"
          >
            <Send size={18} />
          </button>
        </form>
      </div>
    </div>
  );
};
