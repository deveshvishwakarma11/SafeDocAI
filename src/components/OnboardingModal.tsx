import React, { useState } from 'react';
import { X, Shield, Search, FileText, CheckCircle2, ArrowRight, Sparkles } from 'lucide-react';
import { MascotRobot } from './MascotArt';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onGetStarted: () => void;
}

export const OnboardingModal: React.FC<Props> = ({ isOpen, onClose, onGetStarted }) => {
  const [step, setStep] = useState(1);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#281238]/60 backdrop-blur-xs p-4 animate-in fade-in duration-200">
      <div className="bg-white rounded-2xl shadow-2xl border border-[#D9CCFF] max-w-xl w-full flex flex-col overflow-hidden">
        {/* Top Progress */}
        <div className="p-4 border-b border-[#D9CCFF]/40 bg-[#F5EFFF]/50 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-[#7541DA] uppercase tracking-wider">
              Step {step} of 3
            </span>
            <div className="flex gap-1 ml-2">
              <span className={`w-6 h-1.5 rounded-full ${step >= 1 ? 'bg-[#7541DA]' : 'bg-[#D9CCFF]'}`} />
              <span className={`w-6 h-1.5 rounded-full ${step >= 2 ? 'bg-[#7541DA]' : 'bg-[#D9CCFF]'}`} />
              <span className={`w-6 h-1.5 rounded-full ${step >= 3 ? 'bg-[#7541DA]' : 'bg-[#D9CCFF]'}`} />
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg flex items-center justify-center text-[#2A1E39]/50 hover:bg-[#D9CCFF]/40 hover:text-[#281238]"
          >
            <X size={18} />
          </button>
        </div>

        {/* Body Content */}
        <div className="p-8 text-center flex flex-col items-center">
          {step === 1 && (
            <div className="space-y-4 max-w-md">
              <MascotRobot size={110} mood="waving" className="mx-auto" />
              <h3 className="text-xl font-extrabold text-[#281238]">
                Welcome to SafeDocAI
              </h3>
              <p className="text-sm text-[#2A1E39]/70 leading-relaxed">
                SafeDocAI turns your scattered PDFs, marksheets, receipts, and identity cards into a
                searchable, queryable personal knowledge base — <span className="font-semibold text-[#7541DA]">100% on your device</span>.
              </p>
              <div className="p-3.5 rounded-xl bg-[#F5EFFF] border border-[#D9CCFF] text-left text-xs text-[#281238] space-y-1.5">
                <div className="flex items-center gap-2 font-bold text-[#7541DA]">
                  <Shield size={14} /> Local-First Guarantee
                </div>
                <p className="text-[#2A1E39]/70">
                  Your files, OCR text, and AI embeddings never leave this device. No accounts, no subscriptions, no cloud leaks.
                </p>
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="space-y-4 max-w-md">
              <MascotRobot size={110} mood="thinking" className="mx-auto" />
              <h3 className="text-xl font-extrabold text-[#281238]">
                Ask in Natural Language
              </h3>
              <p className="text-sm text-[#2A1E39]/70 leading-relaxed">
                Ask questions in everyday English, Hindi, or Hinglish without memorizing exact filenames or hunting through folders.
              </p>
              <div className="space-y-2 text-left">
                <div className="p-2.5 rounded-xl bg-[#FBF8FF] border border-[#D9CCFF] text-xs flex items-center gap-2 text-[#281238]">
                  <span className="font-bold text-[#7541DA]">“Mera roll number kya hai?”</span>
                  <span className="text-[#2A1E39]/40 ml-auto">Exact Field</span>
                </div>
                <div className="p-2.5 rounded-xl bg-[#FBF8FF] border border-[#D9CCFF] text-xs flex items-center gap-2 text-[#281238]">
                  <span className="font-bold text-[#7541DA]">“Electricity bill amount kitna hai?”</span>
                  <span className="text-[#2A1E39]/40 ml-auto">Exact Field</span>
                </div>
                <div className="p-2.5 rounded-xl bg-[#FBF8FF] border border-[#D9CCFF] text-xs flex items-center gap-2 text-[#281238]">
                  <span className="font-bold text-[#7541DA]">“Summarize my health insurance policy”</span>
                  <span className="text-[#2A1E39]/40 ml-auto">Semantic RAG</span>
                </div>
              </div>
            </div>
          )}

          {step === 3 && (
            <div className="space-y-4 max-w-md">
              <MascotRobot size={110} mood="happy" className="mx-auto" />
              <h3 className="text-xl font-extrabold text-[#281238]">
                Grounded Evidence, Zero Hallucination
              </h3>
              <p className="text-sm text-[#2A1E39]/70 leading-relaxed">
                SafeDocAI follows the strict Grounding Contract: Answers lead with the verified fact,
                cite the exact document and page, and show the source quote. If a fact cannot be proven, SafeDocAI will say so honestly.
              </p>
              <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200 text-left text-xs text-emerald-800 space-y-1">
                <p className="font-bold flex items-center gap-1.5">
                  <CheckCircle2 size={14} className="text-emerald-600" /> Default Local Profile Ready
                </p>
                <p className="text-emerald-700/80">
                  Profile <strong>default_local_profile</strong> initialized on this machine — no signup or cloud account. Upload your first document to begin querying.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Footer controls */}
        <div className="p-4 border-t border-[#D9CCFF]/40 bg-[#FBF8FF] flex items-center justify-between">
          {step > 1 ? (
            <button
              onClick={() => setStep(step - 1)}
              className="px-4 py-2 text-xs font-bold text-[#2A1E39]/70 hover:text-[#7541DA]"
            >
              Back
            </button>
          ) : (
            <button
              onClick={onClose}
              className="px-4 py-2 text-xs font-bold text-[#2A1E39]/50 hover:text-[#281238]"
            >
              Skip
            </button>
          )}

          {step < 3 ? (
            <button
              onClick={() => setStep(step + 1)}
              className="inline-flex items-center gap-1.5 px-5 py-2.5 rounded-xl text-xs font-bold bg-[#7541DA] text-white hover:bg-[#8B42E8] shadow-sm shadow-[#7541DA]/20 transition-all"
            >
              <span>Continue</span>
              <ArrowRight size={14} />
            </button>
          ) : (
            <button
              onClick={() => {
                onClose();
                onGetStarted();
              }}
              className="inline-flex items-center gap-1.5 px-5 py-2.5 rounded-xl text-xs font-bold bg-[#7541DA] text-white hover:bg-[#8B42E8] shadow-sm shadow-[#7541DA]/20 transition-all"
            >
              <span>Enter Workspace</span>
              <ArrowRight size={14} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
