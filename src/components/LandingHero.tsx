import React from 'react';
import {
  ShieldCheck,
  Search,
  FileCheck2,
  Lock,
  Sparkles,
  ArrowRight,
  CheckCircle2,
  HelpCircle,
  FolderLock
} from 'lucide-react';
import { MascotHeroScene, MascotRobot } from './MascotArt';

interface Props {
  onGetStarted: () => void;
  onOpenGuide: () => void;
  onOpenReadiness: () => void;
  documentCount: number;
}

export const LandingHero: React.FC<Props> = ({
  onGetStarted,
  onOpenGuide,
  onOpenReadiness,
  documentCount
}) => {
  return (
    <div className="relative overflow-hidden bg-gradient-to-b from-[#281238] via-[#3A155E] to-[#7541DA] text-white py-12 px-6 sm:px-12 rounded-3xl shadow-xl border border-[#D9CCFF]/20 my-2 select-none">
      {/* Decorative stars and lighting */}
      <div className="absolute top-0 right-0 w-96 h-96 bg-[#8B42E8]/30 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-96 h-96 bg-[#B768F2]/20 rounded-full blur-3xl pointer-events-none" />

      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8 items-center relative z-10">
        {/* Left Column: Headline and Actions (7 cols) */}
        <div className="lg:col-span-7 space-y-6 text-left">
          {/* Brand Tag */}
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/10 backdrop-blur-md border border-white/20 text-xs font-semibold tracking-wide text-[#F3A9DB]">
            <Sparkles size={13} className="text-[#F3A9DB]" />
            <span>SafeDocAI v1.0 • Local-First Personal Document Intelligence</span>
          </div>

          {/* Main Headline */}
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black tracking-tight leading-[1.1] text-white">
            Your Documents, <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#F3A9DB] via-[#D9CCFF] to-white">
              Finally Make Sense.
            </span>
          </h1>

          {/* Subtitle */}
          <p className="text-base sm:text-lg text-[#F5EFFF]/90 leading-relaxed max-w-xl font-normal">
            Upload, search, and chat with your documents in everyday language (English, Hindi, and Hinglish).
            <strong className="text-white font-semibold"> 100% private, processed on your device.</strong>
          </p>

          {/* Action Buttons */}
          <div className="flex flex-wrap items-center gap-3 pt-2">
            <button
              onClick={onGetStarted}
              className="inline-flex items-center gap-2 px-7 py-3.5 rounded-2xl text-sm font-extrabold bg-[#FFFFFF] text-[#281238] hover:bg-[#F5EFFF] shadow-lg shadow-black/20 hover:scale-[1.02] active:scale-[0.98] transition-all cursor-pointer"
            >
              <span>{documentCount > 0 ? 'Open Workspace' : 'Get Started'}</span>
              <ArrowRight size={16} className="text-[#7541DA]" />
            </button>

            <button
              onClick={onOpenGuide}
              className="inline-flex items-center gap-2 px-5 py-3.5 rounded-2xl text-sm font-bold bg-white/10 hover:bg-white/15 text-white border border-white/20 backdrop-blur-md transition-all cursor-pointer"
            >
              <HelpCircle size={16} />
              <span>How It Works</span>
            </button>

            <button
              onClick={onOpenReadiness}
              className="inline-flex items-center gap-1.5 px-4 py-3.5 rounded-2xl text-xs font-semibold text-[#D9CCFF] hover:text-white transition-colors"
            >
              <Lock size={13} className="text-emerald-400" />
              <span>Offline Verified</span>
            </button>
          </div>

          {/* Trust Strip */}
          <div className="pt-4 border-t border-white/15 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div className="flex items-center gap-2 text-white/90">
              <CheckCircle2 size={15} className="text-emerald-400 shrink-0" />
              <span>Private & Local</span>
            </div>
            <div className="flex items-center gap-2 text-white/90">
              <CheckCircle2 size={15} className="text-emerald-400 shrink-0" />
              <span>Understand Any Doc</span>
            </div>
            <div className="flex items-center gap-2 text-white/90">
              <CheckCircle2 size={15} className="text-emerald-400 shrink-0" />
              <span>Find Exact Facts</span>
            </div>
            <div className="flex items-center gap-2 text-white/90">
              <CheckCircle2 size={15} className="text-emerald-400 shrink-0" />
              <span>Zero Hallucination</span>
            </div>
          </div>
        </div>

        {/* Right Column: Hero Mascot Scene (5 cols) */}
        <div className="lg:col-span-5 flex justify-center items-center">
          <MascotHeroScene />
        </div>
      </div>
    </div>
  );
};
