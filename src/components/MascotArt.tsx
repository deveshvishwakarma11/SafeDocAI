import React from 'react';

interface MascotProps {
  className?: string;
  size?: number;
  mood?: 'happy' | 'thinking' | 'waving' | 'search';
}

export const MascotRobot: React.FC<MascotProps> = ({ className = '', size = 120, mood = 'happy' }) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 200 200"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={`${className} transition-transform duration-300 hover:scale-105 select-none`}
    >
      <defs>
        <linearGradient id="bodyGrad" x1="50" y1="50" x2="150" y2="170" gradientUnits="userSpaceOnUse">
          <stop stopColor="#FFFFFF" />
          <stop offset="0.85" stopColor="#F5EFFF" />
          <stop offset="1" stopColor="#D9CCFF" />
        </linearGradient>

        <linearGradient id="violetGrad" x1="40" y1="40" x2="160" y2="160" gradientUnits="userSpaceOnUse">
          <stop stopColor="#8B42E8" />
          <stop offset="1" stopColor="#7541DA" />
        </linearGradient>

        <linearGradient id="visorGrad" x1="60" y1="70" x2="140" y2="110" gradientUnits="userSpaceOnUse">
          <stop stopColor="#281238" />
          <stop offset="1" stopColor="#3A155E" />
        </linearGradient>

        <radialGradient id="eyeGlow" cx="50%" cy="50%" r="50%">
          <stop stopColor="#F3A9DB" offset="0%" />
          <stop stopColor="#B768F2" offset="80%" />
          <stop stopColor="#7541DA" offset="100%" />
        </radialGradient>

        <filter id="softGlow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
      </defs>

      {/* Shadow */}
      <ellipse cx="100" cy="184" rx="42" ry="9" fill="#D9CCFF" fillOpacity="0.55" />

      {/* Antenna */}
      <line x1="100" y1="46" x2="100" y2="28" stroke="#7541DA" strokeWidth="4" strokeLinecap="round" />
      <circle cx="100" cy="24" r="8" fill="url(#violetGrad)" />
      <circle cx="100" cy="24" r="4" fill="#F3A9DB" filter="url(#softGlow)" />

      {/* Floating Ears / Ear cups */}
      <rect x="42" y="72" width="10" height="26" rx="5" fill="#7541DA" />
      <rect x="148" y="72" width="10" height="26" rx="5" fill="#7541DA" />

      {/* Main Head Base */}
      <rect x="52" y="46" width="96" height="78" rx="30" fill="url(#bodyGrad)" stroke="#7541DA" strokeWidth="3" />

      {/* Dark Visor */}
      <rect x="64" y="62" width="72" height="46" rx="16" fill="url(#visorGrad)" />

      {/* Expressive Glowing Eyes */}
      {mood === 'happy' && (
        <>
          <path d="M78 86C78 81 83 78 88 78C93 78 98 81 98 86" stroke="#B768F2" strokeWidth="4.5" strokeLinecap="round" />
          <path d="M102 86C102 81 107 78 112 78C117 78 122 81 122 86" stroke="#B768F2" strokeWidth="4.5" strokeLinecap="round" />
          {/* Soft pink blush */}
          <circle cx="74" cy="98" r="4" fill="#F3A9DB" fillOpacity="0.6" />
          <circle cx="126" cy="98" r="4" fill="#F3A9DB" fillOpacity="0.6" />
        </>
      )}

      {mood === 'thinking' && (
        <>
          <circle cx="86" cy="85" r="6" fill="#F3A9DB" />
          <circle cx="114" cy="85" r="6" fill="#F3A9DB" />
          {/* Curious eyebrow */}
          <path d="M80 75L92 73" stroke="#B768F2" strokeWidth="2.5" strokeLinecap="round" />
          <path d="M108 73L120 75" stroke="#B768F2" strokeWidth="2.5" strokeLinecap="round" />
        </>
      )}

      {mood === 'search' && (
        <>
          <circle cx="86" cy="85" r="7" fill="#B768F2" />
          <circle cx="88" cy="83" r="2.5" fill="#FFFFFF" />
          <circle cx="114" cy="85" r="7" fill="#B768F2" />
          <circle cx="116" cy="83" r="2.5" fill="#FFFFFF" />
          {/* Smile line */}
          <path d="M94 96C97 99 103 99 106 96" stroke="#F3A9DB" strokeWidth="2.5" strokeLinecap="round" />
        </>
      )}

      {mood === 'waving' && (
        <>
          <path d="M78 85C78 80 84 78 88 80C92 82 94 86 94 89" stroke="#B768F2" strokeWidth="4" strokeLinecap="round" />
          <path d="M106 89C106 86 108 82 112 80C116 78 122 80 122 85" stroke="#B768F2" strokeWidth="4" strokeLinecap="round" />
        </>
      )}

      {/* Floating Neck Joint */}
      <rect x="91" y="124" width="18" height="8" rx="4" fill="#7541DA" />

      {/* Robot Torso */}
      <path
        d="M62 134C62 128 67 124 74 124H126C133 124 138 128 138 134V154C138 165 129 174 118 174H82C71 174 62 165 62 154V134Z"
        fill="url(#bodyGrad)"
        stroke="#7541DA"
        strokeWidth="3"
      />

      {/* Document emblem on chest */}
      <rect x="86" y="138" width="28" height="24" rx="4" fill="#7541DA" />
      <path d="M92 144H108M92 149H104M92 154H100" stroke="#FFFFFF" strokeWidth="2" strokeLinecap="round" />

      {/* Floating Hands */}
      {mood === 'waving' ? (
        <>
          <ellipse cx="44" cy="115" rx="9" ry="8" fill="url(#bodyGrad)" stroke="#7541DA" strokeWidth="2" />
          <ellipse cx="156" cy="144" rx="8" ry="9" fill="url(#bodyGrad)" stroke="#7541DA" strokeWidth="2" />
        </>
      ) : (
        <>
          <ellipse cx="44" cy="144" rx="8" ry="9" fill="url(#bodyGrad)" stroke="#7541DA" strokeWidth="2" />
          <ellipse cx="156" cy="144" rx="8" ry="9" fill="url(#bodyGrad)" stroke="#7541DA" strokeWidth="2" />
        </>
      )}
    </svg>
  );
};

export const MascotHeroScene: React.FC<{ className?: string }> = ({ className = '' }) => {
  return (
    <div className={`relative flex items-center justify-center p-6 ${className}`}>
      {/* Ambient background glow */}
      <div className="absolute w-72 h-72 rounded-full bg-gradient-to-tr from-[#7541DA]/20 via-[#B768F2]/15 to-[#F3A9DB]/20 blur-2xl pointer-events-none" />

      {/* Floating Paper 1 (Left: Verified Marksheet) */}
      <div className="absolute -left-2 top-8 transform -rotate-12 bg-white/95 backdrop-blur-md p-3 rounded-xl shadow-lg border border-[#D9CCFF] flex items-center gap-2.5 z-10 animate-pulse">
        <div className="w-8 h-8 rounded-lg bg-[#7541DA]/10 flex items-center justify-center text-[#7541DA] font-bold text-xs">
          PDF
        </div>
        <div className="text-left">
          <p className="text-xs font-semibold text-[#281238]">Marksheet.pdf</p>
          <p className="text-[10px] text-[#7541DA] font-medium">Roll No: 2023CS001</p>
        </div>
      </div>

      {/* Floating Paper 2 (Right: Verified Bill) */}
      <div className="absolute -right-4 top-16 transform rotate-6 bg-white/95 backdrop-blur-md p-3 rounded-xl shadow-lg border border-[#D9CCFF] flex items-center gap-2.5 z-10">
        <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-600 font-bold text-xs">
          ₹
        </div>
        <div className="text-left">
          <p className="text-xs font-semibold text-[#281238]">Electricity Bill</p>
          <p className="text-[10px] text-emerald-600 font-medium">Due: 28 Oct · ₹1,840</p>
        </div>
      </div>

      {/* Floating Privacy Badge (Bottom Left) */}
      <div className="absolute left-4 bottom-2 transform -rotate-3 bg-[#281238] text-white px-3 py-1.5 rounded-lg shadow-md text-[11px] font-semibold flex items-center gap-1.5 z-10">
        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
        100% On-Device Safe
      </div>

      {/* Central Mascot Robot */}
      <div className="relative z-0">
        <MascotRobot size={180} mood="happy" />
      </div>
    </div>
  );
};
