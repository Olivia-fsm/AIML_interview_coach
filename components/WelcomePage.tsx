import React from 'react';

interface Props {
  onStart: () => void;
}

const WelcomePage: React.FC<Props> = ({ onStart }) => {
  return (
    <div className="min-h-screen bg-[#0f172a] text-white overflow-hidden relative selection:bg-pink-500 selection:text-white">
      {/* Background Gradients - Shifted to more romantic/deepspace hues */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden z-0 pointer-events-none">
        <div className="absolute -top-[20%] -left-[10%] w-[60%] h-[60%] rounded-full bg-indigo-600/20 blur-[120px] animate-pulse"></div>
        <div className="absolute top-[40%] -right-[10%] w-[50%] h-[50%] rounded-full bg-pink-600/20 blur-[120px] animate-pulse"></div>
        <div className="absolute -bottom-[20%] left-[20%] w-[40%] h-[40%] rounded-full bg-purple-500/10 blur-[100px]"></div>
      </div>

      <div className="relative z-10 container mx-auto px-6 py-12 flex flex-col items-center justify-center min-h-screen">
        
        {/* Header / Nav placeholder */}
        <div className="absolute top-6 left-6 flex items-center gap-3 opacity-90">
          <div className="w-10 h-10 bg-gradient-to-br from-pink-500 via-purple-500 to-indigo-500 rounded-xl flex items-center justify-center shadow-lg shadow-pink-500/20">
             {/* Love&DeepCode Logo: Heart with Code Brackets */}
             <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
                <path d="M9 10h.01" strokeWidth="4" strokeLinecap="round"/>
                <path d="M15 10h.01" strokeWidth="4" strokeLinecap="round"/>
             </svg>
          </div>
          <span className="font-bold tracking-wider text-xl bg-clip-text text-transparent bg-gradient-to-r from-pink-200 to-indigo-200">
            Love&DeepCode
          </span>
        </div>

        {/* Hero Section */}
        <div className="text-center max-w-5xl mx-auto space-y-8">
          <div className="inline-block px-4 py-1.5 rounded-full border border-pink-500/30 bg-pink-500/10 text-pink-300 text-sm font-medium mb-4 backdrop-blur-sm animate-fade-in">
             ✨ Powered by Google Gemini 2.0
          </div>
          
          <h1 className="text-6xl md:text-8xl font-bold tracking-tight leading-tight animate-slide-up">
            Fall in love with <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-pink-400 via-purple-400 to-indigo-400">
              AIML algorithms
            </span>
          </h1>
          
          <p className="text-xl md:text-2xl text-gray-400 max-w-2xl mx-auto font-light leading-relaxed animate-slide-up delay-100">
            The otome-inspired coding coach. Date your way to a PhD-level offer by mastering RL, Transformers, and System Design.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-8 animate-slide-up delay-200">
            <button 
              onClick={onStart}
              className="px-8 py-4 bg-white text-slate-900 rounded-full font-bold text-lg hover:bg-gray-100 transition-all transform hover:scale-105 shadow-[0_0_20px_rgba(255,255,255,0.3)] flex items-center justify-center gap-2"
            >
              Start Your Route
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"></path></svg>
            </button>
            <a 
              href="https://github.com/google-gemini/cookbook" 
              target="_blank"
              rel="noreferrer"
              className="px-8 py-4 bg-slate-800/50 border border-slate-700 text-white rounded-full font-medium text-lg hover:bg-slate-800 transition-all backdrop-blur-md flex items-center justify-center gap-2"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
              View on GitHub
            </a>
          </div>
        </div>

        {/* Feature Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-20 w-full max-w-6xl animate-fade-in delay-300">
          {[
            { 
              title: "Voice Calls", 
              desc: "Late night voice conversations with your AI interviewer. Get feedback on your verbal delivery.",
              icon: "📞",
              color: "from-pink-500/20 to-rose-500/5",
              border: "border-pink-500/20"
            },
            { 
              title: "Coding Dates", 
              desc: "Solve hard problems together. Curated list of ML challenges with an integrated code editor.",
              icon: "💌",
              color: "from-indigo-500/20 to-blue-500/5",
              border: "border-indigo-500/20"
            },
            { 
              title: "Photo Mode", 
              desc: "Snapshots of your success. Generate architecture diagrams and explainer videos using Gemini & Veo.",
              icon: "📸",
              color: "from-purple-500/20 to-fuchsia-500/5",
              border: "border-purple-500/20"
            }
          ].map((feature, idx) => (
            <div key={idx} className={`p-8 rounded-3xl bg-gradient-to-br ${feature.color} border ${feature.border} hover:border-white/20 transition-all hover:-translate-y-2 backdrop-blur-md group`}>
              <div className="text-5xl mb-6 group-hover:scale-110 transition-transform duration-300">{feature.icon}</div>
              <h3 className="text-2xl font-bold mb-3 text-white group-hover:text-pink-300 transition-colors">{feature.title}</h3>
              <p className="text-gray-400 text-sm leading-relaxed">{feature.desc}</p>
            </div>
          ))}
        </div>

        {/* Footer info */}
        <div className="absolute bottom-6 text-center w-full text-xs text-gray-500">
            Love&DeepCode • Powered by Gemini 2.0 • Made with 💖
        </div>

      </div>
      
      {/* Custom Styles for Animation */}
      <style>{`
        @keyframes fade-in {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slide-up {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in { animation: fade-in 1s ease-out forwards; }
        .animate-slide-up { animation: slide-up 0.8s ease-out forwards; }
        .delay-100 { animation-delay: 0.1s; }
        .delay-200 { animation-delay: 0.2s; }
        .delay-300 { animation-delay: 0.3s; }
      `}</style>
    </div>
  );
};

export default WelcomePage;