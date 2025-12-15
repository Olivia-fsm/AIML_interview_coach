import React from 'react';
import { ThemeId } from '../types';

interface Props {
  onSelectTheme: (theme: ThemeId) => void;
}

const ThemeSelector: React.FC<Props> = ({ onSelectTheme }) => {
  const themes: { id: ThemeId; name: string; desc: string; colors: string[] }[] = [
    { 
      id: 'midnight', 
      name: 'Midnight Pro', 
      desc: 'The classic dark mode for focused coding.',
      colors: ['bg-[#0f172a]', 'bg-[#1e293b]', 'bg-[#3b82f6]']
    },
    { 
      id: 'solar', 
      name: 'Solar Light', 
      desc: 'Clean, bright, and professional.',
      colors: ['bg-[#f8fafc]', 'bg-[#ffffff]', 'bg-[#2563eb]']
    },
    { 
      id: 'neon', 
      name: 'Cyberpunk', 
      desc: 'High contrast neon for the night owls.',
      colors: ['bg-[#000000]', 'bg-[#121212]', 'bg-[#00ff41]']
    },
    { 
      id: 'deepspace', 
      name: 'Deepspace', 
      desc: 'Dreamy violets for a romantic sci-fi vibe.',
      colors: ['bg-[#1a0b2e]', 'bg-[#2e1065]', 'bg-[#d946ef]']
    },
    { 
      id: 'toon', 
      name: 'Toon World', 
      desc: 'Playful comics style with bold borders.',
      colors: ['bg-[#fef3c7]', 'bg-[#ffffff]', 'bg-[#f97316]']
    },
    { 
      id: 'cosmic', 
      name: 'Cosmic Universe', 
      desc: 'Interactive planets and moving stars.',
      colors: ['bg-[#020617]', 'bg-[#1e3a8a]', 'bg-[#38bdf8]']
    },
    { 
      id: 'sea', 
      name: 'Deep Sea', 
      desc: 'Calming bubbles and ocean depths.',
      colors: ['bg-[#001e3c]', 'bg-[#0f766e]', 'bg-[#2dd4bf]']
    },
    { 
      id: 'flower', 
      name: 'Forget-Me-Not', 
      desc: 'Gentle flowers and floating petals.',
      colors: ['bg-[#f0fdf4]', 'bg-[#dcfce7]', 'bg-[#60a5fa]']
    },
    { 
      id: 'snow', 
      name: 'Winter Snow', 
      desc: 'Peaceful snowfall with wind interaction.',
      colors: ['bg-[#1e293b]', 'bg-[#334155]', 'bg-[#60a5fa]']
    },
    { 
      id: 'gothic', 
      name: 'Gothic Castle', 
      desc: 'Dark atmosphere with feathers and electricity.',
      colors: ['bg-[#0f0505]', 'bg-[#2a0a0a]', 'bg-[#9f1239]']
    },
    { 
      id: 'christmas', 
      name: 'Christmas Eve', 
      desc: 'Festive red & green with a tree and reindeer.',
      colors: ['bg-[#022c22]', 'bg-[#064e3b]', 'bg-[#ef4444]']
    }
  ];

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-6 bg-gray-900 animate-fade-in bg-[url('https://images.unsplash.com/photo-1534796636912-3b95b3ab5986?q=80&w=2342&auto=format&fit=crop')] bg-cover bg-center relative">
      <div className="absolute inset-0 bg-gray-900/90 backdrop-blur-sm"></div>
      
      <div className="relative z-10 text-center mb-12">
        <h1 className="text-6xl font-bold text-white mb-4 drop-shadow-lg flex items-center justify-center gap-3">
           Love<span className="text-pink-500">&</span>Deep<span className="text-indigo-400">Code</span>
        </h1>
        <p className="text-gray-300 text-xl font-light tracking-wide">Choose your interface dimension.</p>
      </div>

      <div className="relative z-10 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl w-full">
        {themes.map((theme) => (
          <button
            key={theme.id}
            onClick={() => onSelectTheme(theme.id)}
            className="group relative overflow-hidden rounded-3xl border border-gray-700 hover:border-white/50 transition-all duration-300 hover:-translate-y-2 hover:shadow-2xl text-left bg-gray-800/80 backdrop-blur-md"
          >
            {/* Preview Header */}
            <div className={`h-32 w-full ${theme.colors[0]} relative p-4 flex flex-col justify-end border-b border-gray-700/50`}>
               <div className="flex gap-2 mb-2">
                  <div className={`w-8 h-8 rounded-full shadow-lg ${theme.colors[1]} ring-2 ring-white/10`}></div>
                  <div className={`w-8 h-8 rounded-full shadow-lg ${theme.colors[2]} ring-2 ring-white/10`}></div>
               </div>
            </div>
            
            <div className="p-6">
               <h3 className="text-2xl font-bold text-white mb-2 group-hover:text-blue-400 transition-colors">{theme.name}</h3>
               <p className="text-gray-400 text-sm leading-relaxed">{theme.desc}</p>
            </div>

            {/* Selection Indicator */}
            <div className="absolute top-4 right-4 w-8 h-8 rounded-full border-2 border-white/20 group-hover:bg-blue-500 group-hover:border-blue-500 transition-all flex items-center justify-center">
                <div className="w-2 h-2 bg-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity"></div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

export default ThemeSelector;