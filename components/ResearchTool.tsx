import React, { useState } from 'react';
import { researchTopic } from '../services/gemini';

const ResearchTool: React.FC = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<{text: string, sources: any[]} | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    try {
      const data = await researchTopic(query);
      setResults(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 h-full overflow-y-auto">
        <div className="max-w-3xl mx-auto">
            <h2 className="text-2xl font-bold text-white mb-6">Market Research & Grounding</h2>
            
            <form onSubmit={handleSearch} className="mb-8">
                <div className="flex gap-2">
                    <input 
                        type="text" 
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="e.g. Recent interview questions for Google Research Scientist 2024"
                        className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-white focus:ring-2 focus:ring-primary"
                    />
                    <button 
                        type="submit" 
                        disabled={loading}
                        className="px-6 py-3 bg-primary hover:bg-blue-600 text-white font-bold rounded-lg transition-colors"
                    >
                        {loading ? 'Searching...' : 'Search'}
                    </button>
                </div>
            </form>

            {results && (
                <div className="space-y-6 animate-fade-in">
                    <div className="bg-secondary p-6 rounded-xl border border-gray-700">
                        <div className="prose prose-invert max-w-none">
                            {/* Simple markdown rendering or text display */}
                            <p className="whitespace-pre-wrap text-gray-200 leading-relaxed">{results.text}</p>
                        </div>
                    </div>

                    {results.sources.length > 0 && (
                        <div>
                            <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-3">Sources</h3>
                            <div className="grid gap-3">
                                {results.sources.map((source, i) => (
                                    <a 
                                        key={i} 
                                        href={source.uri} 
                                        target="_blank" 
                                        rel="noreferrer"
                                        className="block p-3 bg-gray-800/50 hover:bg-gray-800 border border-gray-700 rounded-lg transition-colors"
                                    >
                                        <div className="font-medium text-blue-400 truncate">{source.title}</div>
                                        <div className="text-xs text-gray-500 truncate">{source.uri}</div>
                                    </a>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    </div>
  );
};

export default ResearchTool;
