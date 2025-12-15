import React, { useState } from 'react';
import { findJobPostings } from '../services/gemini';
import { JobPosting } from '../types';

const JobBoard: React.FC = () => {
  const [role, setRole] = useState('');
  const [location, setLocation] = useState('');
  const [loading, setLoading] = useState(false);
  const [jobs, setJobs] = useState<JobPosting[] | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!role) return;
    
    setLoading(true);
    setJobs(null);
    try {
        const results = await findJobPostings(role, location);
        setJobs(results);
    } catch (err) {
        console.error(err);
    } finally {
        setLoading(false);
    }
  };

  const getPlatformColor = (platform: string) => {
      const p = platform.toLowerCase();
      if (p.includes('linkedin')) return 'bg-blue-600 text-white';
      if (p.includes('twitter') || p.includes('x')) return 'bg-black border border-gray-700 text-white';
      if (p.includes('combinator')) return 'bg-orange-500 text-white';
      return 'bg-primary/20 text-primary border border-primary/30';
  };

  return (
    <div className="h-full p-6 overflow-y-auto">
        <div className="max-w-6xl mx-auto">
            {/* Header */}
            <div className="text-center mb-10">
                <h2 className="text-3xl font-bold text-text-main mb-2">Job Hunt Intelligence</h2>
                <p className="text-text-muted">
                    Scour LinkedIn, X (Twitter), and Career Pages for real-time opportunities.
                </p>
            </div>

            {/* Search Bar */}
            <div className="bg-panel-bg p-6 rounded-2xl border border-border-col shadow-lg mb-8">
                <form onSubmit={handleSearch} className="flex flex-col md:flex-row gap-4">
                    <div className="flex-1">
                        <label className="block text-xs font-bold text-text-muted uppercase mb-1">Role / Keywords</label>
                        <input 
                            type="text" 
                            required
                            value={role}
                            onChange={(e) => setRole(e.target.value)}
                            placeholder="e.g. Research Scientist Intern, ML Engineer"
                            className="w-full bg-card-bg border border-border-col rounded-lg px-4 py-3 text-text-main focus:ring-2 focus:ring-primary outline-none"
                        />
                    </div>
                    <div className="flex-1">
                        <label className="block text-xs font-bold text-text-muted uppercase mb-1">Location (Optional)</label>
                        <input 
                            type="text" 
                            value={location}
                            onChange={(e) => setLocation(e.target.value)}
                            placeholder="e.g. New York, Remote, London"
                            className="w-full bg-card-bg border border-border-col rounded-lg px-4 py-3 text-text-main focus:ring-2 focus:ring-primary outline-none"
                        />
                    </div>
                    <div className="flex items-end">
                        <button 
                            type="submit" 
                            disabled={loading}
                            className="w-full md:w-auto px-8 py-3 bg-primary hover:opacity-90 text-white font-bold rounded-lg transition-all transform hover:scale-105 shadow-lg flex items-center justify-center gap-2"
                        >
                            {loading ? (
                                <>
                                   <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></div>
                                   Searching...
                                </>
                            ) : 'Find Jobs'}
                        </button>
                    </div>
                </form>
            </div>

            {/* Results */}
            {loading && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-pulse">
                    {[1,2,3,4,5,6].map(i => (
                        <div key={i} className="h-48 bg-panel-bg rounded-xl border border-border-col"></div>
                    ))}
                </div>
            )}

            {jobs && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-fade-in">
                    {jobs.length === 0 ? (
                        <div className="col-span-full text-center py-12 text-text-muted">
                            No listings found. Try broadening your location or keywords.
                        </div>
                    ) : (
                        jobs.map((job, idx) => (
                            <div key={idx} className="bg-panel-bg rounded-xl border border-border-col p-6 hover:border-primary transition-all shadow-sm hover:shadow-lg flex flex-col group">
                                <div className="flex justify-between items-start mb-4">
                                    <div>
                                        <h3 className="font-bold text-lg text-text-main leading-tight group-hover:text-primary transition-colors">{job.title}</h3>
                                        <div className="text-sm font-semibold text-text-muted mt-1">{job.company}</div>
                                    </div>
                                    <span className={`text-[10px] uppercase font-bold px-2 py-1 rounded ${getPlatformColor(job.platform)}`}>
                                        {job.platform}
                                    </span>
                                </div>
                                
                                <div className="flex items-center gap-2 text-xs text-text-muted mb-4">
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"/><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
                                    {job.location}
                                </div>

                                <p className="text-sm text-text-muted flex-1 mb-6 leading-relaxed">
                                    {job.summary}
                                </p>

                                <a 
                                    href={job.url || `https://www.google.com/search?q=${encodeURIComponent(job.title + " " + job.company + " job")}`}
                                    target="_blank" 
                                    rel="noreferrer"
                                    className="block w-full text-center py-2 rounded-lg bg-card-bg border border-border-col hover:bg-primary hover:text-white hover:border-primary transition-all text-sm font-bold"
                                >
                                    View Post ↗
                                </a>
                            </div>
                        ))
                    )}
                </div>
            )}
        </div>
    </div>
  );
};

export default JobBoard;