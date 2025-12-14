import React, { useState } from 'react';
import { PROBLEM_LIBRARY } from '../data/problems';
import { Problem, ProblemCategory, Submission } from '../types';

interface Props {
  onSelectProblem: (problem: Problem) => void;
  submissions: Submission[];
}

const CATEGORIES: ProblemCategory[] = [
  'Supervised Learning', 
  'Unsupervised Learning', 
  'Deep Learning', 
  'NLP', 
  'Computer Vision', 
  'Reinforcement Learning', 
  'Reasoning',
  'Architecture',
  'System Design'
];

const ProblemBank: React.FC<Props> = ({ onSelectProblem, submissions }) => {
  const [selectedCategory, setSelectedCategory] = useState<ProblemCategory | 'All'>('All');

  const filteredProblems = selectedCategory === 'All' 
    ? PROBLEM_LIBRARY 
    : PROBLEM_LIBRARY.filter(p => p.category === selectedCategory);

  const isSolved = (id: string) => submissions.some(s => s.problemId === id && s.feedback.correctnessScore > 80);

  return (
    <div className="flex h-full">
      {/* Categories Sidebar */}
      <div className="w-64 bg-panel-bg border-r border-border-col p-4 space-y-2 overflow-y-auto">
        <h3 className="text-sm font-bold text-text-muted uppercase tracking-wider mb-4">Categories</h3>
        <button 
          onClick={() => setSelectedCategory('All')}
          className={`w-full text-left px-4 py-2 rounded-lg text-sm transition-colors ${selectedCategory === 'All' ? 'bg-primary text-white' : 'text-text-muted hover:text-text-main hover:bg-card-bg'}`}
        >
          All Topics
        </button>
        {CATEGORIES.map(cat => (
           <button 
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={`w-full text-left px-4 py-2 rounded-lg text-sm transition-colors ${selectedCategory === cat ? 'bg-primary text-white' : 'text-text-muted hover:text-text-main hover:bg-card-bg'}`}
          >
            {cat}
          </button>
        ))}
      </div>

      {/* Problems List */}
      <div className="flex-1 p-8 overflow-y-auto bg-app-bg">
        <h2 className="text-3xl font-bold text-text-main mb-6">Coding Practice Library</h2>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          {filteredProblems.map(problem => (
            <div 
              key={problem.id}
              onClick={() => onSelectProblem(problem)}
              className="group bg-panel-bg border border-border-col rounded-xl p-5 hover:border-primary cursor-pointer transition-all shadow-sm"
            >
              <div className="flex justify-between items-start mb-2">
                 <h3 className="text-xl font-bold text-text-main group-hover:text-primary transition-colors">{problem.title}</h3>
                 <div className="flex gap-2">
                    {isSolved(problem.id) && (
                        <span className="bg-green-500/20 text-green-500 text-xs px-2 py-1 rounded-full font-bold">SOLVED</span>
                    )}
                    <span className={`text-xs px-2 py-1 rounded-full font-bold border ${
                        problem.difficulty === 'Easy' ? 'border-green-500 text-green-500' :
                        problem.difficulty === 'Medium' ? 'border-yellow-500 text-yellow-500' :
                        'border-red-500 text-red-500'
                    }`}>
                        {problem.difficulty}
                    </span>
                 </div>
              </div>
              <p className="text-text-muted text-sm line-clamp-2">{problem.description}</p>
              <div className="mt-4 flex items-center gap-2 text-xs text-text-muted">
                <span className="bg-card-bg px-2 py-1 rounded border border-border-col">{problem.category}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ProblemBank;