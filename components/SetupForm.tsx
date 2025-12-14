import React, { useState } from 'react';
import { generateStudyPlan } from '../services/gemini';
import { PrepPlan } from '../types';

interface Props {
  onPlanGenerated: (plan: PrepPlan) => void;
  onSkipToLibrary: () => void;
}

const SetupForm: React.FC<Props> = ({ onPlanGenerated, onSkipToLibrary }) => {
  const [loading, setLoading] = useState(false);
  const [jobDesc, setJobDesc] = useState('');
  const [topics, setTopics] = useState('');
  const [date, setDate] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const plan = await generateStudyPlan(jobDesc, topics, date);
      onPlanGenerated(plan);
    } catch (error) {
      console.error(error);
      alert('Failed to generate plan. Please check your inputs and try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto p-8 mt-10 bg-panel-bg rounded-2xl shadow-xl border border-border-col">
      <h2 className="text-3xl font-bold mb-6 text-primary">
        AI Interview Prep Setup
      </h2>
      <p className="mb-8 text-text-muted">
        Enter your target role details and interview date. Our Gemini-powered planner will create a custom curriculum for you.
      </p>
      
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label className="block text-sm font-medium text-text-muted mb-2">Job Description / Requirements</label>
          <textarea
            required
            className="w-full h-32 bg-card-bg border border-border-col rounded-lg p-3 text-text-main focus:ring-2 focus:ring-primary focus:border-transparent transition"
            placeholder="Paste the JD here... (e.g., Senior ML Engineer, proficiency in PyTorch, Transformers, System Design...)"
            value={jobDesc}
            onChange={(e) => setJobDesc(e.target.value)}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-text-muted mb-2">Specific Focus Topics</label>
          <input
            type="text"
            className="w-full bg-card-bg border border-border-col rounded-lg p-3 text-text-main focus:ring-2 focus:ring-primary focus:border-transparent transition"
            placeholder="e.g., Diffusion Models, LLM Finetuning, CUDA optimization"
            value={topics}
            onChange={(e) => setTopics(e.target.value)}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-text-muted mb-2">Interview Date</label>
          <input
            type="text"
            required
            className="w-full bg-card-bg border border-border-col rounded-lg p-3 text-text-main focus:ring-2 focus:ring-primary focus:border-transparent transition"
            placeholder="e.g., Next Friday, October 15th, or in 2 weeks"
            value={date}
            onChange={(e) => setDate(e.target.value)}
          />
        </div>

        <div className="flex flex-col gap-4 pt-2">
            <button
            type="submit"
            disabled={loading}
            className={`w-full py-4 rounded-lg font-bold text-lg text-white transition-all transform hover:scale-[1.01] ${
                loading ? 'bg-gray-600 cursor-not-allowed' : 'bg-primary hover:opacity-90 shadow-lg'
            }`}
            >
            {loading ? (
                <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Generating Personal Plan...
                </span>
            ) : (
                'Generate Preparation Plan'
            )}
            </button>

            <div className="relative flex items-center py-2">
                <div className="flex-grow border-t border-border-col"></div>
                <span className="flex-shrink-0 mx-4 text-text-muted text-sm">OR</span>
                <div className="flex-grow border-t border-border-col"></div>
            </div>

            <button
                type="button"
                onClick={onSkipToLibrary}
                className="w-full py-3 rounded-lg font-medium text-text-muted border border-border-col hover:bg-card-bg hover:text-text-main transition-all"
            >
                Browse Problem Library Only
            </button>
        </div>
      </form>
    </div>
  );
};

export default SetupForm;