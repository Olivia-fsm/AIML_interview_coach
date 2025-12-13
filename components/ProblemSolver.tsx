import React, { useState, useEffect } from 'react';
import { Problem, CodeFeedback } from '../types';
import { evaluateCodeSubmission } from '../services/gemini';

interface Props {
  problem: Problem;
  onBack: () => void;
  onSubmitSuccess: (problemId: string, code: string, feedback: CodeFeedback) => void;
}

const ProblemSolver: React.FC<Props> = ({ problem, onBack, onSubmitSuccess }) => {
  const [code, setCode] = useState(problem.starterCode);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [feedback, setFeedback] = useState<CodeFeedback | null>(null);
  const [showSolution, setShowSolution] = useState(false);

  // Update code when the problem changes
  useEffect(() => {
    setCode(problem.starterCode);
    setFeedback(null);
    setShowSolution(false);
  }, [problem]);

  const handleSubmit = async () => {
    setIsSubmitting(true);
    setFeedback(null);
    try {
      const result = await evaluateCodeSubmission(problem.title, problem.description, code);
      setFeedback(result);
      onSubmitSuccess(problem.id, code, result);
    } catch (e) {
      console.error(e);
      alert("Failed to submit code. Check console.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#0f172a]">
      {/* Header */}
      <div className="bg-secondary border-b border-gray-700 px-6 py-4 flex items-center justify-between">
         <div className="flex items-center gap-4">
             <button onClick={onBack} className="text-gray-400 hover:text-white transition-colors">
                ← Back
             </button>
             <h2 className="text-xl font-bold text-white">{problem.title}</h2>
             <span className="text-xs text-gray-500 border border-gray-700 px-2 py-1 rounded">{problem.category}</span>
         </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Problem Description */}
        <div className="w-1/3 bg-gray-900 border-r border-gray-800 p-6 overflow-y-auto">
            <h3 className="text-lg font-bold text-white mb-4">Description</h3>
            <p className="text-gray-300 mb-6 leading-relaxed">{problem.description}</p>
            
            <h4 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">Examples</h4>
            <div className="space-y-3 mb-8">
                {problem.examples.map((ex, i) => (
                    <div key={i} className="bg-black/30 p-3 rounded-lg text-sm border border-gray-800">
                        <div className="text-gray-400 mb-1">Input: <span className="text-green-400 font-mono">{ex.input}</span></div>
                        <div className="text-gray-400">Output: <span className="text-blue-400 font-mono">{ex.output}</span></div>
                    </div>
                ))}
            </div>

            <details className="group mb-8">
                <summary className="cursor-pointer text-gray-400 hover:text-primary transition-colors font-medium">
                    Show Hints
                </summary>
                <ul className="mt-2 pl-4 list-disc text-gray-500 text-sm space-y-1">
                    {problem.hints.map((hint, i) => <li key={i}>{hint}</li>)}
                </ul>
            </details>

            <button
                onClick={() => setShowSolution(!showSolution)}
                className="w-full py-2 border border-gray-600 rounded text-gray-400 hover:text-white hover:bg-gray-800 transition-all text-sm"
            >
                {showSolution ? 'Hide Solution' : 'Reveal Solution'}
            </button>
            
            {showSolution && (
                <div className="mt-4 p-4 bg-[#1e1e1e] rounded border border-gray-700 overflow-x-auto">
                    <pre className="text-sm font-mono text-green-400 whitespace-pre-wrap">{problem.solution}</pre>
                </div>
            )}
        </div>

        {/* Code Editor Area */}
        <div className="flex-1 flex flex-col bg-[#1e1e1e]">
            <textarea
                value={code}
                onChange={(e) => setCode(e.target.value)}
                className="flex-1 w-full bg-[#1e1e1e] text-gray-200 font-mono p-6 resize-none focus:outline-none"
                spellCheck={false}
            />
            
            <div className="bg-gray-900 p-4 border-t border-gray-800 flex justify-end">
                <button
                    onClick={handleSubmit}
                    disabled={isSubmitting}
                    className="bg-green-600 hover:bg-green-500 text-white font-bold py-2 px-6 rounded-lg transition-colors flex items-center gap-2"
                >
                    {isSubmitting ? (
                        <>
                            <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                            Analyzing...
                        </>
                    ) : (
                        'Submit Solution'
                    )}
                </button>
            </div>
        </div>
      </div>

      {/* Feedback Modal / Overlay */}
      {feedback && (
          <div className="absolute inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-8">
              <div className="bg-secondary border border-gray-600 rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto shadow-2xl p-8">
                  <div className="flex justify-between items-center mb-6">
                      <h2 className="text-2xl font-bold text-white">Analysis Result</h2>
                      <button onClick={() => setFeedback(null)} className="text-gray-400 hover:text-white">✕</button>
                  </div>
                  
                  <div className="flex items-center gap-6 mb-8">
                      <div className={`text-5xl font-bold ${feedback.correctnessScore > 80 ? 'text-green-400' : 'text-yellow-400'}`}>
                          {feedback.correctnessScore}/100
                      </div>
                      <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <span className="text-gray-400 text-sm w-24">Correctness:</span>
                            <span className={feedback.isCorrect ? 'text-green-400' : 'text-red-400'}>
                                {feedback.isCorrect ? 'Passed' : 'Failed Logic'}
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-gray-400 text-sm w-24">Time Comp:</span>
                            <span className="text-white font-mono text-sm">{feedback.timeComplexity}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-gray-400 text-sm w-24">Space Comp:</span>
                            <span className="text-white font-mono text-sm">{feedback.spaceComplexity}</span>
                          </div>
                      </div>
                  </div>

                  <div className="mb-6">
                      <h4 className="font-bold text-white mb-2">Analysis</h4>
                      <p className="text-gray-300 leading-relaxed bg-gray-900/50 p-4 rounded-lg border border-gray-700">{feedback.analysis}</p>
                  </div>

                  {feedback.improvements.length > 0 && (
                      <div className="mb-6">
                          <h4 className="font-bold text-white mb-2">Suggested Improvements</h4>
                          <ul className="list-disc pl-5 space-y-2 text-gray-300">
                              {feedback.improvements.map((imp, i) => (
                                  <li key={i}>{imp}</li>
                              ))}
                          </ul>
                      </div>
                  )}

                  {/* Auto-show solution on success or failure if not already shown */}
                  <div>
                      <h4 className="font-bold text-white mb-2">Reference Solution</h4>
                      <div className="p-4 bg-gray-900 rounded-lg border border-gray-700 overflow-x-auto">
                          <pre className="text-sm font-mono text-green-400 whitespace-pre-wrap">{problem.solution}</pre>
                      </div>
                  </div>

                  <div className="mt-8 flex justify-end">
                      <button 
                        onClick={() => { setFeedback(null); onBack(); }}
                        className="bg-primary hover:bg-blue-600 text-white font-bold py-2 px-6 rounded-lg"
                      >
                          Continue Learning
                      </button>
                  </div>
              </div>
          </div>
      )}
    </div>
  );
};

export default ProblemSolver;