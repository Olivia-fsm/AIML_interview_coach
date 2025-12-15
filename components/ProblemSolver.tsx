import React, { useState, useEffect } from 'react';
import { Problem, CodeFeedback, TestCaseResult } from '../types';
import { evaluateCodeSubmission, explainCodeSnippet, evaluateCodeExplanation, runCodeAgainstTests } from '../services/gemini';
import Editor from 'react-simple-code-editor';
import Prism from 'prismjs';
import 'prismjs/components/prism-python';

interface Props {
  problem: Problem;
  onBack: () => void;
  onSubmitSuccess: (problemId: string, code: string, feedback: CodeFeedback) => void;
  isLiked: boolean;
  onToggleLike: () => void;
}

const ProblemSolver: React.FC<Props> = ({ problem, onBack, onSubmitSuccess, isLiked, onToggleLike }) => {
  const [code, setCode] = useState(problem.starterCode);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [feedback, setFeedback] = useState<CodeFeedback | null>(null);
  const [showSolution, setShowSolution] = useState(false);

  // Tools State
  const [selection, setSelection] = useState<{start: number, end: number} | null>(null);
  const [activeTool, setActiveTool] = useState<'explain' | 'verify' | null>(null);
  const [toolLoading, setToolLoading] = useState(false);
  const [toolResult, setToolResult] = useState<any>(null);
  const [userExplanationInput, setUserExplanationInput] = useState('');
  
  // Test Console State
  const [isRunningTests, setIsRunningTests] = useState(false);
  const [testResults, setTestResults] = useState<TestCaseResult[] | null>(null);
  const [showConsole, setShowConsole] = useState(true);

  // Update code when the problem changes
  useEffect(() => {
    setCode(problem.starterCode);
    setFeedback(null);
    setShowSolution(false);
    resetToolState();
    setTestResults(null);
  }, [problem]);

  const resetToolState = () => {
    setSelection(null);
    setActiveTool(null);
    setToolResult(null);
    setToolLoading(false);
    setUserExplanationInput('');
  };

  const handleSelectionChange = (e: any) => {
    // This handler captures selection from the textarea
    if (e.target) {
        const { selectionStart, selectionEnd } = e.target;
        if (selectionEnd > selectionStart) {
            setSelection({ start: selectionStart, end: selectionEnd });
        } else {
            // Only clear selection if we are not actively using a tool
            // This prevents the tool UI from disappearing if user clicks inside modal
            if (!activeTool) {
               setSelection(null);
            }
        }
    }
  };

  const getSelectedCode = () => {
    if (!selection) return '';
    return code.substring(selection.start, selection.end);
  };

  const handleExplain = async () => {
    const snippet = getSelectedCode();
    if (!snippet) return;
    
    setActiveTool('explain');
    setToolLoading(true);
    setToolResult(null);
    try {
        const explanation = await explainCodeSnippet(snippet, code, problem.title);
        setToolResult(explanation);
    } catch (e) {
        console.error(e);
        setToolResult("Failed to explain code.");
    } finally {
        setToolLoading(false);
    }
  };

  const handleVerify = async () => {
    if (!userExplanationInput) return;
    const snippet = getSelectedCode();
    if (!snippet) return;
    
    setToolLoading(true);
    try {
        const result = await evaluateCodeExplanation(snippet, userExplanationInput, code, problem.title);
        setToolResult(result);
    } catch (e) {
        console.error(e);
        setToolResult({ isCorrect: false, score: 0, feedback: "Error verifying explanation." });
    } finally {
        setToolLoading(false);
    }
  };

  const handleRunTests = async () => {
    setIsRunningTests(true);
    setShowConsole(true);
    setTestResults(null);
    try {
        const results = await runCodeAgainstTests(
            code, 
            problem.examples, 
            problem.title, 
            problem.hiddenTestCase // Pass hidden test case
        );
        setTestResults(results);
    } catch (e) {
        console.error(e);
        // We can't easily show an error toast here without adding more UI state, 
        // but we can set testResults to an error state.
    } finally {
        setIsRunningTests(false);
    }
  };

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

  const highlight = (code: string) => (
    Prism.highlight(code, Prism.languages.python || Prism.languages.extend('clike', {}), 'python')
  );

  return (
    <div className="flex flex-col h-full bg-app-bg">
      {/* Header */}
      <div className="bg-panel-bg border-b border-border-col px-6 py-4 flex items-center justify-between">
         <div className="flex items-center gap-4">
             <button onClick={onBack} className="text-text-muted hover:text-text-main transition-colors">
                ← Back
             </button>
             <h2 className="text-xl font-bold text-text-main">{problem.title}</h2>
             <span className="text-xs text-text-muted border border-border-col px-2 py-1 rounded">{problem.category}</span>
         </div>
         <button 
            onClick={onToggleLike}
            className={`p-2 rounded-full hover:bg-card-bg transition-colors ${isLiked ? 'text-red-500' : 'text-text-muted'}`}
            title={isLiked ? "Unlike Problem" : "Like Problem"}
         >
            <svg className="w-6 h-6" fill={isLiked ? "currentColor" : "none"} stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
            </svg>
         </button>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Problem Description */}
        <div className="w-1/3 bg-card-bg border-r border-border-col p-6 overflow-y-auto">
            <h3 className="text-lg font-bold text-text-main mb-4">Description</h3>
            <p className="text-text-muted mb-6 leading-relaxed">{problem.description}</p>
            
            <h4 className="text-sm font-bold text-text-muted uppercase tracking-wider mb-2">Examples</h4>
            <div className="space-y-3 mb-8">
                {problem.examples.map((ex, i) => (
                    <div key={i} className="bg-app-bg/50 p-3 rounded-lg text-sm border border-border-col">
                        <div className="text-text-muted mb-1">Input: <span className="text-green-500 font-mono">{ex.input}</span></div>
                        <div className="text-text-muted">Output: <span className="text-blue-400 font-mono">{ex.output}</span></div>
                    </div>
                ))}
            </div>

            <details className="group mb-8">
                <summary className="cursor-pointer text-text-muted hover:text-primary transition-colors font-medium">
                    Show Hints
                </summary>
                <ul className="mt-2 pl-4 list-disc text-text-muted text-sm space-y-1">
                    {problem.hints.map((hint, i) => <li key={i}>{hint}</li>)}
                </ul>
            </details>

            <button
                onClick={() => setShowSolution(!showSolution)}
                className="w-full py-2 border border-border-col rounded text-text-muted hover:text-text-main hover:bg-panel-bg transition-all text-sm"
            >
                {showSolution ? 'Hide Solution' : 'Reveal Solution'}
            </button>
            
            {showSolution && (
                <div className="mt-4 p-4 bg-app-bg rounded border border-border-col overflow-x-auto">
                    <pre className="text-sm font-mono text-green-400 whitespace-pre-wrap">{problem.solution}</pre>
                </div>
            )}
        </div>

        {/* Right Column: Code Editor + Test Console */}
        <div className="flex-1 flex flex-col bg-[#1e1e1e] relative min-w-0"> 
            
            {/* Context Toolbar */}
            {selection && (
                <div className="bg-panel-bg border-b border-border-col p-2 flex items-center justify-between px-4 animate-slide-up z-20">
                    <div className="text-xs text-text-muted flex items-center gap-2">
                        <span className="bg-primary/20 text-primary px-2 py-1 rounded">Text Selected</span>
                        <span className="font-mono">{selection.end - selection.start} chars</span>
                    </div>
                    <div className="flex gap-2">
                        <button 
                            onClick={handleExplain}
                            disabled={toolLoading}
                            className="text-xs flex items-center gap-1 bg-card-bg hover:bg-border-col border border-border-col px-3 py-1.5 rounded transition-colors text-text-main"
                        >
                            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                            Explain This
                        </button>
                        <button 
                            onClick={() => { setActiveTool('verify'); setToolResult(null); }}
                            disabled={toolLoading}
                            className="text-xs flex items-center gap-1 bg-card-bg hover:bg-border-col border border-border-col px-3 py-1.5 rounded transition-colors text-text-main"
                        >
                             <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                            Verify My Understanding
                        </button>
                        <button 
                            onClick={resetToolState} 
                            className="text-text-muted hover:text-white px-2"
                        >
                            ✕
                        </button>
                    </div>
                </div>
            )}

            {/* Editor Area - Fixed Scroll */}
            <div className="flex-1 relative overflow-y-auto custom-scrollbar bg-[#1e1e1e] min-h-0">
                <Editor
                    value={code}
                    onValueChange={setCode}
                    highlight={highlight}
                    padding={24}
                    className="prism-editor font-mono"
                    style={{
                        fontFamily: '"Fira Code", "Fira Mono", monospace',
                        fontSize: 14,
                        backgroundColor: '#1e1e1e', // Match VS Code dark
                        color: '#d4d4d4',
                        minHeight: '100%'
                    }}
                    textareaClassName="focus:outline-none"
                    // Pass props to underlying textarea to capture selection
                    textareaId="code-textarea"
                    onSelect={handleSelectionChange}
                    onClick={handleSelectionChange}
                    onKeyUp={handleSelectionChange}
                />
            </div>

            {/* Test Console (Bottom Panel) */}
            <div className={`bg-panel-bg border-t border-border-col flex flex-col transition-all duration-300 ${showConsole ? 'h-64' : 'h-12'}`}>
                {/* Console Header */}
                <div className="flex items-center justify-between px-4 py-2 border-b border-border-col bg-card-bg">
                    <div className="flex items-center gap-2">
                         <button 
                            onClick={() => setShowConsole(!showConsole)}
                            className="text-text-muted hover:text-text-main text-xs font-bold uppercase flex items-center gap-1"
                        >
                            {showConsole ? '▼' : '▲'} Test Console
                        </button>
                    </div>
                    <div className="flex gap-2">
                        <button
                            onClick={handleRunTests}
                            disabled={isRunningTests || isSubmitting}
                            className="bg-card-bg hover:bg-border-col text-text-main text-xs font-bold py-1.5 px-4 rounded border border-border-col transition-colors"
                        >
                            {isRunningTests ? 'Running on Python Kernel...' : 'Run Tests'}
                        </button>
                        <button
                            onClick={handleSubmit}
                            disabled={isRunningTests || isSubmitting}
                            className="bg-green-600 hover:bg-green-500 text-white text-xs font-bold py-1.5 px-4 rounded transition-colors"
                        >
                            {isSubmitting ? 'Submitting...' : 'Submit Solution'}
                        </button>
                    </div>
                </div>
                
                {/* Console Content */}
                {showConsole && (
                    <div className="flex-1 overflow-y-auto p-4 font-mono text-sm bg-[#1e1e1e]">
                        {isRunningTests ? (
                             <div className="text-text-muted animate-pulse">
                                <span className="block mb-2 text-primary">Connected to Python Execution Environment...</span>
                                <span className="block">Running test cases against your code...</span>
                             </div>
                        ) : testResults ? (
                             <div className="space-y-4">
                                 {testResults.map((result, idx) => (
                                     <div key={idx} className="border-b border-white/10 pb-3 last:border-0 last:pb-0">
                                         <div className="flex items-center gap-2 mb-2">
                                             <span className={`w-2 h-2 rounded-full ${result.passed ? 'bg-green-500' : 'bg-red-500'}`}></span>
                                             <span className={result.passed ? 'text-green-400' : 'text-red-400'}>
                                                 {result.isHidden ? 'Hidden Test Case' : `Test Case ${idx + 1}`}: {result.passed ? 'Passed' : 'Failed'}
                                             </span>
                                         </div>
                                         {!result.isHidden && (
                                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                                                <div>
                                                    <div className="text-text-muted mb-1">Input</div>
                                                    <div className="bg-white/5 p-2 rounded text-gray-300 whitespace-pre-wrap font-mono">{result.input}</div>
                                                </div>
                                                <div>
                                                    <div className="text-text-muted mb-1">Expected Output</div>
                                                    <div className="bg-white/5 p-2 rounded text-gray-300 whitespace-pre-wrap font-mono">{result.expected}</div>
                                                </div>
                                            </div>
                                         )}
                                         {result.isHidden && !result.passed && (
                                             <div className="text-xs text-text-muted bg-red-500/10 p-2 rounded border border-red-500/20">
                                                 Hidden test case failed. Check edge cases.
                                             </div>
                                         )}
                                         {!result.passed && !result.isHidden && (
                                              <div className="mt-2">
                                                  <div className="text-text-muted mb-1 text-xs">Actual Output</div>
                                                  <div className="bg-red-500/10 border border-red-500/20 p-2 rounded text-red-200 text-xs whitespace-pre-wrap font-mono">
                                                      {result.actual}
                                                      {result.logs && <div className="mt-2 pt-2 border-t border-red-500/20 opacity-75">{result.logs}</div>}
                                                  </div>
                                              </div>
                                         )}
                                     </div>
                                 ))}
                             </div>
                        ) : (
                             <div className="text-text-muted opacity-50 italic">
                                 Click "Run Tests" to verify your solution against the examples.
                             </div>
                        )}
                    </div>
                )}
            </div>
            
            {/* Tool Overlays */}
            {activeTool && (
                 <div className="absolute top-12 right-6 w-96 bg-panel-bg border border-border-col rounded-xl shadow-2xl z-20 flex flex-col max-h-[80%] animate-fade-in">
                      <div className="flex justify-between items-center p-4 border-b border-border-col">
                          <h3 className="font-bold text-text-main">
                              {activeTool === 'explain' ? 'AI Explanation' : 'Verify Understanding'}
                          </h3>
                          <button onClick={resetToolState} className="text-text-muted hover:text-white">✕</button>
                      </div>
                      
                      <div className="p-4 overflow-y-auto custom-scrollbar flex-1">
                          {toolLoading ? (
                               <div className="flex flex-col items-center justify-center py-8 text-text-muted">
                                   <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mb-2"></div>
                                   <p className="text-sm">Thinking...</p>
                               </div>
                          ) : (
                              activeTool === 'explain' ? (
                                  <div className="prose prose-invert prose-sm">
                                      {toolResult ? (
                                          <div dangerouslySetInnerHTML={{ __html: toolResult.replace(/\n/g, '<br/>') }} />
                                      ) : (
                                          <p className="text-text-muted">Waiting for results...</p>
                                      )}
                                  </div>
                              ) : (
                                  <div className="space-y-4">
                                      {toolResult ? (
                                          <div className="space-y-3">
                                              <div className="flex items-center justify-between">
                                                  <span className="text-sm font-medium text-text-muted">Score</span>
                                                  <span className={`text-xl font-bold ${toolResult.score > 70 ? 'text-green-500' : 'text-yellow-500'}`}>
                                                      {toolResult.score}/100
                                                  </span>
                                              </div>
                                              <div className={`p-3 rounded border text-sm ${toolResult.isCorrect ? 'bg-green-500/10 border-green-500/30 text-green-200' : 'bg-red-500/10 border-red-500/30 text-red-200'}`}>
                                                  {toolResult.isCorrect ? 'Correct Understanding' : 'Misconception Detected'}
                                              </div>
                                              <p className="text-sm text-text-muted leading-relaxed">
                                                  {toolResult.feedback}
                                              </p>
                                              <button 
                                                  onClick={() => { setToolResult(null); setUserExplanationInput(''); }}
                                                  className="w-full py-2 bg-card-bg hover:bg-border-col text-sm rounded border border-border-col text-text-main"
                                              >
                                                  Try Again
                                              </button>
                                          </div>
                                      ) : (
                                          <>
                                              <p className="text-xs text-text-muted">
                                                  Explain the selected code in your own words. The AI will check if you are correct.
                                              </p>
                                              <div className="bg-black/30 p-2 rounded border border-border-col mb-2">
                                                  <pre className="text-xs font-mono text-gray-400 overflow-x-auto whitespace-pre-wrap max-h-20">
                                                      {getSelectedCode()}
                                                  </pre>
                                              </div>
                                              <textarea
                                                  value={userExplanationInput}
                                                  onChange={(e) => setUserExplanationInput(e.target.value)}
                                                  placeholder="This code calculates..."
                                                  className="w-full h-32 bg-card-bg border border-border-col rounded p-2 text-sm text-text-main focus:ring-1 focus:ring-primary outline-none resize-none"
                                              />
                                              <button
                                                  onClick={handleVerify}
                                                  disabled={!userExplanationInput.trim()}
                                                  className="w-full py-2 bg-primary hover:opacity-90 text-white text-sm font-bold rounded disabled:opacity-50"
                                              >
                                                  Submit Explanation
                                              </button>
                                          </>
                                      )}
                                  </div>
                              )
                          )}
                      </div>
                 </div>
            )}
        </div>
      </div>

      {/* Feedback Modal / Overlay */}
      {feedback && (
          <div className="absolute inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-8">
              <div className="bg-panel-bg border border-border-col rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto shadow-2xl p-8">
                  <div className="flex justify-between items-center mb-6">
                      <h2 className="text-2xl font-bold text-text-main">Analysis Result</h2>
                      <button onClick={() => setFeedback(null)} className="text-text-muted hover:text-text-main">✕</button>
                  </div>
                  
                  <div className="flex items-center gap-6 mb-8">
                      <div className={`text-5xl font-bold ${feedback.correctnessScore > 80 ? 'text-green-500' : 'text-yellow-400'}`}>
                          {feedback.correctnessScore}/100
                      </div>
                      <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <span className="text-text-muted text-sm w-24">Correctness:</span>
                            <span className={feedback.isCorrect ? 'text-green-500' : 'text-red-400'}>
                                {feedback.isCorrect ? 'Passed' : 'Failed Logic'}
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-text-muted text-sm w-24">Time Comp:</span>
                            <span className="text-text-main font-mono text-sm">{feedback.timeComplexity}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-text-muted text-sm w-24">Space Comp:</span>
                            <span className="text-text-main font-mono text-sm">{feedback.spaceComplexity}</span>
                          </div>
                      </div>
                  </div>

                  <div className="mb-6">
                      <h4 className="font-bold text-text-main mb-2">Analysis</h4>
                      <p className="text-text-muted leading-relaxed bg-card-bg/50 p-4 rounded-lg border border-border-col">{feedback.analysis}</p>
                  </div>

                  {feedback.improvements.length > 0 && (
                      <div className="mb-6">
                          <h4 className="font-bold text-text-main mb-2">Suggested Improvements</h4>
                          <ul className="list-disc pl-5 space-y-2 text-text-muted">
                              {feedback.improvements.map((imp, i) => (
                                  <li key={i}>{imp}</li>
                              ))}
                          </ul>
                      </div>
                  )}

                  <div>
                      <h4 className="font-bold text-text-main mb-2">Reference Solution</h4>
                      <div className="p-4 bg-app-bg rounded-lg border border-border-col overflow-x-auto">
                          <pre className="text-sm font-mono text-green-400 whitespace-pre-wrap">{problem.solution}</pre>
                      </div>
                  </div>

                  <div className="mt-8 flex justify-end">
                      <button 
                        onClick={() => { setFeedback(null); onBack(); }}
                        className="bg-primary hover:opacity-90 text-white font-bold py-2 px-6 rounded-lg"
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