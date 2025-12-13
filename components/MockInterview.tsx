import React, { useEffect, useState, useRef } from 'react';
import { LiveClient, generateInterviewReport } from '../services/gemini';
import { InterviewTurn, InterviewReport } from '../types';

const MockInterview: React.FC = () => {
  const [isActive, setIsActive] = useState(false);
  const [status, setStatus] = useState('Disconnected');
  const [turns, setTurns] = useState<InterviewTurn[]>([]);
  const [report, setReport] = useState<InterviewReport | null>(null);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  
  const clientRef = useRef<LiveClient | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const startSession = async () => {
    try {
      setStatus('Connecting...');
      setReport(null);
      setTurns([]);
      clientRef.current = new LiveClient(
        (audioData) => {
            drawVisualizer();
        }, 
        (turn) => {
            // Append new turn to history
            setTurns(prev => [...prev, turn]);
        }
      );
      await clientRef.current.connect();
      setStatus('Live - Interviewing');
      setIsActive(true);
    } catch (e) {
      console.error(e);
      setStatus('Error connecting');
    }
  };

  const endSession = async () => {
    clientRef.current?.disconnect();
    setIsActive(false);
    setStatus('Disconnected');
    
    if (turns.length > 0) {
        setIsGeneratingReport(true);
        try {
            const result = await generateInterviewReport(turns);
            setReport(result);
        } catch (e) {
            console.error("Failed to generate report", e);
        } finally {
            setIsGeneratingReport(false);
        }
    }
  };

  const drawVisualizer = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const bars = 20;
      const width = canvas.width / bars;
      
      for(let i=0; i<bars; i++) {
          const height = Math.random() * canvas.height * 0.8;
          ctx.fillStyle = `rgba(139, 92, 246, ${Math.random() + 0.2})`;
          ctx.fillRect(i * width, (canvas.height - height) / 2, width - 2, height);
      }
  };

  useEffect(() => {
    return () => {
      if (isActive) clientRef.current?.disconnect();
    };
  }, [isActive]);

  return (
    <div className="flex flex-col items-center justify-center h-full p-8 space-y-8 overflow-y-auto">
      <div className="max-w-4xl w-full">
          {!report ? (
              <div className="bg-secondary p-8 rounded-2xl border border-gray-700 shadow-2xl relative overflow-hidden text-center">
                {/* Status Indicator */}
                <div className="absolute top-4 right-4 flex items-center gap-2">
                    <div className={`w-3 h-3 rounded-full ${isActive ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
                    <span className="text-xs text-gray-400 font-mono uppercase">{status}</span>
                </div>

                <div className="mb-8">
                    <div className="w-24 h-24 mx-auto bg-gray-800 rounded-full flex items-center justify-center mb-4 border-2 border-gray-700">
                        <span className="text-4xl">🎙️</span>
                    </div>
                    <h2 className="text-2xl font-bold text-white mb-2">AI Mock Interviewer</h2>
                    <p className="text-gray-400 max-w-lg mx-auto">
                        Experience a real-time technical interview with Gemini. 
                        Speak naturally about your experience, solve verbal coding problems, and get used to the pressure.
                    </p>
                </div>

                <div className="h-32 bg-gray-900 rounded-lg mb-8 relative flex items-center justify-center border border-gray-800 mx-auto max-w-xl">
                    {!isActive && <span className="text-gray-600 text-sm">Audio visualization will appear here</span>}
                    <canvas ref={canvasRef} width={400} height={128} className="w-full h-full rounded-lg"></canvas>
                </div>

                <div className="flex gap-4 justify-center">
                    {!isActive ? (
                        <button 
                            onClick={startSession}
                            disabled={isGeneratingReport}
                            className="px-8 py-3 bg-green-600 hover:bg-green-500 text-white rounded-full font-bold transition-all transform hover:scale-105 shadow-lg flex items-center gap-2 disabled:opacity-50"
                        >
                             {isGeneratingReport ? 'Generating Report...' : (
                                <>
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path></svg>
                                    Start Interview
                                </>
                             )}
                        </button>
                    ) : (
                        <button 
                            onClick={endSession}
                            className="px-8 py-3 bg-red-600 hover:bg-red-500 text-white rounded-full font-bold transition-all transform hover:scale-105 shadow-lg flex items-center gap-2"
                        >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                            End Session & Get Feedback
                        </button>
                    )}
                </div>
                
                {turns.length > 0 && isActive && (
                    <div className="mt-6 text-sm text-gray-500">
                        Session active • {turns.length} interaction turns recorded
                    </div>
                )}
              </div>
          ) : (
              <div className="bg-secondary rounded-2xl border border-gray-700 shadow-2xl p-8 animate-fade-in text-left">
                  <div className="flex justify-between items-start mb-8 border-b border-gray-700 pb-6">
                      <div>
                          <h2 className="text-3xl font-bold text-white mb-2">Interview Report</h2>
                          <p className="text-gray-400">Detailed breakdown of your session performance</p>
                      </div>
                      <div className="text-center bg-gray-800 p-4 rounded-xl border border-gray-700">
                          <div className={`text-4xl font-bold ${report.overallScore >= 80 ? 'text-green-400' : report.overallScore >= 60 ? 'text-yellow-400' : 'text-red-400'}`}>
                              {report.overallScore}
                          </div>
                          <div className="text-xs text-gray-500 uppercase font-bold mt-1">Overall Score</div>
                      </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
                      <div>
                          <h3 className="text-lg font-bold text-green-400 mb-3 flex items-center gap-2">
                             <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                             Strengths
                          </h3>
                          <ul className="list-disc pl-5 space-y-2 text-gray-300">
                              {report.strengths.map((s, i) => <li key={i}>{s}</li>)}
                          </ul>
                      </div>
                      <div>
                          <h3 className="text-lg font-bold text-red-400 mb-3 flex items-center gap-2">
                             <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
                             Areas for Improvement
                          </h3>
                          <ul className="list-disc pl-5 space-y-2 text-gray-300">
                              {report.weaknesses.map((w, i) => <li key={i}>{w}</li>)}
                          </ul>
                      </div>
                  </div>

                  <div className="mb-8">
                      <h3 className="text-xl font-bold text-white mb-4">Question & Answer Analysis</h3>
                      <div className="space-y-6">
                          {report.qna.map((item, idx) => (
                              <div key={idx} className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
                                  <div className="font-bold text-white text-lg mb-3">Q: {item.question}</div>
                                  <div className="grid md:grid-cols-2 gap-6">
                                      <div>
                                          <div className="text-xs text-gray-500 uppercase font-bold mb-1">Your Answer</div>
                                          <p className="text-gray-300 text-sm">{item.userAnswer}</p>
                                          <div className="mt-3 p-3 bg-blue-500/10 border border-blue-500/30 rounded text-blue-300 text-sm">
                                              <strong>Feedback:</strong> {item.feedback}
                                          </div>
                                      </div>
                                      <div>
                                          <div className="text-xs text-gray-500 uppercase font-bold mb-1">Expected Answer</div>
                                          <p className="text-gray-300 text-sm bg-black/20 p-3 rounded">{item.expectedAnswer}</p>
                                      </div>
                                  </div>
                              </div>
                          ))}
                      </div>
                  </div>

                  <div className="flex justify-center">
                      <button 
                        onClick={() => setReport(null)}
                        className="px-8 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-full font-bold transition-all"
                      >
                          Start New Session
                      </button>
                  </div>
              </div>
          )}
      </div>
    </div>
  );
};

export default MockInterview;