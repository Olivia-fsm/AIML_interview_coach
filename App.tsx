import React, { useState } from 'react';
import { AppView, PrepPlan, Problem, Submission, CodeFeedback } from './types';
import SetupForm from './components/SetupForm';
import PlanDashboard from './components/PlanDashboard';
import MockInterview from './components/MockInterview';
import VisualLab from './components/VisualLab';
import ResearchTool from './components/ResearchTool';
import ProblemBank from './components/ProblemBank';
import ProblemSolver from './components/ProblemSolver';

const App: React.FC = () => {
  const [view, setView] = useState<AppView>(AppView.SETUP);
  const [plan, setPlan] = useState<PrepPlan | null>(null);
  
  // New state for problem solving
  const [submissions, setSubmissions] = useState<Submission[]>([]);
  const [activeProblem, setActiveProblem] = useState<Problem | null>(null);

  const handlePlanGenerated = (newPlan: PrepPlan) => {
    setPlan(newPlan);
    setView(AppView.DASHBOARD);
  };

  const handleSkipToLibrary = () => {
    setView(AppView.PROBLEM_BANK);
  };

  const handleSelectProblem = (problem: Problem) => {
    setActiveProblem(problem);
    setView(AppView.PROBLEM_SOLVER);
  };

  const handleSubmitSuccess = (problemId: string, code: string, feedback: CodeFeedback) => {
    setSubmissions(prev => [
      ...prev, 
      { problemId, code, feedback, timestamp: Date.now() }
    ]);
  };

  const navItems = [
    // Only show Dashboard if a plan exists
    ...(plan ? [{ id: AppView.DASHBOARD, label: 'Dashboard', icon: '📊' }] : []),
    { id: AppView.PROBLEM_BANK, label: 'Practice Bank', icon: '💻' },
    { id: AppView.MOCK_INTERVIEW, label: 'Mock Interview', icon: '🎙️' },
    { id: AppView.VISUAL_LAB, label: 'Visual Lab', icon: '🎨' },
    { id: AppView.RESEARCH, label: 'Research', icon: '🔍' },
  ];

  return (
    <div className="min-h-screen flex text-gray-100 font-sans selection:bg-primary selection:text-white">
      {/* Sidebar Navigation */}
      {view !== AppView.SETUP && (
        <aside className="w-64 bg-secondary border-r border-gray-800 flex flex-col fixed h-full z-10">
          <div className="p-6 border-b border-gray-800">
            <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
              PrepAI
            </h1>
            <p className="text-xs text-gray-500 mt-1">ML Interview Coach</p>
          </div>
          <nav className="flex-1 p-4 space-y-2">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => setView(item.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
                  view === item.id || (view === AppView.PROBLEM_SOLVER && item.id === AppView.PROBLEM_BANK)
                    ? 'bg-primary text-white shadow-lg shadow-blue-900/50' 
                    : 'text-gray-400 hover:bg-gray-800 hover:text-white'
                }`}
              >
                <span>{item.icon}</span>
                <span className="font-medium">{item.label}</span>
              </button>
            ))}
          </nav>
          <div className="p-4 border-t border-gray-800">
             <div className="bg-gray-800/50 rounded-lg p-3 text-xs text-gray-500">
                <p>Powered by Gemini 2.5 & Veo</p>
             </div>
          </div>
        </aside>
      )}

      {/* Main Content Area */}
      <main className={`flex-1 flex flex-col ${view !== AppView.SETUP ? 'ml-64' : ''} h-screen overflow-hidden`}>
        {view === AppView.SETUP && (
            <div className="flex-1 flex flex-col items-center justify-center bg-[url('https://picsum.photos/1920/1080?blur=10')] bg-cover bg-center relative">
                <div className="absolute inset-0 bg-gray-900/90 backdrop-blur-sm"></div>
                <div className="relative z-10 w-full">
                    <SetupForm 
                        onPlanGenerated={handlePlanGenerated} 
                        onSkipToLibrary={handleSkipToLibrary}
                    />
                </div>
            </div>
        )}

        {view !== AppView.SETUP && (
            <div className="flex-1 bg-[#0f172a] overflow-hidden">
                {view === AppView.DASHBOARD && plan && <PlanDashboard plan={plan} submissions={submissions} />}
                {view === AppView.PROBLEM_BANK && <ProblemBank onSelectProblem={handleSelectProblem} submissions={submissions} />}
                {view === AppView.PROBLEM_SOLVER && activeProblem && (
                    <ProblemSolver 
                        problem={activeProblem} 
                        onBack={() => setView(AppView.PROBLEM_BANK)}
                        onSubmitSuccess={handleSubmitSuccess}
                    />
                )}
                {view === AppView.MOCK_INTERVIEW && <MockInterview />}
                {view === AppView.VISUAL_LAB && <VisualLab />}
                {view === AppView.RESEARCH && <ResearchTool />}
            </div>
        )}
      </main>
    </div>
  );
};

export default App;