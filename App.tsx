import React, { useState, useEffect } from 'react';
import { AppView, PrepPlan, Problem, Submission, CodeFeedback, ThemeId, ThemeColors, VisualHistoryItem, User, UserProfile } from './types';
import SetupForm from './components/SetupForm';
import PlanDashboard from './components/PlanDashboard';
import MockInterview from './components/MockInterview';
import VisualLab from './components/VisualLab';
import ResearchTool from './components/ResearchTool';
import ProblemBank from './components/ProblemBank';
import ProblemSolver from './components/ProblemSolver';
import JobBoard from './components/JobBoard';
import Wishes from './components/Wishes';
import Playground from './components/Playground';
import ThemeSelector from './components/ThemeSelector';
import WelcomePage from './components/WelcomePage';
import AuthPage from './components/AuthPage';
import ProfilePage from './components/ProfilePage';
import ClickEffects from './components/ClickEffects';
import CosmicBackground from './components/CosmicBackground';
import SeaBackground from './components/SeaBackground';
import FlowerBackground from './components/FlowerBackground';
import SnowBackground from './components/SnowBackground';
import GothicBackground from './components/GothicBackground';
import ChristmasBackground from './components/ChristmasBackground';
import { restoreSession, saveSubmission, saveUserPlan, saveVisualGeneration, toggleLikeProblem, logoutUser, saveGameScore } from './services/userService';

const THEME_CONFIG: Record<ThemeId, ThemeColors> = {
  midnight: {
    bgApp: '#0f172a',
    bgPanel: '#1e293b',
    bgCard: '#111827',
    textMain: '#f8fafc',
    textMuted: '#9ca3af',
    colPrimary: '#3b82f6',
    borderCol: '#374151',
  },
  solar: {
    bgApp: '#f0f9ff',
    bgPanel: '#ffffff',
    bgCard: '#e2e8f0',
    textMain: '#0f172a',
    textMuted: '#475569',
    colPrimary: '#2563eb',
    borderCol: '#cbd5e1',
  },
  neon: {
    bgApp: '#050505',
    bgPanel: '#121212',
    bgCard: '#1a1a1a',
    textMain: '#e5e5e5',
    textMuted: '#a3a3a3',
    colPrimary: '#00ff41',
    borderCol: '#333333',
  },
  deepspace: {
    bgApp: '#1a0b2e',
    bgPanel: '#2e1065',
    bgCard: '#4c1d95',
    textMain: '#f3e8ff',
    textMuted: '#d8b4fe',
    colPrimary: '#d946ef',
    borderCol: '#7e22ce',
  },
  toon: {
    bgApp: '#fef3c7',
    bgPanel: '#ffffff',
    bgCard: '#dbeafe',
    textMain: '#000000',
    textMuted: '#4b5563',
    colPrimary: '#f97316',
    borderCol: '#000000',
  },
  cosmic: {
    bgApp: 'transparent', 
    bgPanel: 'rgba(15, 23, 42, 0.75)', 
    bgCard: 'rgba(30, 41, 59, 0.6)',
    textMain: '#e2e8f0',
    textMuted: '#94a3b8',
    colPrimary: '#38bdf8', 
    borderCol: 'rgba(148, 163, 184, 0.2)',
  },
  sea: {
    bgApp: 'transparent',
    bgPanel: 'rgba(2, 44, 34, 0.75)',
    bgCard: 'rgba(19, 78, 74, 0.6)',
    textMain: '#ecfeff', // cyan-50
    textMuted: '#99f6e4', // teal-200
    colPrimary: '#2dd4bf', // teal-400
    borderCol: 'rgba(45, 212, 191, 0.2)',
  },
  flower: {
    bgApp: 'transparent',
    bgPanel: 'rgba(255, 255, 255, 0.7)',
    bgCard: 'rgba(240, 253, 244, 0.8)',
    textMain: '#14532d', // green-900
    textMuted: '#166534', // green-800
    colPrimary: '#3b82f6', // Blue for the flowers
    borderCol: 'rgba(134, 239, 172, 0.5)',
  },
  snow: {
    bgApp: 'transparent',
    bgPanel: 'rgba(30, 41, 59, 0.8)', // slate-800
    bgCard: 'rgba(51, 65, 85, 0.6)', // slate-700
    textMain: '#f1f5f9', // slate-100
    textMuted: '#cbd5e1', // slate-300
    colPrimary: '#60a5fa', // blue-400
    borderCol: 'rgba(203, 213, 225, 0.2)',
  },
  gothic: {
    bgApp: 'transparent',
    bgPanel: 'rgba(20, 5, 5, 0.85)', // Very dark red/black
    bgCard: 'rgba(40, 10, 15, 0.7)',
    textMain: '#e5e5e5', // Light gray
    textMuted: '#a8a29e', // Warm gray
    colPrimary: '#9f1239', // Rose 800
    borderCol: 'rgba(159, 18, 57, 0.3)', // Reddish border
  },
  christmas: {
    bgApp: 'transparent',
    bgPanel: 'rgba(6, 78, 59, 0.8)', // Emerald 900/80
    bgCard: 'rgba(2, 44, 34, 0.7)', // Emerald 950/70
    textMain: '#f0fdf4', // Green 50 (Ice White)
    textMuted: '#86efac', // Green 300
    colPrimary: '#ef4444', // Red 500
    borderCol: 'rgba(16, 185, 129, 0.3)', // Emerald 500
  }
};

const App: React.FC = () => {
  const [showWelcome, setShowWelcome] = useState(true);
  const [theme, setTheme] = useState<ThemeId | null>(null);
  const [view, setView] = useState<AppView>(AppView.AUTH);
  const [activeProblem, setActiveProblem] = useState<Problem | null>(null);
  
  // Auth State
  const [user, setUser] = useState<User | null>(null);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);

  // Initialize Session
  useEffect(() => {
      const initSession = async () => {
          const session = await restoreSession();
          if (session) {
              setUser(session.user);
              setUserProfile(session.profile);
              setView(session.profile.currentPlan ? AppView.DASHBOARD : AppView.SETUP);
              // FIXED: Do not auto-hide welcome page to prevent "jumping" effect.
              // User must manually click "Start" to enter, even if session is restored.
              // setShowWelcome(false); 
              setTheme('midnight'); // Default theme if restored
          }
      };
      initSession();
  }, []);

  // Update CSS Variables
  useEffect(() => {
    if (theme) {
      const colors = THEME_CONFIG[theme];
      const root = document.documentElement;
      root.style.setProperty('--bg-app', colors.bgApp);
      root.style.setProperty('--bg-panel', colors.bgPanel);
      root.style.setProperty('--bg-card', colors.bgCard);
      root.style.setProperty('--text-main', colors.textMain);
      root.style.setProperty('--text-muted', colors.textMuted);
      root.style.setProperty('--col-primary', colors.colPrimary);
      root.style.setProperty('--border-col', colors.borderCol);
    }
  }, [theme]);

  // Handlers
  const handleLogin = (u: User, p: UserProfile) => {
      setUser(u);
      setUserProfile(p);
      setView(p.currentPlan ? AppView.DASHBOARD : AppView.SETUP);
  };

  const handleLogout = async () => {
      await logoutUser();
      setUser(null);
      setUserProfile(null);
      setView(AppView.AUTH);
      setShowWelcome(true);
  };

  const handlePlanGenerated = async (newPlan: PrepPlan) => {
    if (user) {
        const updated = await saveUserPlan(user.id, newPlan);
        setUserProfile(updated);
    }
    setView(AppView.DASHBOARD);
  };

  const handleSkipToLibrary = () => {
    setView(AppView.PROBLEM_BANK);
  };

  const handleSelectProblem = (problem: Problem) => {
    setActiveProblem(problem);
    setView(AppView.PROBLEM_SOLVER);
  };

  const handleSubmitSuccess = async (problemId: string, code: string, feedback: CodeFeedback) => {
    if (user) {
        const submission: Submission = { problemId, code, feedback, timestamp: Date.now() };
        const updated = await saveSubmission(user.id, submission);
        setUserProfile(updated);
    }
  };

  const handleAddToHistory = async (item: VisualHistoryItem) => {
      if (user) {
          const updated = await saveVisualGeneration(user.id, item);
          setUserProfile(updated);
      }
  };
  
  const handleLikeProblem = async (problemId: string) => {
      if (user) {
          const updated = await toggleLikeProblem(user.id, problemId);
          setUserProfile(updated);
      }
  };

  const handleSaveScore = async (game: string, score: number) => {
      if (user) {
          const updated = await saveGameScore(user.id, game, score);
          setUserProfile(updated);
      }
  };

  if (showWelcome) {
    return <WelcomePage onStart={() => setShowWelcome(false)} />;
  }

  if (!theme) {
    return (
      <>
        <ClickEffects theme={theme} />
        <ThemeSelector onSelectTheme={setTheme} />
      </>
    );
  }

  // Not Authenticated -> Show Auth Page
  if (!user || !userProfile) {
      return (
        <div className="min-h-screen bg-app-bg text-text-main">
            <AuthPage onLogin={handleLogin} />
        </div>
      );
  }

  const navItems = [
    // Only show Dashboard if a plan exists
    ...(userProfile.currentPlan ? [{ id: AppView.DASHBOARD, label: 'Dashboard', icon: '📊' }] : []),
    { id: AppView.PROBLEM_BANK, label: 'Practice Bank', icon: '💻' },
    { id: AppView.MOCK_INTERVIEW, label: 'Mock Interview', icon: '🎙️' },
    { id: AppView.VISUAL_LAB, label: 'Visual Lab', icon: '🎨' },
    { id: AppView.RESEARCH, label: 'Research', icon: '🔍' },
    { id: AppView.JOB_HUNT, label: 'Job Hunt', icon: '💼' },
    { id: AppView.PLAYGROUND, label: 'Playground', icon: '🎮' },
    { id: AppView.PROFILE, label: 'Profile', icon: '👤' },
    { id: AppView.WISHES, label: 'Wishes', icon: '✨' },
  ];

  return (
    <div className={`min-h-screen flex font-sans selection:bg-primary selection:text-white bg-app-bg text-text-main transition-colors duration-300 relative ${theme === 'toon' ? 'font-bold' : ''}`}>
      {theme === 'cosmic' && <CosmicBackground />}
      {theme === 'sea' && <SeaBackground />}
      {theme === 'flower' && <FlowerBackground />}
      {theme === 'snow' && <SnowBackground />}
      {theme === 'gothic' && <GothicBackground />}
      {theme === 'christmas' && <ChristmasBackground />}
      
      <ClickEffects theme={theme} />
      
      {/* Sidebar Navigation */}
      {view !== AppView.SETUP && (
        <aside className="w-64 bg-panel-bg border-r border-border-col flex flex-col fixed h-full z-10 transition-colors duration-300 backdrop-blur-md">
          <div className="p-6 border-b border-border-col flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-pink-500 to-indigo-500 flex items-center justify-center text-white shadow-lg shadow-pink-500/20">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
                </svg>
            </div>
            <div>
                <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-pink-400 to-indigo-400">
                Love&DeepCode
                </h1>
                <p className="text-[10px] text-text-muted uppercase tracking-widest">Logged in as {user.name.split(' ')[0]}</p>
            </div>
          </div>
          <nav className="flex-1 p-4 space-y-2">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => setView(item.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
                  view === item.id || (view === AppView.PROBLEM_SOLVER && item.id === AppView.PROBLEM_BANK)
                    ? 'bg-primary text-white shadow-lg' 
                    : 'text-text-muted hover:bg-card-bg hover:text-text-main'
                }`}
              >
                <span>{item.icon}</span>
                <span className="font-medium">{item.label}</span>
              </button>
            ))}
          </nav>
          <div className="p-4 border-t border-border-col">
             <div className="bg-card-bg rounded-lg p-3 text-xs text-text-muted border border-border-col">
                <p className="mb-2 flex justify-between">Theme: <span className="uppercase font-bold">{theme}</span></p>
                <div className="flex justify-between mt-2">
                    <button onClick={() => setTheme(null)} className="underline hover:text-primary">Change Theme</button>
                    <button onClick={handleLogout} className="text-red-400 hover:text-red300">Logout</button>
                </div>
             </div>
          </div>
        </aside>
      )}

      {/* Main Content Area */}
      <main className={`flex-1 flex flex-col ${view !== AppView.SETUP ? 'ml-64' : ''} h-screen overflow-hidden relative z-10`}>
        {view === AppView.SETUP && (
            <div className={`flex-1 flex flex-col items-center justify-center relative ${['cosmic', 'sea', 'flower', 'snow', 'gothic', 'christmas'].includes(theme) ? "" : "bg-[url('https://picsum.photos/1920/1080?blur=10')] bg-cover bg-center"}`}>
                {!['cosmic', 'sea', 'flower', 'snow', 'gothic', 'christmas'].includes(theme) && <div className="absolute inset-0 bg-app-bg/90 backdrop-blur-sm"></div>}
                <div className="relative z-10 w-full">
                    <SetupForm 
                        onPlanGenerated={handlePlanGenerated} 
                        onSkipToLibrary={handleSkipToLibrary}
                    />
                </div>
            </div>
        )}

        {view !== AppView.SETUP && (
            <div className="flex-1 bg-app-bg overflow-hidden transition-colors duration-300">
                {view === AppView.DASHBOARD && userProfile.currentPlan && <PlanDashboard plan={userProfile.currentPlan} submissions={userProfile.submissions} />}
                {view === AppView.PROBLEM_BANK && <ProblemBank onSelectProblem={handleSelectProblem} submissions={userProfile.submissions} />}
                {view === AppView.PROBLEM_SOLVER && activeProblem && (
                    <ProblemSolver 
                        problem={activeProblem} 
                        onBack={() => setView(AppView.PROBLEM_BANK)}
                        onSubmitSuccess={handleSubmitSuccess}
                        isLiked={userProfile.likedProblemIds.includes(activeProblem.id)}
                        onToggleLike={() => handleLikeProblem(activeProblem.id)}
                    />
                )}
                {view === AppView.MOCK_INTERVIEW && <MockInterview />}
                {view === AppView.VISUAL_LAB && (
                    <VisualLab 
                      history={userProfile.visualHistory} 
                      onAddToHistory={handleAddToHistory} 
                    />
                )}
                {view === AppView.RESEARCH && <ResearchTool />}
                {view === AppView.JOB_HUNT && <JobBoard />}
                {view === AppView.PLAYGROUND && (
                    <Playground 
                        onSaveScore={handleSaveScore} 
                    />
                )}
                {view === AppView.PROFILE && <ProfilePage user={user} profile={userProfile} />}
                {view === AppView.WISHES && <Wishes />}
            </div>
        )}
      </main>
    </div>
  );
};

export default App;