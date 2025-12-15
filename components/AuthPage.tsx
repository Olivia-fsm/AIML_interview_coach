import React, { useState } from 'react';
import { loginUser, registerUser } from '../services/userService';
import { User, UserProfile } from '../types';

interface Props {
  onLogin: (user: User, profile: UserProfile) => void;
}

const AuthPage: React.FC<Props> = ({ onLogin }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      let result;
      if (isLogin) {
        result = await loginUser(email, password);
      } else {
        if (!name) throw new Error("Name is required");
        result = await registerUser(email, password, name);
      }
      onLogin(result.user, result.profile);
    } catch (err: any) {
      setError(err.message || "Authentication failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-app-bg relative overflow-hidden">
        {/* Background Ambient */}
        <div className="absolute top-0 left-0 w-full h-full overflow-hidden z-0">
             <div className="absolute top-[-10%] right-[-10%] w-[50%] h-[50%] bg-primary/20 blur-[100px] rounded-full animate-pulse"></div>
             <div className="absolute bottom-[-10%] left-[-10%] w-[50%] h-[50%] bg-pink-600/20 blur-[100px] rounded-full"></div>
        </div>

        <div className="relative z-10 w-full max-w-md p-8 bg-panel-bg/80 backdrop-blur-md border border-border-col rounded-3xl shadow-2xl">
            <div className="text-center mb-8">
                <div className="w-12 h-12 bg-gradient-to-br from-pink-500 to-indigo-500 rounded-xl flex items-center justify-center shadow-lg shadow-pink-500/20 mx-auto mb-4">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
                    </svg>
                </div>
                <h2 className="text-3xl font-bold text-text-main">{isLogin ? 'Welcome Back' : 'Create Account'}</h2>
                <p className="text-text-muted mt-2">{isLogin ? 'Continue your interview prep journey.' : 'Start your journey to a dream ML role.'}</p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
                {!isLogin && (
                    <div>
                        <label className="block text-sm font-medium text-text-muted mb-1">Full Name</label>
                        <input 
                            type="text" 
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            className="w-full bg-card-bg border border-border-col rounded-lg px-4 py-3 text-text-main focus:ring-2 focus:ring-primary outline-none"
                            placeholder="Ada Lovelace"
                        />
                    </div>
                )}
                
                <div>
                    <label className="block text-sm font-medium text-text-muted mb-1">Email Address</label>
                    <input 
                        type="email" 
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        className="w-full bg-card-bg border border-border-col rounded-lg px-4 py-3 text-text-main focus:ring-2 focus:ring-primary outline-none"
                        placeholder="you@example.com"
                    />
                </div>

                <div>
                    <label className="block text-sm font-medium text-text-muted mb-1">Password</label>
                    <input 
                        type="password" 
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        className="w-full bg-card-bg border border-border-col rounded-lg px-4 py-3 text-text-main focus:ring-2 focus:ring-primary outline-none"
                        placeholder="••••••••"
                    />
                </div>

                {error && (
                    <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-200 text-sm">
                        {error}
                    </div>
                )}

                <button 
                    type="submit" 
                    disabled={loading}
                    className="w-full py-4 bg-primary hover:opacity-90 text-white font-bold rounded-xl shadow-lg transition-all transform active:scale-95 disabled:opacity-50"
                >
                    {loading ? 'Processing...' : isLogin ? 'Sign In' : 'Sign Up'}
                </button>
            </form>

            <div className="mt-6 text-center">
                <p className="text-sm text-text-muted">
                    {isLogin ? "Don't have an account? " : "Already have an account? "}
                    <button 
                        onClick={() => { setIsLogin(!isLogin); setError(''); }}
                        className="text-primary font-bold hover:underline"
                    >
                        {isLogin ? 'Sign Up' : 'Log In'}
                    </button>
                </p>
            </div>
        </div>
    </div>
  );
};

export default AuthPage;