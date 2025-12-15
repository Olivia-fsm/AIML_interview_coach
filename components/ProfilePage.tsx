import React from 'react';
import { User, UserProfile, Problem } from '../types';
import { PROBLEM_LIBRARY } from '../data/problems';

interface Props {
  user: User;
  profile: UserProfile;
}

const ProfilePage: React.FC<Props> = ({ user, profile }) => {
  const likedProblems = PROBLEM_LIBRARY.filter(p => profile.likedProblemIds.includes(p.id));
  const totalSubmissions = profile.submissions.length;
  // Safety check: ensure feedback exists before accessing properties
  const successfulSubmissions = profile.submissions.filter(s => s.feedback && s.feedback.correctnessScore > 80).length;

  // Sort submissions by date (newest first)
  const recentSubmissions = [...profile.submissions]
    .sort((a, b) => b.timestamp - a.timestamp)
    .slice(0, 10); // Show last 10

  const getProblemTitle = (id: string) => PROBLEM_LIBRARY.find(p => p.id === id)?.title || id;

  return (
    <div className="h-full overflow-y-auto p-8 animate-fade-in custom-scrollbar">
        {/* Header Profile Card */}
        <div className="bg-panel-bg rounded-3xl p-8 border border-border-col shadow-2xl mb-8 flex flex-col md:flex-row items-center gap-8 relative overflow-hidden">
             <div className="absolute top-0 left-0 w-full h-32 bg-gradient-to-r from-primary/20 to-pink-500/20"></div>
             
             <div className="relative z-10 w-32 h-32 rounded-full border-4 border-panel-bg shadow-xl overflow-hidden bg-card-bg">
                 <img src={user.avatar} alt="avatar" className="w-full h-full object-cover" />
             </div>
             
             <div className="relative z-10 text-center md:text-left flex-1">
                 <h2 className="text-4xl font-bold text-text-main mb-1">{user.name}</h2>
                 <p className="text-text-muted mb-4">{user.email}</p>
                 
                 <div className="flex gap-4 justify-center md:justify-start">
                     <div className="bg-card-bg px-4 py-2 rounded-lg border border-border-col">
                         <span className="text-xs text-text-muted uppercase font-bold">Level</span>
                         <div className="text-2xl font-bold text-primary">{profile.level}</div>
                     </div>
                     <div className="bg-card-bg px-4 py-2 rounded-lg border border-border-col">
                         <span className="text-xs text-text-muted uppercase font-bold">XP</span>
                         <div className="text-2xl font-bold text-pink-500">{profile.xp}</div>
                     </div>
                     <div className="bg-card-bg px-4 py-2 rounded-lg border border-border-col">
                         <span className="text-xs text-text-muted uppercase font-bold">Solved</span>
                         <div className="text-2xl font-bold text-green-500">{successfulSubmissions}</div>
                     </div>
                 </div>
             </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Left Col: Activity & Likes */}
            <div className="space-y-8">
                
                {/* Recent Activity Section (NEW) */}
                <div>
                    <h3 className="text-2xl font-bold text-text-main flex items-center gap-2 mb-4">
                        <span className="text-purple-400">⚡</span> Recent Activity
                    </h3>
                    <div className="bg-panel-bg rounded-xl border border-border-col overflow-hidden">
                        {recentSubmissions.length === 0 ? (
                            <div className="p-6 text-center text-text-muted italic">No coding activity recorded yet.</div>
                        ) : (
                            <div className="divide-y divide-border-col">
                                {recentSubmissions.map((sub, idx) => (
                                    <div key={idx} className="p-4 hover:bg-card-bg transition-colors flex justify-between items-center">
                                        <div>
                                            <div className="font-bold text-text-main text-sm">{getProblemTitle(sub.problemId)}</div>
                                            <div className="text-xs text-text-muted">
                                                {new Date(sub.timestamp).toLocaleDateString()} • {new Date(sub.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <div className={`text-lg font-bold ${sub.feedback && sub.feedback.correctnessScore > 80 ? 'text-green-500' : 'text-yellow-500'}`}>
                                                {sub.feedback ? sub.feedback.correctnessScore : '?'}
                                            </div>
                                            <div className="text-[10px] uppercase font-bold text-text-muted">Score</div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>

                {/* Liked Problems */}
                <div>
                    <h3 className="text-2xl font-bold text-text-main flex items-center gap-2 mb-4">
                        <span className="text-red-500">❤️</span> Liked Problems
                    </h3>
                    {likedProblems.length === 0 ? (
                        <div className="text-text-muted italic bg-panel-bg p-6 rounded-xl border border-border-col text-center">
                            No liked problems yet. Go to the Practice Bank!
                        </div>
                    ) : (
                        <div className="grid gap-4">
                            {likedProblems.map(p => (
                                <div key={p.id} className="bg-panel-bg p-4 rounded-xl border border-border-col hover:border-primary transition-colors cursor-pointer group">
                                    <div className="flex justify-between items-start">
                                        <h4 className="font-bold text-text-main group-hover:text-primary">{p.title}</h4>
                                        <span className="text-xs bg-card-bg px-2 py-1 rounded border border-border-col">{p.difficulty}</span>
                                    </div>
                                    <p className="text-xs text-text-muted mt-1">{p.category}</p>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* High Scores */}
                <div>
                    <h3 className="text-2xl font-bold text-text-main flex items-center gap-2 mb-4">
                        <span className="text-yellow-400">🏆</span> Game High Scores
                    </h3>
                    <div className="bg-panel-bg p-6 rounded-xl border border-border-col">
                        {Object.keys(profile.gameHighScores).length === 0 ? (
                            <p className="text-text-muted italic">No games played yet.</p>
                        ) : (
                            <div className="space-y-2">
                                {Object.entries(profile.gameHighScores).map(([game, score]) => (
                                    <div key={game} className="flex justify-between items-center p-3 bg-card-bg rounded-lg">
                                        <span className="font-bold text-text-main capitalize">{game.replace(/_/g, ' ')}</span>
                                        <span className="font-mono text-xl text-primary">{score}</span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Right Col: Visual History Gallery */}
            <div>
                <h3 className="text-2xl font-bold text-text-main mb-6 flex items-center gap-2">
                    <span className="text-blue-400">🎨</span> Visual Library
                </h3>
                {profile.visualHistory.length === 0 ? (
                     <div className="text-text-muted italic bg-panel-bg p-6 rounded-xl border border-border-col text-center">
                        Your generated diagrams and videos will appear here.
                    </div>
                ) : (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        {profile.visualHistory.map((item) => (
                            <div key={item.id} className="group relative aspect-video bg-black rounded-xl overflow-hidden border border-border-col shadow-lg hover:ring-2 hover:ring-primary transition-all">
                                {item.type === 'image' ? (
                                    <img src={item.mediaUrl} alt={item.prompt} className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110" />
                                ) : (
                                    <div className="relative w-full h-full">
                                        <video src={item.mediaUrl} className="w-full h-full object-cover" />
                                        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                                            <div className="bg-black/50 p-2 rounded-full backdrop-blur-sm">
                                                <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                                            </div>
                                        </div>
                                    </div>
                                )}
                                <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex flex-col justify-end p-4">
                                    <p className="text-white text-xs font-bold line-clamp-2 mb-1">{item.prompt}</p>
                                    <div className="flex justify-between items-center">
                                        <span className="text-[10px] text-primary uppercase font-bold tracking-wider">{item.mode}</span>
                                        <span className="text-[10px] text-gray-400">{new Date(item.timestamp).toLocaleDateString()}</span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    </div>
  );
};

export default ProfilePage;