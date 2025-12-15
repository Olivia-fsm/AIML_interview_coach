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
  const successfulSubmissions = profile.submissions.filter(s => s.feedback.correctnessScore > 80).length;

  return (
    <div className="h-full overflow-y-auto p-8 animate-fade-in">
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
            {/* Left Col: Liked Problems */}
            <div className="space-y-6">
                <h3 className="text-2xl font-bold text-text-main flex items-center gap-2">
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

                <h3 className="text-2xl font-bold text-text-main flex items-center gap-2 mt-10">
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
                    <div className="grid grid-cols-2 gap-4">
                        {profile.visualHistory.map((item) => (
                            <div key={item.id} className="group relative aspect-video bg-black rounded-xl overflow-hidden border border-border-col shadow-lg">
                                {item.type === 'image' ? (
                                    <img src={item.mediaUrl} alt={item.prompt} className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110" />
                                ) : (
                                    <video src={item.mediaUrl} className="w-full h-full object-cover" />
                                )}
                                <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex flex-col justify-end p-3">
                                    <p className="text-white text-xs font-bold line-clamp-2">{item.prompt}</p>
                                    <span className="text-[10px] text-gray-300 uppercase mt-1">{item.mode}</span>
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