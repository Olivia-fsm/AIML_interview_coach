import React, { useState, useEffect } from 'react';
import { PrepPlan, Submission } from '../types';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, PieChart, Pie } from 'recharts';

interface Props {
  plan: PrepPlan;
  submissions: Submission[];
  onEndPlan: () => void;
}

const PlanDashboard: React.FC<Props> = ({ plan, submissions, onEndPlan }) => {
  const [isMounted, setIsMounted] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);

  // Guard against Recharts calculation errors on initial mount
  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Plan Chart Data
  const scheduleData = (plan.schedule || []).map(d => ({
    name: `Day ${d.day}`,
    tasks: d.tasks?.length || 0,
    focus: d.focusArea
  }));

  // Progress Data
  const solvedCount = submissions.filter(s => s.feedback && s.feedback.correctnessScore > 80).length;
  const avgScore = submissions.length > 0 
    ? Math.round(submissions.reduce((acc, curr) => acc + (curr.feedback?.correctnessScore || 0), 0) / submissions.length)
    : 0;
  
  const pieData = [
      { name: 'Solved', value: solvedCount },
      { name: 'Remaining', value: Math.max(0, 15 - solvedCount) }
  ];
  const COLORS = ['var(--col-primary)', 'rgba(255, 255, 255, 0.05)'];

  return (
    <div className="p-6 space-y-8 animate-fade-in h-full overflow-y-auto custom-scrollbar">
      {/* Header */}
      <div className="bg-panel-bg p-6 rounded-2xl border border-border-col shadow-lg">
        <div className="flex flex-col md:flex-row justify-between items-start gap-4">
            <div className="flex-1">
                <h2 className="text-3xl font-bold text-text-main mb-2">{plan.roleTitle}</h2>
                <p className="text-primary font-medium text-lg">Target: {plan.targetCompany}</p>
            </div>
            <div className="md:text-right">
                <p className="text-sm text-text-muted">Interview Date</p>
                <p className="text-xl font-bold text-text-main">{plan.interviewDate}</p>
            </div>
        </div>
        <p className="mt-4 text-text-muted italic">"{plan.summary}"</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Schedule */}
        <div className="lg:col-span-2 space-y-4">
            <h3 className="text-xl font-semibold text-text-main flex items-center gap-2">
                <span className="bg-primary/20 text-primary p-1 rounded">📅</span> Daily Schedule
            </h3>
            <div className="space-y-4">
                {(plan.schedule || []).length > 0 ? (
                  plan.schedule.map((day) => (
                    <div key={day.day} className="bg-panel-bg p-5 rounded-xl border border-border-col hover:border-primary transition-colors">
                        <div className="flex justify-between mb-3">
                            <span className="font-bold text-text-main text-lg">{day.date}</span>
                            <span className="px-3 py-1 bg-accent/20 text-accent rounded-full text-xs font-bold uppercase tracking-wider">{day.focusArea}</span>
                        </div>
                        <ul className="space-y-2">
                            {(day.tasks || []).map((task, idx) => (
                                <li key={idx} className="flex items-start gap-2 text-text-muted text-sm">
                                    <span className="mt-1.5 w-1.5 h-1.5 bg-primary rounded-full shrink-0"></span>
                                    {task}
                                </li>
                            ))}
                        </ul>
                    </div>
                  ))
                ) : (
                  <div className="text-text-muted p-4 border border-dashed border-border-col rounded-xl text-center">
                    No schedule generated.
                  </div>
                )}
            </div>
        </div>

        {/* Right Column: Analytics */}
        <div className="space-y-6">
            {/* Progress Card */}
            <div className="bg-panel-bg p-5 rounded-xl border border-border-col">
                <h3 className="text-lg font-semibold text-text-main mb-4">Coding Progress</h3>
                <div style={{ width: '100%', height: '240px', position: 'relative' }}>
                    {isMounted && (
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={pieData}
                                    innerRadius={60}
                                    outerRadius={80}
                                    paddingAngle={5}
                                    dataKey="value"
                                    isAnimationActive={false}
                                >
                                    {pieData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} stroke="none"/>
                                    ))}
                                </Pie>
                            </PieChart>
                        </ResponsiveContainer>
                    )}
                    <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                        <span className="text-3xl font-bold text-text-main">{solvedCount}</span>
                        <span className="text-xs text-text-muted">Solved</span>
                    </div>
                </div>
                <div className="flex justify-between mt-4 px-4">
                     <div className="text-center">
                         <div className="text-2xl font-bold text-text-main">{submissions.length}</div>
                         <div className="text-xs text-text-muted">Attempts</div>
                     </div>
                     <div className="text-center">
                         <div className="text-2xl font-bold text-blue-400">{avgScore}%</div>
                         <div className="text-xs text-text-muted">Avg Score</div>
                     </div>
                </div>
            </div>

            {/* Load Chart */}
            <div className="bg-panel-bg p-5 rounded-xl border border-border-col">
                <h3 className="text-lg font-semibold text-text-main mb-4">Study Load</h3>
                <div style={{ width: '100%', height: '180px' }}>
                    {isMounted && (
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={scheduleData}>
                                <XAxis dataKey="name" hide />
                                <Tooltip 
                                    contentStyle={{ backgroundColor: 'var(--bg-panel)', border: '1px solid var(--border-col)', color: 'var(--text-main)' }}
                                    cursor={{fill: 'var(--bg-card)'}}
                                />
                                <Bar dataKey="tasks" fill="var(--col-primary)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                            </BarChart>
                        </ResponsiveContainer>
                    )}
                </div>
            </div>

             <div className="bg-gradient-to-br from-indigo-900/40 to-purple-900/40 p-5 rounded-xl border border-indigo-700/30 text-center">
                <h3 className="text-lg font-bold text-white mb-2">Mock Interview</h3>
                <p className="text-sm text-indigo-200 mb-4">Test your verbal skills with the Live API.</p>
                <div className="inline-block px-4 py-2 bg-white/10 rounded-lg text-xs font-mono text-white">
                    Status: READY
                </div>
            </div>
        </div>
      </div>

      {/* FOOTER: End Plan Section */}
      <div className="pt-10 pb-20 flex flex-col items-center">
          <div className="w-full max-w-lg p-8 bg-panel-bg border border-dashed border-border-col rounded-3xl text-center">
              <h3 className="text-xl font-bold text-text-main mb-2">Ready for a new direction?</h3>
              <p className="text-text-muted text-sm mb-6">
                  Ending this plan will return you to the setup page. You'll keep your XP and solving history, but the current daily schedule will be cleared.
              </p>
              
              {!showConfirm ? (
                <button 
                  type="button"
                  onClick={() => setShowConfirm(true)}
                  className="group px-8 py-4 bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/30 rounded-2xl font-bold transition-all flex items-center gap-3 mx-auto shadow-lg active:scale-95"
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                  End Current Plan
                </button>
              ) : (
                <div className="flex flex-col gap-4 animate-fade-in">
                  <p className="text-red-400 font-bold text-sm">Are you absolutely sure?</p>
                  <div className="flex justify-center gap-4">
                    <button 
                      type="button"
                      onClick={() => {
                        console.log("End Plan confirmed in UI");
                        onEndPlan();
                      }}
                      className="px-6 py-3 bg-red-600 hover:bg-red-500 text-white rounded-xl font-bold transition-all shadow-lg active:scale-95"
                    >
                      Yes, Restart Route
                    </button>
                    <button 
                      type="button"
                      onClick={() => setShowConfirm(false)}
                      className="px-6 py-3 bg-card-bg hover:bg-border-col text-text-main rounded-xl font-bold border border-border-col transition-all active:scale-95"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}
          </div>
      </div>
    </div>
  );
};

export default PlanDashboard;