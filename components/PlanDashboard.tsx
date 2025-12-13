import React from 'react';
import { PrepPlan, Submission } from '../types';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, PieChart, Pie } from 'recharts';

interface Props {
  plan: PrepPlan;
  submissions: Submission[];
}

const PlanDashboard: React.FC<Props> = ({ plan, submissions }) => {
  // Plan Chart Data
  const scheduleData = plan.schedule.map(d => ({
    name: `Day ${d.day}`,
    tasks: d.tasks.length,
    focus: d.focusArea
  }));

  // Progress Data
  const solvedCount = submissions.filter(s => s.feedback.correctnessScore > 80).length;
  const avgScore = submissions.length > 0 
    ? Math.round(submissions.reduce((acc, curr) => acc + curr.feedback.correctnessScore, 0) / submissions.length)
    : 0;
  
  const pieData = [
      { name: 'Solved', value: solvedCount },
      { name: 'Remaining', value: 15 - solvedCount } // Assuming ~15 problems in library for demo
  ];
  const COLORS = ['#3b82f6', '#1e293b'];

  return (
    <div className="p-6 space-y-8 animate-fade-in h-full overflow-y-auto">
      {/* Header */}
      <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-6 rounded-2xl border border-gray-700 shadow-lg">
        <div className="flex justify-between items-start">
            <div>
                <h2 className="text-3xl font-bold text-white mb-2">{plan.roleTitle}</h2>
                <p className="text-primary font-medium text-lg">Target: {plan.targetCompany}</p>
            </div>
            <div className="text-right">
                <p className="text-sm text-gray-400">Interview Date</p>
                <p className="text-xl font-bold text-white">{plan.interviewDate}</p>
            </div>
        </div>
        <p className="mt-4 text-gray-300 italic">"{plan.summary}"</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Schedule */}
        <div className="lg:col-span-2 space-y-4">
            <h3 className="text-xl font-semibold text-white flex items-center gap-2">
                <span className="bg-primary/20 text-primary p-1 rounded">📅</span> Daily Schedule
            </h3>
            <div className="space-y-4">
                {plan.schedule.map((day) => (
                    <div key={day.day} className="bg-secondary p-5 rounded-xl border border-gray-700 hover:border-primary transition-colors">
                        <div className="flex justify-between mb-3">
                            <span className="font-bold text-white text-lg">{day.date}</span>
                            <span className="px-3 py-1 bg-accent/20 text-accent rounded-full text-xs font-bold uppercase tracking-wider">{day.focusArea}</span>
                        </div>
                        <ul className="space-y-2">
                            {day.tasks.map((task, idx) => (
                                <li key={idx} className="flex items-start gap-2 text-gray-300 text-sm">
                                    <span className="mt-1.5 w-1.5 h-1.5 bg-primary rounded-full shrink-0"></span>
                                    {task}
                                </li>
                            ))}
                        </ul>
                    </div>
                ))}
            </div>
        </div>

        {/* Right Column: Analytics */}
        <div className="space-y-6">
            
            {/* Progress Card */}
            <div className="bg-secondary p-5 rounded-xl border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-4">Coding Progress</h3>
                <div className="h-48 w-full flex items-center justify-center relative">
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={pieData}
                                innerRadius={60}
                                outerRadius={80}
                                paddingAngle={5}
                                dataKey="value"
                            >
                                {pieData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} stroke="none"/>
                                ))}
                            </Pie>
                        </PieChart>
                    </ResponsiveContainer>
                    <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                        <span className="text-3xl font-bold text-white">{solvedCount}</span>
                        <span className="text-xs text-gray-400">Solved</span>
                    </div>
                </div>
                <div className="flex justify-between mt-4 px-4">
                     <div className="text-center">
                         <div className="text-2xl font-bold text-white">{submissions.length}</div>
                         <div className="text-xs text-gray-500">Attempts</div>
                     </div>
                     <div className="text-center">
                         <div className="text-2xl font-bold text-blue-400">{avgScore}%</div>
                         <div className="text-xs text-gray-500">Avg Score</div>
                     </div>
                </div>
            </div>

            {/* Load Chart */}
            <div className="bg-secondary p-5 rounded-xl border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-4">Study Load</h3>
                <div className="h-40 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={scheduleData}>
                            <XAxis dataKey="name" hide />
                            <Tooltip 
                                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', color: '#f8fafc' }}
                                cursor={{fill: '#334155'}}
                            />
                            <Bar dataKey="tasks" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

             <div className="bg-gradient-to-br from-indigo-900 to-purple-900 p-5 rounded-xl border border-indigo-700 text-center">
                <h3 className="text-lg font-bold text-white mb-2">Mock Interview</h3>
                <p className="text-sm text-indigo-200 mb-4">Test your verbal skills with the Live API.</p>
                <div className="inline-block px-4 py-2 bg-white/10 rounded-lg text-xs font-mono text-white">
                    Status: READY
                </div>
            </div>
        </div>
      </div>
    </div>
  );
};

export default PlanDashboard;