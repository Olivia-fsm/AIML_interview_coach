

export enum AppView {
  AUTH = 'AUTH',
  SETUP = 'SETUP',
  DASHBOARD = 'DASHBOARD',
  MOCK_INTERVIEW = 'MOCK_INTERVIEW',
  VISUAL_LAB = 'VISUAL_LAB',
  TUTOR = 'TUTOR',
  RESEARCH = 'RESEARCH',
  PROBLEM_BANK = 'PROBLEM_BANK',
  PROBLEM_SOLVER = 'PROBLEM_SOLVER',
  JOB_HUNT = 'JOB_HUNT',
  WISHES = 'WISHES',
  PLAYGROUND = 'PLAYGROUND',
  PROFILE = 'PROFILE'
}

export type ThemeId = 'midnight' | 'solar' | 'neon' | 'deepspace' | 'toon' | 'cosmic' | 'sea' | 'flower' | 'snow' | 'gothic';

export interface User {
  id: string;
  email: string;
  name: string;
  password?: string; // In a real app, never store plain text!
  avatar?: string;
  joinedAt: number;
}

export interface UserProfile {
  userId: string;
  level: number;
  xp: number;
  likedProblemIds: string[];
  visualHistory: VisualHistoryItem[];
  gameHighScores: Record<string, number>;
  currentPlan?: PrepPlan;
  submissions: Submission[];
}

export interface ThemeColors {
  bgApp: string;
  bgPanel: string;
  bgCard: string;
  textMain: string;
  textMuted: string;
  colPrimary: string;
  borderCol: string;
}

export type ProblemCategory = 
  | 'Supervised Learning'
  | 'Unsupervised Learning'
  | 'Deep Learning'
  | 'NLP'
  | 'Computer Vision'
  | 'Reinforcement Learning'
  | 'Reasoning'
  | 'Architecture'
  | 'System Design';

export interface Problem {
  id: string;
  title: string;
  category: ProblemCategory;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  description: string;
  examples: { input: string; output: string }[];
  hiddenTestCase: { input: string; output: string };
  hints: string[];
  solution: string;
  starterCode: string;
}

export interface TestCaseResult {
  input: string;
  expected: string;
  actual: string;
  passed: boolean;
  logs?: string;
  isHidden?: boolean;
}

export interface CodeFeedback {
  correctnessScore: number; // 0-100
  isCorrect: boolean;
  timeComplexity: string;
  spaceComplexity: string;
  analysis: string;
  improvements: string[];
}

export interface Submission {
  problemId: string;
  code: string;
  feedback: CodeFeedback;
  timestamp: number;
}

export interface PlanDay {
  day: number;
  date: string;
  focusArea: string;
  tasks: string[];
  resources: string[];
}

export interface PrepPlan {
  roleTitle: string;
  targetCompany: string;
  interviewDate: string;
  schedule: PlanDay[];
  summary: string;
}

export interface UserContext {
  jobDescription: string;
  topics: string;
  interviewDate: string;
  hasPlan: boolean;
}

export interface InterviewTurn {
  role: 'user' | 'model';
  text: string;
  timestamp: number;
}

export interface InterviewReport {
  overallScore: number;
  summary: string;
  strengths: string[];
  weaknesses: string[];
  qna: {
    question: string;
    userAnswer: string;
    expectedAnswer: string;
    feedback: string;
  }[];
}

export interface VisualHistoryItem {
  id: string;
  type: 'image' | 'video';
  mode: 'create' | 'edit' | 'video';
  prompt: string;
  mediaUrl: string;
  explanation: string;
  sourceImage?: string;
  timestamp: number;
}

export interface JobPosting {
  title: string;
  company: string;
  location: string;
  summary: string;
  platform: string;
  url: string;
}