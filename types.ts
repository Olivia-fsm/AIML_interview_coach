export enum AppView {
  SETUP = 'SETUP',
  DASHBOARD = 'DASHBOARD',
  MOCK_INTERVIEW = 'MOCK_INTERVIEW',
  VISUAL_LAB = 'VISUAL_LAB',
  TUTOR = 'TUTOR',
  RESEARCH = 'RESEARCH',
  PROBLEM_BANK = 'PROBLEM_BANK',
  PROBLEM_SOLVER = 'PROBLEM_SOLVER'
}

export type ProblemCategory = 
  | 'Supervised Learning'
  | 'Unsupervised Learning'
  | 'Deep Learning'
  | 'NLP'
  | 'Computer Vision'
  | 'Reinforcement Learning'
  | 'Reasoning';

export interface Problem {
  id: string;
  title: string;
  category: ProblemCategory;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  description: string;
  examples: { input: string; output: string }[];
  hints: string[];
  solution: string;
  starterCode: string;
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