# Love&DeepCode: AI Career Architect

[**Watch Demo Video**](https://drive.google.com/file/d/1DmVanLcnbRR5c8JqcvFk3cCkvL3f6DGM/view?usp=sharing)

## Project Overview
Love&DeepCode is a comprehensive, AI-powered platform designed to help PhD students and graduates prepare for Machine Learning and AI engineering interviews. It leverages Google's latest Gemini models to provide personalized study plans, live voice-based mock interviews, coding challenges, and visual concept generation.

## Technology Stack
- **Frontend Framework**: React 19
- **Styling**: Tailwind CSS
- **AI Integration**: Google GenAI SDK (`@google/genai`)
- **Persistence**: LocalStorage Database Simulation
- **Visualization**: Recharts (Charts), HTML5 Canvas (Animations)
- **Build Tool**: Vite / ES Modules

## Key Features
1. **User Authentication & Profiles**: Secure login system with gamified profiles. Tracks XP, Level, Liked Problems, and Visual Gallery history.
2. **Personalized Curriculum**: Generates day-by-day study plans based on job descriptions (Gemini 3 Pro).
3. **Practice Bank**: A curated library of advanced ML coding problems (Transformers, RL, System Design).
4. **Mock Interview**: Real-time voice interaction with an AI interviewer using the **Gemini Live API**.
5. **Visual Lab**: Generates architecture diagrams and explainer videos using **Imagen** and **Veo**.
6. **Research Tool**: Grounded search for latest research papers and interview trends using Google Search Tool.
7. **Theming Engine**: 10+ distinct themes with interactive canvas backgrounds (Cosmic, Gothic, Sea, etc.).

## Directory Structure
- `components/`: React UI components.
- `services/`: API integration & Local Database logic.
- `data/`: Static assets (problem library).
- `types.ts`: TypeScript definitions.

## ChangeLog

### Recent Updates
- **New Feature: User Profiles & Database Simulation**:
  - Added a complete Auth system (Login/Signup).
  - Implemented `userService` to simulate a database using LocalStorage.
  - Users now have profiles tracking their **XP**, **Level**, **Game High Scores**, **Saved Visuals**, and **Liked Problems**.
  
- **New Feature: Playground**:
  - Added "Neural Navigate", a casual game where users guide a learning agent through neural network layers to relax after studying. Features neon aesthetics, physics-based movement, and high score tracking.

- **New Feature: Job Hunt Intelligence**:
  - Implemented a real-time job search aggregator.
  - Users can search for roles (e.g. "Research Intern") and locations.
  - Utilizes Gemini's Google Search grounding to find postings on LinkedIn, X (Twitter), and Y Combinator.
  - Returns structured cards with AI-generated summaries and direct links.

- **New Feature: Wishes**:
  - Added a dedicated feedback page where users can submit requests for specific algorithms, features, or question types directly via email.
  
- **Coding Console Improvements**:
  - **Scroll Fix**: Resolved layout issues to ensure smooth scrolling within the code editor for long solutions.
  
- **Test Console & Execution**:
  - **Real Code Execution**: Integrated Gemini's `codeExecution` tool to actually run user Python code in a sandboxed environment, providing real runtime results instead of simulated feedback.
  - **Hidden Test Cases**: Added `hiddenTestCase` support for every problem in the library to verify edge cases and robust logic.
  - **Concrete Data**: Updated problem examples in `data/problems.ts` to use real numerical matrices and arrays instead of abstract descriptions, making it easier for users to debug.
  - **Result UI**: Added a dedicated test results panel showing Pass/Fail status, Actual vs. Expected outputs, and execution logs.