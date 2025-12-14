# Love&DeepCode: AI Career Architect

[**Watch Demo Video**](https://drive.google.com/file/d/1DmVanLcnbRR5c8JqcvFk3cCkvL3f6DGM/view?usp=sharing)

## Project Overview
Love&DeepCode is a comprehensive, AI-powered platform designed to help PhD students and graduates prepare for Machine Learning and AI engineering interviews. It leverages Google's latest Gemini models to provide personalized study plans, live voice-based mock interviews, coding challenges, and visual concept generation.

## Technology Stack
- **Frontend Framework**: React 19
- **Styling**: Tailwind CSS
- **AI Integration**: Google GenAI SDK (`@google/genai`)
- **Visualization**: Recharts (Charts), HTML5 Canvas (Animations)
- **Build Tool**: Vite / ES Modules

## Key Features
1. **Personalized Curriculum**: Generates day-by-day study plans based on job descriptions (Gemini 3 Pro).
2. **Practice Bank**: A curated library of advanced ML coding problems (Transformers, RL, System Design).
3. **Mock Interview**: Real-time voice interaction with an AI interviewer using the **Gemini Live API**.
4. **Visual Lab**: Generates architecture diagrams and explainer videos using **Imagen** and **Veo**.
5. **Research Tool**: Grounded search for latest research papers and interview trends using Google Search Tool.
6. **Theming Engine**: 10+ distinct themes with interactive canvas backgrounds (Cosmic, Gothic, Sea, etc.).

## Directory Structure
- `components/`: React UI components.
- `services/`: API integration logic.
- `data/`: Static assets (problem library).
- `types.ts`: TypeScript definitions.