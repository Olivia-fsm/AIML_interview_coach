# Coding Components

## `ProblemBank.tsx`
Displays the library of coding questions.
- **Categorization**: Filters problems by categories (SL, UL, DL, NLP, CV, RL, Reasoning, Architecture, System Design).
- **State**: Tracks solved status based on `submissions` prop.
- **UI**: Grid layout of problem cards showing difficulty and title.

## `ProblemSolver.tsx`
The main interface for solving a selected problem.
- **Code Editor**: Simple textarea for writing Python code.
- **Problem Details**: Shows description, examples, and expandable hints.
- **Submission**: Calls `evaluateCodeSubmission` service.
- **Feedback Overlay**: Displays the AI-generated analysis (Correctness Score, Time/Space Complexity) and Reference Solution.

## `data/problems.ts`
Contains the `PROBLEM_LIBRARY` array.
- **Structure**:
    - `id`: Unique identifier.
    - `starterCode`: Python function signature.
    - `solution`: Reference implementation.
    - `category`: Domain (e.g., 'Reinforcement Learning').
    - `difficulty`: 'Easy' | 'Medium' | 'Hard'.
- **Content**: Includes advanced topics like PPO, DPO, GRPO, MoE Routing, KV Cache, and Attention mechanisms.
