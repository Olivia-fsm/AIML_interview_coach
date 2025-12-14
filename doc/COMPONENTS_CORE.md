# Core Components

## `App.tsx`
The root component acting as the main controller.
- **State Management**:
    - `view`: Current screen (`SETUP`, `DASHBOARD`, `PROBLEM_BANK`, etc.).
    - `theme`: Active visual theme.
    - `plan`: The generated study plan data.
    - `submissions`: Array of user code submissions.
- **Routing**: Sidebar navigation switches the `view` state.
- **Layout**: Conditionally renders background components and global overlays (`ClickEffects`).

## `SetupForm.tsx`
The input mechanism for personalization.
- **Inputs**: Job Description, Focus Topics, Interview Date.
- **Action**: Calls `generateStudyPlan` and transitions to the Dashboard upon success.

## `PlanDashboard.tsx`
The user's progress hub.
- **Visualizations**:
    - **Pie Chart**: Solved vs. Remaining problems.
    - **Bar Chart**: Task load per day.
- **Content**: Displays the schedule, daily focus areas, and tasks derived from the Gemini-generated plan.
