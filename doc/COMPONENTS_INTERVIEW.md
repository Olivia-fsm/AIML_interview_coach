# Interview & Research Components

## `MockInterview.tsx`
Handles the live voice simulation.
- **State**:
    - `isActive`: Boolean flag for session status.
    - `report`: Stores the final feedback report after session ends.
- **Audio Visualization**: Uses HTML5 Canvas to draw frequency bars based on audio amplitude.
- **Flow**:
    1. User clicks "Start Interview".
    2. Instantiates `LiveClient` and connects.
    3. User speaks; Model responds via audio.
    4. Transcripts are accumulated.
    5. On "End Session", calls `generateInterviewReport` to analyze the full transcript.

## `ResearchTool.tsx`
Provides grounded knowledge retrieval.
- **Input**: Query text field.
- **Display**: Renders Markdown response from the model.
- **Sources**: Lists citations/links returned by the `googleSearch` tool grounding metadata.
