# Services Documentation (`services/gemini.ts`)

This file contains all interactions with the Google GenAI SDK.

## Core Functions

### `evaluateCodeSubmission`
- **Model**: `gemini-2.5-flash`
- **Purpose**: Analyzes user code for correctness, complexity, and style.
- **Output**: JSON object (`CodeFeedback`) containing score, analysis, and suggested improvements.

### `generateStudyPlan`
- **Model**: `gemini-3-pro-preview`
- **Purpose**: Creates a structured schedule based on the user's interview date and job description.
- **Output**: JSON object (`PrepPlan`) with daily tasks and focus areas.

### `researchTopic`
- **Model**: `gemini-2.5-flash`
- **Tool**: `googleSearch`
- **Purpose**: Retrieves up-to-date information from the web. Returns text summary and source URLs.

### `generateConceptImage` / `editConceptImage`
- **Models**: `gemini-2.5-flash-image`, `gemini-3-pro-image-preview`
- **Purpose**: Generates diagrams or edits uploaded images.
- **Auth**: Triggers `window.aistudio.openSelectKey()` for paid quota if needed.

### `generateConceptVideo`
- **Model**: `veo-3.1-fast-generate-preview`
- **Purpose**: Generates short video clips from text prompts or image inputs.
- **Implementation**: Uses polling loop to check operation status.

## Live API Client (`LiveClient` Class)

Manages the WebSocket connection for real-time audio interaction.

### Key Methods
- **`connect()`**: Establishes connection to `gemini-2.5-flash-native-audio-preview`. Sets up AudioContext.
- **`startAudioStream()`**: Captures microphone input (`MediaStream`), converts to PCM, and sends `realtimeInput` to the model.
- **`handleMessage()`**: Processes incoming `LiveServerMessage`. Handles audio playback, interruptions, and transcription accumulation.
- **Audio Processing**:
    - **Encoding**: Converts Float32 microphone data to PCM 16-bit Base64.
    - **Decoding**: Converts received Base64 PCM to AudioBuffer for playback.
