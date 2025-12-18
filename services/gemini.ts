import { 
  GoogleGenAI, 
  Type, 
  LiveServerMessage, 
  Modality, 
  Blob, 
  GenerateContentResponse 
} from "@google/genai";
import { PrepPlan, CodeFeedback, InterviewTurn, InterviewReport, TestCaseResult, JobPosting } from "../types";

// Helper to get API client - strictly uses environment variable API_KEY in named parameter object
const getAiClient = () => {
  return new GoogleGenAI({ apiKey: process.env.API_KEY });
};

/**
 * Helper: Exponential Backoff Retry Wrapper
 * Retries the provided function if it fails with a 503 (Overloaded) or 429 (Rate Limit) error.
 */
async function withRetry<T>(fn: () => Promise<T>, retries = 3, delay = 1000): Promise<T> {
  try {
    return await fn();
  } catch (error: any) {
    const isRetryable = error.message?.includes("503") || 
                        error.message?.includes("overloaded") || 
                        error.message?.includes("429") ||
                        error.message?.includes("rate limit");

    if (retries > 0 && isRetryable) {
      console.warn(`Gemini API busy. Retrying in ${delay}ms... (${retries} retries left)`);
      await new Promise(resolve => setTimeout(resolve, delay));
      return withRetry(fn, retries - 1, delay * 2);
    }
    throw error;
  }
}

// --- Helper: Robust JSON Parser ---
const parseGeminiJson = <T>(text: string): T => {
    try {
        const match = text.match(/```json\s*([\s\S]*?)\s*```/) || text.match(/```\s*([\s\S]*?)\s*```/);
        if (match) {
            return JSON.parse(match[1]);
        }
        return JSON.parse(text);
    } catch (e) {
        console.error("JSON Parse failed for text:", text);
        throw new Error("Failed to parse AI response: Invalid JSON format");
    }
};

// --- Helper: Domain Validation & Prompt Refinement ---
const validateAndRefinePrompt = async (userPrompt: string, targetType: 'image' | 'video'): Promise<{isValid: boolean, refusalReason?: string, lectureNotes: string, visualPrompt: string}> => {
  return withRetry(async () => {
    const ai = getAiClient();
    const metaPrompt = `
      You are an AI/ML education supervisor.
      User Query: "${userPrompt}"
      Target Media: ${targetType}
      
      Task:
      1. VALIDATION: Determine if the query is related to AI, Machine Learning, Data Science, Math, Algorithms, System Design, or Coding.
      2. EXPLANATION: Write concise "Lecture Notes" (Markdown) explaining the technical concept.
      3. VISUAL_PROMPT: Write an optimized prompt for ${targetType} generation.
      
      Return strict JSON:
      {
        "isValid": boolean,
        "refusalReason": string,
        "lectureNotes": string,
        "visualPrompt": string
      }
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview", // Updated to gemini-3 series for basic text tasks
      contents: metaPrompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            isValid: { type: Type.BOOLEAN },
            refusalReason: { type: Type.STRING },
            lectureNotes: { type: Type.STRING },
            visualPrompt: { type: Type.STRING }
          }
        }
      }
    });

    if (!response.text) throw new Error("Validation failed");
    const result = JSON.parse(response.text);
    
    if (!result.isValid) {
        throw new Error(`Content Blocked: ${result.refusalReason || "Topic is unrelated to Machine Learning curriculum."}`);
    }
    
    return result;
  });
};

// --- Code Evaluation ---
export const evaluateCodeSubmission = async (
  problemTitle: string,
  problemDesc: string,
  userCode: string
): Promise<CodeFeedback> => {
  return withRetry(async () => {
    const ai = getAiClient();
    const prompt = `
      You are a Senior Machine Learning Engineer interviewing a candidate.
      Problem: ${problemTitle}
      Description: ${problemDesc}
      Candidate Code: ${userCode}
      Analyze for Correctness, Efficiency, and Style.
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3-pro-preview", // Updated to gemini-3 series for complex coding analysis
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            correctnessScore: { type: Type.INTEGER },
            isCorrect: { type: Type.BOOLEAN },
            timeComplexity: { type: Type.STRING },
            spaceComplexity: { type: Type.STRING },
            analysis: { type: Type.STRING },
            improvements: { type: Type.ARRAY, items: { type: Type.STRING } }
          }
        }
      }
    });

    if (!response.text) throw new Error("Analysis failed");
    const parsed = parseGeminiJson<any>(response.text);

    return {
      correctnessScore: typeof parsed.correctnessScore === 'number' ? parsed.correctnessScore : 0,
      isCorrect: !!parsed.isCorrect,
      timeComplexity: String(parsed.timeComplexity || '?'),
      spaceComplexity: String(parsed.spaceComplexity || '?'),
      analysis: String(parsed.analysis || "No analysis provided."),
      improvements: Array.isArray(parsed.improvements) ? parsed.improvements.map((i: any) => String(i)) : []
    };
  });
};

// --- Run Test Cases ---
export const runCodeAgainstTests = async (
  code: string,
  testCases: { input: string; output: string }[],
  problemTitle: string,
  hiddenTestCase?: { input: string; output: string }
): Promise<TestCaseResult[]> => {
  return withRetry(async () => {
    const ai = getAiClient();
    const allTests = hiddenTestCase ? [...testCases, { ...hiddenTestCase, isHidden: true }] : testCases;
    const testCasesStr = JSON.stringify(allTests);
    
    const prompt = `
      You are a Python execution engine.
      Problem: ${problemTitle}
      Code: ${code}
      Test Cases: ${testCasesStr}
      Execute and return result in JSON format: [{"actual": string, "passed": boolean, "logs": string}]
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview", // Using flash for code execution/verification
      contents: prompt,
      config: { tools: [{ codeExecution: {} }] }
    });

    const text = response.text || "";
    let aiResults = parseGeminiJson<any[]>(text);

    return allTests.map((tc, index) => {
        const res = aiResults[index] || { actual: "Execution failed", passed: false, logs: "No output" };
        return {
            input: tc.input,
            expected: tc.output,
            actual: typeof res.actual === 'object' ? JSON.stringify(res.actual) : String(res.actual || "No output"),
            passed: !!res.passed,
            logs: typeof res.logs === 'object' ? JSON.stringify(res.logs) : String(res.logs || ""),
            isHidden: (tc as any).isHidden
        };
    });
  });
};

// --- Plan Generation ---
export const generateStudyPlan = async (
  jobDescription: string, 
  topics: string, 
  interviewDate: string
): Promise<PrepPlan> => {
  return withRetry(async () => {
    const ai = getAiClient();
    const today = new Date().toISOString().split('T')[0];
    
    const prompt = `
      Create a detailed interview preparation plan for an AI/ML role.
      TODAY'S DATE: ${today} (Use this as the absolute reference point for all date calculations).
      
      Job Description: ${jobDescription}
      Interested Topics: ${topics}
      Interview Date Target: ${interviewDate} (If relative like "in 2 weeks", calculate it from TODAY'S DATE: ${today}).
      
      Generate a condensed plan (max 14 days). Ensure the 'date' field in the schedule follows the YYYY-MM-DD format based on your calculation.
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview", // Updated to gemini-3 series
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            roleTitle: { type: Type.STRING },
            targetCompany: { type: Type.STRING },
            interviewDate: { type: Type.STRING },
            summary: { type: Type.STRING },
            schedule: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  day: { type: Type.INTEGER },
                  date: { type: Type.STRING },
                  focusArea: { type: Type.STRING },
                  tasks: { type: Type.ARRAY, items: { type: Type.STRING } },
                  resources: { type: Type.ARRAY, items: { type: Type.STRING } },
                }
              }
            }
          }
        }
      }
    });

    if (!response.text) throw new Error("No plan generated");
    const parsed = parseGeminiJson<any>(response.text);
    return parsed as PrepPlan;
  });
};

export const explainCodeSnippet = async (snippet: string, code: string, title: string) => {
  return withRetry(async () => {
    const ai = getAiClient();
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview", // Updated to gemini-3 series
      contents: `Expert tutor explanation for snippet: ${snippet}\nContext: ${code}\nProblem: ${title}`
    });
    return response.text || "No explanation available.";
  });
};

export const evaluateCodeExplanation = async (snippet: string, explanation: string, code: string, title: string) => {
  return withRetry(async () => {
    const ai = getAiClient();
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview", // Updated to gemini-3 series
      contents: `Evaluate explanation: ${explanation}\nSnippet: ${snippet}\nContext: ${code}`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            isCorrect: { type: Type.BOOLEAN },
            score: { type: Type.INTEGER },
            feedback: { type: Type.STRING }
          }
        }
      }
    });
    return parseGeminiJson<any>(response.text);
  });
};

export const researchTopic = async (query: string) => {
  return withRetry(async () => {
    const ai = getAiClient();
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview", // Updated to gemini-3 series for text search grounding
      contents: query,
      config: { tools: [{ googleSearch: {} }] }
    });
    return {
      text: response.text || "No results found.",
      sources: response.candidates?.[0]?.groundingMetadata?.groundingChunks?.map((chunk: any) => chunk.web).filter(Boolean) || []
    };
  });
};

export const findJobPostings = async (role: string, location: string): Promise<JobPosting[]> => {
  return withRetry(async () => {
    const ai = getAiClient();
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview", // Updated to gemini-3 series
      contents: `Find 6 jobs for ${role} in ${location}. Return JSON array.`,
      config: { tools: [{ googleSearch: {} }] }
    });
    return parseGeminiJson<JobPosting[]>(response.text || "[]");
  });
};

export const getTutorResponse = async (history: any[], message: string) => {
  return withRetry(async () => {
    const ai = getAiClient();
    const chat = ai.chats.create({
      model: "gemini-3-pro-preview", // Using pro for high-quality chat tutoring
      history: history,
      config: { systemInstruction: "You are an expert ML interviewer and tutor." }
    });
    const result = await chat.sendMessage({ message });
    return result.text || ""; // Accessed as property, not method
  });
};

export const generateConceptImage = async (prompt: string, size: '1K' | '2K' | '4K' = '1K') => {
  const plan = await validateAndRefinePrompt(prompt, 'image');
  if (window.aistudio && !await window.aistudio.hasSelectedApiKey()) await window.aistudio.openSelectKey();
  
  return withRetry(async () => {
    const ai = getAiClient();
    const model = size === '1K' ? "gemini-2.5-flash-image" : "gemini-3-pro-image-preview";
    const response = await ai.models.generateContent({
      model,
      contents: plan.visualPrompt,
      config: { imageConfig: { aspectRatio: "16:9", ...(size !== '1K' && { imageSize: size }) } }
    });
    
    let imageUrl = null;
    for (const part of response.candidates?.[0]?.content?.parts || []) {
      if (part.inlineData) imageUrl = `data:image/png;base64,${part.inlineData.data}`;
    }
    return { imageUrl, explanation: plan.lectureNotes };
  });
};

export const editConceptImage = async (base64Image: string, prompt: string) => {
  if (window.aistudio && !await window.aistudio.hasSelectedApiKey()) await window.aistudio.openSelectKey();
  return withRetry(async () => {
    const ai = getAiClient();
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash-image",
      contents: { parts: [{ inlineData: { mimeType: "image/png", data: base64Image } }, { text: prompt }] }
    });
    let imageUrl = null;
    for (const part of response.candidates?.[0]?.content?.parts || []) {
      if (part.inlineData) imageUrl = `data:image/png;base64,${part.inlineData.data}`;
    }
    return { imageUrl, explanation: "Image edited." };
  });
};

export const generateConceptVideo = async (prompt: string, imageBase64?: string) => {
  let visualPrompt = prompt;
  let explanation = "Video generated.";
  if (!imageBase64) {
      const plan = await validateAndRefinePrompt(prompt, 'video');
      visualPrompt = plan.visualPrompt;
      explanation = plan.lectureNotes;
  }
  if (window.aistudio && !await window.aistudio.hasSelectedApiKey()) await window.aistudio.openSelectKey();

  return withRetry(async () => {
    const aiWithKey = getAiClient(); // Create new instance right before use
    let operation = await aiWithKey.models.generateVideos({
      model: 'veo-3.1-fast-generate-preview',
      prompt: visualPrompt,
      ...(imageBase64 && { image: { imageBytes: imageBase64, mimeType: 'image/png' } }),
      config: { numberOfVideos: 1, resolution: '720p', aspectRatio: '16:9' }
    });
    while (!operation.done) {
      await new Promise(resolve => setTimeout(resolve, 5000));
      operation = await aiWithKey.operations.getVideosOperation({ operation });
    }
    const videoUri = operation.response?.generatedVideos?.[0]?.video?.uri;
    const finalUrl = `${videoUri}&key=${process.env.API_KEY}`;
    const fetchResponse = await fetch(finalUrl);
    const blob = await fetchResponse.blob();
    return { videoUrl: URL.createObjectURL(blob), explanation };
  });
};

export const generateInterviewReport = async (turns: InterviewTurn[]): Promise<InterviewReport> => {
    return withRetry(async () => {
        const ai = getAiClient();
        const response = await ai.models.generateContent({
            model: "gemini-3-pro-preview", // Complex analysis task
            contents: `Analyze interview transcript and return report JSON.\n\n${turns.map(t => `${t.role}: ${t.text}`).join('\n')}`,
            config: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        overallScore: { type: Type.INTEGER },
                        summary: { type: Type.STRING },
                        strengths: { type: Type.ARRAY, items: { type: Type.STRING } },
                        weaknesses: { type: Type.ARRAY, items: { type: Type.STRING } },
                        qna: { type: Type.ARRAY, items: { type: Type.OBJECT, properties: { question: { type: Type.STRING }, userAnswer: { type: Type.STRING }, expectedAnswer: { type: Type.STRING }, feedback: { type: Type.STRING } } } }
                    }
                }
            }
        });
        return parseGeminiJson<any>(response.text);
    });
};

export class LiveClient {
    private sessionPromise: Promise<any> | null = null;
    private inputAudioContext: AudioContext | null = null;
    private outputAudioContext: AudioContext | null = null;
    private nextStartTime = 0;
    private sources = new Set<AudioBufferSourceNode>();
    private currentInputTranscription = '';
    private currentOutputTranscription = '';

    constructor(
      private onAudioData: (base64: string) => void,
      private onTranscript: (turn: InterviewTurn) => void
    ) {}
  
    async connect() {
      const ai = getAiClient();
      this.inputAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      this.outputAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.sessionPromise = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
        callbacks: {
          onopen: () => this.startAudioStream(stream),
          onmessage: (message: LiveServerMessage) => this.handleMessage(message),
          onclose: () => console.log("Live Closed"),
          onerror: (e) => console.error("Live Error", e),
        },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } } },
          systemInstruction: "You are an ML interviewer. Be friendly but ask probing technical questions.",
          inputAudioTranscription: {},
          outputAudioTranscription: {},
        },
      });
    }
  
    private startAudioStream(stream: MediaStream) {
      if (!this.inputAudioContext) return;
      const source = this.inputAudioContext.createMediaStreamSource(stream);
      const scriptProcessor = this.inputAudioContext.createScriptProcessor(4096, 1, 1);
      scriptProcessor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        const pcmBlob = this.createBlob(inputData);
        // CRITICAL: Solely rely on sessionPromise resolves to send realtime input
        this.sessionPromise?.then((session) => session.sendRealtimeInput({ media: pcmBlob }));
      };
      source.connect(scriptProcessor);
      scriptProcessor.connect(this.inputAudioContext.destination);
    }
  
    private async handleMessage(message: LiveServerMessage) {
       // Access audio data part from candidates
       const audioData = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
       if (audioData) {
         this.onAudioData(audioData);
         await this.playAudio(audioData);
       }
       if (message.serverContent?.interrupted) {
         this.stopAudio();
         this.currentOutputTranscription = '';
       }
       if (message.serverContent?.outputTranscription) this.currentOutputTranscription += message.serverContent.outputTranscription.text;
       if (message.serverContent?.inputTranscription) this.currentInputTranscription += message.serverContent.inputTranscription.text;
       if (message.serverContent?.turnComplete) {
           if (this.currentInputTranscription.trim()) {
               this.onTranscript({ role: 'user', text: this.currentInputTranscription, timestamp: Date.now() });
               this.currentInputTranscription = '';
           }
           if (this.currentOutputTranscription.trim()) {
               this.onTranscript({ role: 'model', text: this.currentOutputTranscription, timestamp: Date.now() });
               this.currentOutputTranscription = '';
           }
       }
    }
  
    private async playAudio(base64: string) {
      if (!this.outputAudioContext) return;
      this.nextStartTime = Math.max(this.nextStartTime, this.outputAudioContext.currentTime);
      // Implementation of manual base64 decoding as per guideline rules
      const binaryString = atob(base64);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
      const audioBuffer = await this.decodeAudioData(bytes, this.outputAudioContext);
      const source = this.outputAudioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.outputAudioContext.destination);
      source.onended = () => this.sources.delete(source);
      // Scheduled playback for gapless audio
      source.start(this.nextStartTime);
      this.nextStartTime += audioBuffer.duration;
      this.sources.add(source);
    }

    private stopAudio() {
        this.sources.forEach(s => {
          try { s.stop(); } catch(e) {}
        });
        this.sources.clear();
        this.nextStartTime = 0;
    }
  
    private createBlob(data: Float32Array): Blob {
      const int16 = new Int16Array(data.length);
      for (let i = 0; i < data.length; i++) int16[i] = data[i] * 32768;
      const bytes = new Uint8Array(int16.buffer);
      // Manual base64 encoding
      let binary = '';
      for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
      return { data: btoa(binary), mimeType: 'audio/pcm;rate=16000' };
    }

    private async decodeAudioData(data: Uint8Array, ctx: AudioContext): Promise<AudioBuffer> {
      const dataInt16 = new Int16Array(data.buffer);
      const buffer = ctx.createBuffer(1, dataInt16.length, 24000);
      const channelData = buffer.getChannelData(0);
      for (let i = 0; i < dataInt16.length; i++) channelData[i] = dataInt16[i] / 32768.0;
      return buffer;
    }

    disconnect() {
        this.stopAudio();
        this.inputAudioContext?.close();
        this.outputAudioContext?.close();
    }
}
