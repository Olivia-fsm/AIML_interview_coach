import { 
  GoogleGenAI, 
  Type, 
  LiveServerMessage,
  Modality,
  Blob,
  GenerateContentResponse
} from "@google/genai";
import { PrepPlan, CodeFeedback, InterviewTurn, InterviewReport } from "../types";

// Helper to get API client
const getAiClient = () => {
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    throw new Error("API Key not found");
  }
  return new GoogleGenAI({ apiKey });
};

// --- Code Evaluation ---
export const evaluateCodeSubmission = async (
  problemTitle: string,
  problemDesc: string,
  userCode: string
): Promise<CodeFeedback> => {
  const ai = getAiClient();
  const prompt = `
    You are a Senior Machine Learning Engineer interviewing a candidate.
    
    Problem: ${problemTitle}
    Description: ${problemDesc}
    
    Candidate Code:
    ${userCode}
    
    Analyze the code for:
    1. Correctness (Does it solve the math/logic correctly?)
    2. Efficiency (Time/Space complexity, vectorization usage)
    3. Style (Pythonic conventions, variable naming)
    
    Return the result in strict JSON format.
  `;

  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          correctnessScore: { type: Type.INTEGER, description: "Score from 0 to 100" },
          isCorrect: { type: Type.BOOLEAN },
          timeComplexity: { type: Type.STRING },
          spaceComplexity: { type: Type.STRING },
          analysis: { type: Type.STRING, description: "Brief paragraph analyzing the approach." },
          improvements: { 
            type: Type.ARRAY, 
            items: { type: Type.STRING },
            description: "List of 1-3 specific improvements." 
          }
        }
      }
    }
  });

  if (!response.text) throw new Error("Analysis failed");
  return JSON.parse(response.text) as CodeFeedback;
};

// --- Plan Generation ---
export const generateStudyPlan = async (
  jobDescription: string, 
  topics: string, 
  interviewDate: string
): Promise<PrepPlan> => {
  const ai = getAiClient();
  const prompt = `
    Create a detailed interview preparation plan for an AI/ML role.
    
    Job Description: ${jobDescription}
    Interested Topics: ${topics}
    Interview Date: ${interviewDate}
    
    Today is ${new Date().toDateString()}.
    
    Generate a day-by-day plan leading up to the interview. 
    If the interview is far away, group by weeks, but return a list of "PlanDay" items.
    Focus on coding problems, system design, and ML concepts.
  `;

  const response = await ai.models.generateContent({
    model: "gemini-3-pro-preview",
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
  return JSON.parse(response.text) as PrepPlan;
};

// --- Research / Search Grounding ---
export const researchTopic = async (query: string) => {
  const ai = getAiClient();
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: query,
    config: {
      tools: [{ googleSearch: {} }]
    }
  });
  
  return {
    text: response.text,
    sources: response.candidates?.[0]?.groundingMetadata?.groundingChunks?.map((chunk: any) => chunk.web).filter(Boolean) || []
  };
};

// --- Tutor Chat ---
export const getTutorResponse = async (history: {role: string, parts: {text: string}[]}[], message: string) => {
  const ai = getAiClient();
  const chat = ai.chats.create({
    model: "gemini-3-pro-preview",
    history: history,
    config: {
      systemInstruction: "You are an expert AI/ML technical interviewer and tutor. Help the user solve coding problems, understand complex algorithms, and prepare for their interview. Be concise but deep."
    }
  });
  
  const result = await chat.sendMessage({ message });
  return result.text;
};

// --- Visual Lab: Image Gen ---
export const generateConceptImage = async (prompt: string, size: '1K' | '2K' | '4K' = '1K') => {
  // Always ensure we have a valid key selected for image generation to avoid free tier limits
  if (window.aistudio && !await window.aistudio.hasSelectedApiKey()) {
     await window.aistudio.openSelectKey();
  }
  
  const ai = getAiClient();
  const model = size === '1K' ? "gemini-2.5-flash-image" : "gemini-3-pro-image-preview";
  
  const config = size === '1K' 
    ? { imageConfig: { aspectRatio: "16:9" } } 
    : { imageConfig: { imageSize: size, aspectRatio: "16:9" } };

  const response = await ai.models.generateContent({
    model,
    contents: prompt,
    config
  });
  
  for (const part of response.candidates?.[0]?.content?.parts || []) {
    if (part.inlineData) {
      return `data:image/png;base64,${part.inlineData.data}`;
    }
  }
  return null;
};

// --- Visual Lab: Image Edit ---
export const editConceptImage = async (base64Image: string, prompt: string) => {
  // Always ensure we have a valid key selected for image editing
  if (window.aistudio && !await window.aistudio.hasSelectedApiKey()) {
     await window.aistudio.openSelectKey();
  }

  const ai = getAiClient();
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash-image",
    contents: {
      parts: [
        { inlineData: { mimeType: "image/png", data: base64Image } },
        { text: prompt }
      ]
    }
  });

  for (const part of response.candidates?.[0]?.content?.parts || []) {
    if (part.inlineData) {
      return `data:image/png;base64,${part.inlineData.data}`;
    }
  }
  return null;
};

// --- Visual Lab: Video Gen (Veo) ---
export const generateConceptVideo = async (prompt: string, imageBase64?: string) => {
  // Need to ensure the user has selected a paid key for Veo
  if (window.aistudio && !await window.aistudio.hasSelectedApiKey()) {
     await window.aistudio.openSelectKey();
  }

  // Re-init with potentially new key context if needed
  const aiWithKey = getAiClient();

  let operation;
  
  if (imageBase64) {
      operation = await aiWithKey.models.generateVideos({
      model: 'veo-3.1-fast-generate-preview',
      prompt: prompt,
      image: {
        imageBytes: imageBase64,
        mimeType: 'image/png'
      },
      config: {
        numberOfVideos: 1,
        resolution: '720p',
        aspectRatio: '16:9'
      }
    });
  } else {
    operation = await aiWithKey.models.generateVideos({
      model: 'veo-3.1-fast-generate-preview',
      prompt: prompt,
      config: {
        numberOfVideos: 1,
        resolution: '720p',
        aspectRatio: '16:9'
      }
    });
  }

  // Polling
  while (!operation.done) {
    await new Promise(resolve => setTimeout(resolve, 5000));
    operation = await aiWithKey.operations.getVideosOperation({ operation });
  }

  const videoUri = operation.response?.generatedVideos?.[0]?.video?.uri;
  if (!videoUri) throw new Error("Video generation failed");

  const finalUrl = `${videoUri}&key=${process.env.API_KEY}`;
  const fetchResponse = await fetch(finalUrl);
  const blob = await fetchResponse.blob();
  return URL.createObjectURL(blob);
};

// --- Interview Report Generation ---
export const generateInterviewReport = async (turns: InterviewTurn[]): Promise<InterviewReport> => {
    const ai = getAiClient();
    
    const conversationText = turns.map(t => `${t.role.toUpperCase()}: ${t.text}`).join('\n');
    
    const prompt = `
      Analyze the following transcript from a technical mock interview for an ML Engineer role.
      
      TRANSCRIPT:
      ${conversationText}
      
      Generate a detailed feedback report in JSON format.
      Identify specific questions asked and the quality of the candidate's answers.
      Provide an expected "ideal" answer for each question.
      Give an overall score (0-100).
    `;

    const response = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: prompt,
        config: {
            responseMimeType: "application/json",
            responseSchema: {
                type: Type.OBJECT,
                properties: {
                    overallScore: { type: Type.INTEGER },
                    summary: { type: Type.STRING },
                    strengths: { type: Type.ARRAY, items: { type: Type.STRING } },
                    weaknesses: { type: Type.ARRAY, items: { type: Type.STRING } },
                    qna: {
                        type: Type.ARRAY,
                        items: {
                            type: Type.OBJECT,
                            properties: {
                                question: { type: Type.STRING },
                                userAnswer: { type: Type.STRING, description: "Summary of user's answer" },
                                expectedAnswer: { type: Type.STRING },
                                feedback: { type: Type.STRING }
                            }
                        }
                    }
                }
            }
        }
    });

    if (!response.text) throw new Error("Failed to generate report");
    return JSON.parse(response.text) as InterviewReport;
}


// --- Live API Helper ---
export class LiveClient {
    private sessionPromise: Promise<any> | null = null;
    private inputAudioContext: AudioContext | null = null;
    private outputAudioContext: AudioContext | null = null;
    private nextStartTime = 0;
    private sources = new Set<AudioBufferSourceNode>();
    
    // Accumulators for transcription
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
          onopen: () => {
            console.log("Live Session Open");
            this.startAudioStream(stream);
          },
          onmessage: (message: LiveServerMessage) => this.handleMessage(message),
          onclose: () => console.log("Live Session Closed"),
          onerror: (e) => console.error("Live Session Error", e),
        },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } },
          },
          systemInstruction: "You are a tough but fair technical interviewer conducting a mock interview for a Senior Machine Learning Engineer role. Ask about transformers, gradients, and system design. Keep responses conversational and concise.",
          inputAudioTranscription: {}, // Enable Input Transcription
          outputAudioTranscription: {}, // Enable Output Transcription
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
        this.sessionPromise?.then((session) => {
          session.sendRealtimeInput({ media: pcmBlob });
        });
      };
      
      source.connect(scriptProcessor);
      scriptProcessor.connect(this.inputAudioContext.destination);
    }
  
    private async handleMessage(message: LiveServerMessage) {
       // Handle Audio
       const audioData = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
       if (audioData) {
         this.onAudioData(audioData); // For visualization
         await this.playAudio(audioData);
       }

       // Handle Interruptions
       if (message.serverContent?.interrupted) {
         this.stopAudio();
         this.currentOutputTranscription = ''; // Clear stale output
       }

       // Handle Transcription Accumulation
       if (message.serverContent?.outputTranscription) {
           this.currentOutputTranscription += message.serverContent.outputTranscription.text;
       }
       if (message.serverContent?.inputTranscription) {
           this.currentInputTranscription += message.serverContent.inputTranscription.text;
       }

       // Handle Turn Completion
       if (message.serverContent?.turnComplete) {
           if (this.currentInputTranscription.trim()) {
               this.onTranscript({
                   role: 'user',
                   text: this.currentInputTranscription,
                   timestamp: Date.now()
               });
               this.currentInputTranscription = '';
           }
           if (this.currentOutputTranscription.trim()) {
               this.onTranscript({
                   role: 'model',
                   text: this.currentOutputTranscription,
                   timestamp: Date.now()
               });
               this.currentOutputTranscription = '';
           }
       }
    }
  
    private async playAudio(base64: string) {
      if (!this.outputAudioContext) return;
      
      this.nextStartTime = Math.max(this.nextStartTime, this.outputAudioContext.currentTime);
      
      const binaryString = atob(base64);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      const audioBuffer = await this.decodeAudioData(bytes, this.outputAudioContext);
      
      const source = this.outputAudioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.outputAudioContext.destination);
      source.onended = () => this.sources.delete(source);
      
      source.start(this.nextStartTime);
      this.nextStartTime += audioBuffer.duration;
      this.sources.add(source);
    }

    private stopAudio() {
        this.sources.forEach(s => s.stop());
        this.sources.clear();
        this.nextStartTime = 0;
    }
  
    private createBlob(data: Float32Array): Blob {
      const l = data.length;
      const int16 = new Int16Array(l);
      for (let i = 0; i < l; i++) {
        int16[i] = data[i] * 32768;
      }
      let binary = '';
      const bytes = new Uint8Array(int16.buffer);
      const len = bytes.byteLength;
      for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      return {
        data: btoa(binary),
        mimeType: 'audio/pcm;rate=16000',
      };
    }

    private async decodeAudioData(data: Uint8Array, ctx: AudioContext): Promise<AudioBuffer> {
      const dataInt16 = new Int16Array(data.buffer);
      const frameCount = dataInt16.length;
      const buffer = ctx.createBuffer(1, frameCount, 24000);
      const channelData = buffer.getChannelData(0);
      for (let i = 0; i < frameCount; i++) {
        channelData[i] = dataInt16[i] / 32768.0;
      }
      return buffer;
    }

    disconnect() {
        this.stopAudio();
        this.inputAudioContext?.close();
        this.outputAudioContext?.close();
    }
}