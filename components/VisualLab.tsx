import React, { useState } from 'react';
import { generateConceptImage, editConceptImage, generateConceptVideo } from '../services/gemini';

const VisualLab: React.FC = () => {
  const [mode, setMode] = useState<'create' | 'edit' | 'video'>('create');
  const [prompt, setPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  // Results State
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [explanation, setExplanation] = useState<string | null>(null);
  const [uploadData, setUploadData] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64String = reader.result as string;
        setUploadData(base64String);
      };
      reader.readAsDataURL(file);
    }
  };

  const executeAction = async () => {
    if (!prompt) return;
    setIsLoading(true);
    setResultUrl(null);
    setExplanation(null);
    
    try {
      if (mode === 'create') {
        const result = await generateConceptImage(prompt, '1K'); 
        setResultUrl(result.imageUrl);
        setExplanation(result.explanation);
      } else if (mode === 'edit') {
        if (!uploadData) {
            alert("Please upload an image to edit");
            return;
        }
        const rawBase64 = uploadData.split(',')[1];
        const result = await editConceptImage(rawBase64, prompt);
        setResultUrl(result.imageUrl);
        setExplanation(result.explanation);
      } else if (mode === 'video') {
         let rawBase64 = undefined;
         if (uploadData) {
             rawBase64 = uploadData.split(',')[1];
         }
         const result = await generateConceptVideo(prompt, rawBase64);
         setResultUrl(result.videoUrl);
         setExplanation(result.explanation);
      }
    } catch (e: any) {
      console.error(e);
      // Handle the "Content Blocked" error specifically
      if (e.message && e.message.includes("Content Blocked")) {
          alert(e.message);
      } else {
          alert('Operation failed. Please ensure you have selected a valid API Key and try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-full p-6 overflow-y-auto">
        <div className="max-w-5xl mx-auto space-y-8">
            <header className="mb-6">
                <h2 className="text-3xl font-bold text-text-main mb-2">Visual Concept Lab</h2>
                <p className="text-text-muted">
                    Generate technical diagrams and educational videos for ML concepts. 
                    <span className="text-primary block mt-1 text-sm">Now with auto-generated lecture notes!</span>
                </p>
            </header>

            <div className="flex gap-4 mb-6 border-b border-border-col pb-4">
                {(['create', 'edit', 'video'] as const).map((m) => (
                    <button
                        key={m}
                        onClick={() => { setMode(m); setResultUrl(null); setExplanation(null); setUploadData(null); }}
                        className={`px-4 py-2 rounded-lg font-medium transition-colors capitalize ${mode === m ? 'bg-primary text-white' : 'text-text-muted hover:text-text-main'}`}
                    >
                        {m === 'video' ? 'Animate (Veo)' : m === 'create' ? 'Generate Diagram' : 'Edit Image'}
                    </button>
                ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Input Column */}
                <div className="space-y-6">
                    {(mode === 'edit' || mode === 'video') && (
                        <div>
                            <label className="block text-sm font-medium text-text-muted mb-2">
                                {mode === 'video' ? 'Reference Image (Optional)' : 'Source Image'}
                            </label>
                            <input 
                                type="file" 
                                accept="image/*"
                                onChange={handleFileChange}
                                className="block w-full text-sm text-text-muted file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-card-bg file:text-primary hover:file:bg-border-col"
                            />
                            {uploadData && (
                                <div className="mt-4 relative h-40 w-full bg-card-bg rounded-lg overflow-hidden border border-border-col">
                                    <img src={uploadData} alt="Preview" className="w-full h-full object-contain" />
                                </div>
                            )}
                        </div>
                    )}

                    <div>
                        <label className="block text-sm font-medium text-text-muted mb-2">
                            {mode === 'create' ? 'Enter an ML Term or Concept' : 
                             mode === 'edit' ? 'How should the image be modified?' : 
                             'Describe the video animation'}
                        </label>
                        <textarea
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            className="w-full h-32 bg-card-bg border border-border-col rounded-lg p-3 text-text-main focus:ring-2 focus:ring-primary"
                            placeholder={mode === 'create' ? "e.g. Transformer Attention Mechanism, Gradient Descent Valley..." : "e.g. Highlight the residual connections..."}
                        />
                    </div>

                    <button
                        onClick={executeAction}
                        disabled={isLoading || (!prompt && mode !== 'edit')}
                        className="w-full py-3 bg-primary hover:opacity-90 rounded-lg font-bold text-white shadow-lg transition-all disabled:opacity-50"
                    >
                        {isLoading ? (
                            <span className="flex items-center justify-center gap-2">
                                <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                Generating Content...
                            </span>
                        ) : 'Generate'}
                    </button>
                    
                    <p className="text-xs text-text-muted">
                        * Image/Video generation requires a paid API key. 
                        We also validate prompts to ensure they are relevant to ML/AI/Math.
                    </p>
                </div>

                {/* Output Column */}
                <div className="space-y-6">
                    {/* Visual Result */}
                    <div className="bg-card-bg/40 rounded-xl border border-border-col flex items-center justify-center min-h-[300px] lg:min-h-[400px]">
                        {isLoading ? (
                             <div className="text-center p-8">
                                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
                                <p className="text-text-muted animate-pulse">
                                    AI is creating your {mode === 'video' ? 'video' : 'diagram'}...
                                </p>
                             </div>
                        ) : resultUrl ? (
                            mode === 'video' ? (
                                <video controls src={resultUrl} className="max-w-full max-h-full rounded-lg shadow-lg" autoPlay loop />
                            ) : (
                                <img src={resultUrl} alt="Generated result" className="max-w-full max-h-full rounded-lg object-contain shadow-lg" />
                            )
                        ) : (
                            <div className="text-center text-text-muted opacity-50">
                                <svg className="w-16 h-16 mx-auto mb-4" fill="currentColor" viewBox="0 0 24 24"><path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
                                <p>Visuals will appear here</p>
                            </div>
                        )}
                    </div>

                    {/* Lecture Notes */}
                    {explanation && (
                        <div className="bg-panel-bg border border-border-col rounded-xl p-6 shadow-md animate-fade-in">
                            <h3 className="text-lg font-bold text-primary mb-3 flex items-center gap-2">
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" /></svg>
                                Lecture Notes
                            </h3>
                            <div className="prose prose-invert prose-sm max-w-none text-text-muted">
                                <div dangerouslySetInnerHTML={{ __html: explanation.replace(/\n/g, '<br/>') }} />
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    </div>
  );
};

export default VisualLab;