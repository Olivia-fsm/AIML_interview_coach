import React, { useState } from 'react';
import { generateConceptImage, editConceptImage, generateConceptVideo } from '../services/gemini';

const VisualLab: React.FC = () => {
  const [mode, setMode] = useState<'create' | 'edit' | 'video'>('create');
  const [prompt, setPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
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
    try {
      if (mode === 'create') {
        const url = await generateConceptImage(prompt, '1K'); 
        setResultUrl(url);
      } else if (mode === 'edit') {
        if (!uploadData) {
            alert("Please upload an image to edit");
            return;
        }
        const rawBase64 = uploadData.split(',')[1];
        const url = await editConceptImage(rawBase64, prompt);
        setResultUrl(url);
      } else if (mode === 'video') {
         let rawBase64 = undefined;
         if (uploadData) {
             rawBase64 = uploadData.split(',')[1];
         }
         const url = await generateConceptVideo(prompt, rawBase64);
         setResultUrl(url);
      }
    } catch (e) {
      console.error(e);
      alert('Operation failed. Please ensure you have selected a valid API Key with sufficient quota.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-full p-6 overflow-y-auto">
        <div className="max-w-4xl mx-auto space-y-8">
            <header className="mb-6">
                <h2 className="text-3xl font-bold text-text-main mb-2">Visual Concept Lab</h2>
                <p className="text-text-muted">Generate diagrams, edit whiteboard photos, or create explainer videos for your concepts.</p>
            </header>

            <div className="flex gap-4 mb-6 border-b border-border-col pb-4">
                {(['create', 'edit', 'video'] as const).map((m) => (
                    <button
                        key={m}
                        onClick={() => { setMode(m); setResultUrl(null); setUploadData(null); }}
                        className={`px-4 py-2 rounded-lg font-medium transition-colors capitalize ${mode === m ? 'bg-primary text-white' : 'text-text-muted hover:text-text-main'}`}
                    >
                        {m === 'video' ? 'Animate (Veo)' : m === 'create' ? 'Generate Image' : 'Edit Image'}
                    </button>
                ))}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
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
                            {mode === 'create' ? 'Describe the diagram/image' : 
                             mode === 'edit' ? 'How should the image be modified?' : 
                             'Describe the video animation'}
                        </label>
                        <textarea
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            className="w-full h-32 bg-card-bg border border-border-col rounded-lg p-3 text-text-main focus:ring-2 focus:ring-primary"
                            placeholder={mode === 'create' ? "A complex neural network architecture diagram..." : "Remove the background and make it look like a sketch..."}
                        />
                    </div>

                    <button
                        onClick={executeAction}
                        disabled={isLoading || (!prompt && mode !== 'edit')}
                        className="w-full py-3 bg-primary hover:opacity-90 rounded-lg font-bold text-white shadow-lg transition-all disabled:opacity-50"
                    >
                        {isLoading ? 'Processing...' : 'Generate'}
                    </button>
                    
                    <p className="text-xs text-text-muted">
                        * Image/Video generation requires a paid API key to avoid quota limits. You will be prompted to select one.
                        {mode === 'video' && " Video generation uses Veo 3.1."}
                    </p>
                </div>

                <div className="bg-card-bg/40 rounded-xl border border-border-col flex items-center justify-center min-h-[400px]">
                    {isLoading ? (
                         <div className="text-center">
                            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
                            <p className="text-text-muted animate-pulse">AI is working...</p>
                         </div>
                    ) : resultUrl ? (
                        mode === 'video' ? (
                            <video controls src={resultUrl} className="max-w-full max-h-full rounded-lg" autoPlay loop />
                        ) : (
                            <img src={resultUrl} alt="Generated result" className="max-w-full max-h-full rounded-lg object-contain" />
                        )
                    ) : (
                        <div className="text-center text-text-muted opacity-50">
                            <svg className="w-16 h-16 mx-auto mb-4" fill="currentColor" viewBox="0 0 24 24"><path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
                            <p>Result will appear here</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    </div>
  );
};

export default VisualLab;