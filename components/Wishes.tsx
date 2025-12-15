import React, { useState } from 'react';

const Wishes: React.FC = () => {
  const [email, setEmail] = useState('');
  const [request, setRequest] = useState('');
  const [sent, setSent] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !request) return;

    // Construct mailto link
    const subject = encodeURIComponent("Love&DeepCode Feature Wish");
    const body = encodeURIComponent(`Sender Email: ${email}\n\nMy Wish/Request:\n${request}\n\n----------------\nSent from Love&DeepCode App`);
    
    // Open email client
    window.location.href = `mailto:oliviafan1999@gmail.com?subject=${subject}&body=${body}`;
    
    // Show local success state
    setSent(true);
    setTimeout(() => setSent(false), 5000);
  };

  return (
    <div className="h-full p-8 overflow-y-auto flex flex-col items-center justify-center animate-fade-in">
      <div className="max-w-2xl w-full bg-panel-bg border border-border-col rounded-2xl shadow-2xl p-8 relative overflow-hidden">
        
        {/* Decorative Background Elements */}
        <div className="absolute top-0 right-0 -mt-10 -mr-10 w-40 h-40 bg-primary/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 left-0 -mb-10 -ml-10 w-40 h-40 bg-purple-500/10 rounded-full blur-3xl"></div>

        <div className="relative z-10 text-center mb-8">
          <div className="w-20 h-20 bg-card-bg border border-border-col rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg text-4xl">
            ✨
          </div>
          <h2 className="text-3xl font-bold text-text-main mb-2">Make a Wish</h2>
          <p className="text-text-muted">
            Missing a specific algorithm? Want a new feature? <br/>
            Let us know and help shape the future of Love&DeepCode.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6 relative z-10">
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-text-muted mb-2">
              Your Email Address
            </label>
            <input
              type="email"
              id="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="researcher@example.com"
              className="w-full bg-card-bg border border-border-col rounded-lg px-4 py-3 text-text-main focus:ring-2 focus:ring-primary focus:border-transparent transition-all outline-none"
            />
          </div>

          <div>
            <label htmlFor="request" className="block text-sm font-medium text-text-muted mb-2">
              Your Wish / Request
            </label>
            <textarea
              id="request"
              required
              value={request}
              onChange={(e) => setRequest(e.target.value)}
              placeholder="I wish there were more questions about Diffusion Transformers or System Design for RecSys..."
              rows={5}
              className="w-full bg-card-bg border border-border-col rounded-lg px-4 py-3 text-text-main focus:ring-2 focus:ring-primary focus:border-transparent transition-all outline-none resize-none"
            />
          </div>

          <button
            type="submit"
            className="w-full py-4 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-400 hover:to-purple-500 text-white font-bold rounded-xl shadow-lg transform transition-all hover:scale-[1.01] flex items-center justify-center gap-2"
          >
            {sent ? (
              <>
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" /></svg>
                Opened Mail Client!
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" /></svg>
                Send Request
              </>
            )}
          </button>
          
          <p className="text-center text-xs text-text-muted mt-4">
            This will open your default email client to send a message to <span className="text-primary">oliviafan1999@gmail.com</span>
          </p>
        </form>
      </div>
    </div>
  );
};

export default Wishes;