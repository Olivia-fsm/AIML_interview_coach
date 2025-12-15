import React, { useRef, useEffect, useState, useCallback } from 'react';

// --- Simple Audio Engine ---
// We use a class-like structure (or just helper functions) to manage Web Audio API
// to avoid React render cycle issues with audio scheduling.

class SoundEngine {
  ctx: AudioContext | null = null;
  masterGain: GainNode | null = null;
  isMuted: boolean = false;
  
  // Music State
  isPlaying: boolean = false;
  nextNoteTime: number = 0;
  currentNote: number = 0;
  tempo: number = 120;
  timerID: number | null = null;
  
  // A simple minor pentatonic arpeggio sequence (frequencies in Hz)
  // C2, Eb2, F2, G2, Bb2, C3 ...
  sequence = [
    65.41, 77.78, 87.31, 98.00, 116.54, 130.81, 
    116.54, 98.00, 87.31, 77.78, 65.41, 58.27
  ];

  constructor() {
    const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
    if (AudioCtx) {
      this.ctx = new AudioCtx();
      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.value = 0.15; // Low volume for BGM
      this.masterGain.connect(this.ctx.destination);
    }
  }

  init() {
    if (this.ctx?.state === 'suspended') {
      this.ctx.resume();
    }
  }

  toggleMute() {
    this.isMuted = !this.isMuted;
    if (this.masterGain) {
        // Ramp to avoid clicking
        const now = this.ctx?.currentTime || 0;
        this.masterGain.gain.cancelScheduledValues(now);
        this.masterGain.gain.linearRampToValueAtTime(this.isMuted ? 0 : 0.15, now + 0.1);
    }
    return this.isMuted;
  }

  playJump() {
    if (this.isMuted || !this.ctx) return;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    
    osc.connect(gain);
    gain.connect(this.ctx.destination);
    
    osc.type = 'sine';
    const now = this.ctx.currentTime;
    
    // Pitch sweep
    osc.frequency.setValueAtTime(400, now);
    osc.frequency.exponentialRampToValueAtTime(800, now + 0.1);
    
    // Volume envelope
    gain.gain.setValueAtTime(0.1, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.1);
    
    osc.start(now);
    osc.stop(now + 0.1);
  }

  playCrash() {
    if (this.isMuted || !this.ctx) return;
    const bufferSize = this.ctx.sampleRate * 0.5; // 0.5 sec
    const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
    const data = buffer.getChannelData(0);
    
    for (let i = 0; i < bufferSize; i++) {
        data[i] = Math.random() * 2 - 1;
    }

    const noise = this.ctx.createBufferSource();
    noise.buffer = buffer;
    const gain = this.ctx.createGain();
    
    noise.connect(gain);
    gain.connect(this.ctx.destination);
    
    const now = this.ctx.currentTime;
    gain.gain.setValueAtTime(0.2, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.5);
    
    noise.start(now);
  }

  // --- Scheduler for Music ---
  startMusic() {
    if (!this.ctx || this.isPlaying) return;
    this.isPlaying = true;
    this.nextNoteTime = this.ctx.currentTime;
    this.scheduler();
  }

  stopMusic() {
    this.isPlaying = false;
    if (this.timerID) window.clearTimeout(this.timerID);
  }

  private scheduler() {
    if (!this.isPlaying || !this.ctx) return;
    
    // Lookahead: 0.1 seconds
    while (this.nextNoteTime < this.ctx.currentTime + 0.1) {
        this.scheduleNote(this.nextNoteTime);
        this.advanceNote();
    }
    this.timerID = window.setTimeout(() => this.scheduler(), 25);
  }

  private scheduleNote(time: number) {
    if (!this.ctx || !this.masterGain) return;
    
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    
    osc.type = 'sawtooth';
    osc.frequency.value = this.sequence[this.currentNote % this.sequence.length];
    
    // Filter for "synthwave" pluck sound
    const filter = this.ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(800, time);
    filter.frequency.exponentialRampToValueAtTime(100, time + 0.15); // Pluck effect
    
    osc.connect(filter);
    filter.connect(gain);
    gain.connect(this.masterGain);
    
    // Envelope
    gain.gain.setValueAtTime(0, time);
    gain.gain.linearRampToValueAtTime(0.5, time + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.001, time + 0.2);
    
    osc.start(time);
    osc.stop(time + 0.25);
    
    // Bass sub-oscillator
    if (this.currentNote % 4 === 0) {
       const sub = this.ctx.createOscillator();
       const subGain = this.ctx.createGain();
       sub.type = 'square';
       sub.frequency.value = this.sequence[0] / 2; // Octave down
       subGain.gain.setValueAtTime(0.3, time);
       subGain.gain.exponentialRampToValueAtTime(0.001, time + 0.4);
       sub.connect(subGain);
       subGain.connect(this.masterGain);
       sub.start(time);
       sub.stop(time + 0.4);
    }
  }

  private advanceNote() {
    const secondsPerBeat = 60.0 / this.tempo;
    this.nextNoteTime += 0.25 * secondsPerBeat; // 16th notes
    this.currentNote++;
  }
}

const Playground: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const soundRef = useRef<SoundEngine | null>(null);
  
  const [gameState, setGameState] = useState<'START' | 'PLAYING' | 'GAMEOVER'>('START');
  const [score, setScore] = useState(0);
  const [highScore, setHighScore] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [isFirstVisit, setIsFirstVisit] = useState(true);

  // Game Constants
  const GRAVITY = 0.4;
  const JUMP_STRENGTH = -7;
  const SPEED = 3.5;
  const OBSTACLE_WIDTH = 60;
  const OBSTACLE_GAP = 200;
  const OBSTACLE_SPACING = 350;

  // Refs for game loop
  const stateRef = useRef({
    agentY: 300,
    agentVelocity: 0,
    obstacles: [] as { x: number; topHeight: number; passed: boolean }[],
    particles: [] as { x: number; y: number; vx: number; vy: number; life: number; color: string }[],
    frame: 0,
    score: 0,
    isActive: false
  });

  useEffect(() => {
    // Init sound engine once
    soundRef.current = new SoundEngine();
    
    const saved = localStorage.getItem('ldc_high_score');
    if (saved) setHighScore(parseInt(saved));
    
    // Check if user has played before to determine if we show the full tutorial
    const hasPlayed = localStorage.getItem('ldc_has_played');
    if (hasPlayed) setIsFirstVisit(false);

    return () => {
      soundRef.current?.stopMusic();
    };
  }, []);

  const toggleMute = () => {
      if (soundRef.current) {
          const muted = soundRef.current.toggleMute();
          setIsMuted(muted);
      }
  };

  const startGame = () => {
    if (!canvasRef.current) return;
    
    // Init audio context on user gesture
    soundRef.current?.init();
    soundRef.current?.startMusic();
    
    localStorage.setItem('ldc_has_played', 'true');
    setIsFirstVisit(false);

    const height = canvasRef.current.height;
    
    stateRef.current = {
      agentY: height / 2,
      agentVelocity: 0,
      obstacles: [],
      particles: [],
      frame: 0,
      score: 0,
      isActive: true
    };
    
    // Create initial obstacles
    for (let i = 0; i < 3; i++) {
        addObstacle(canvasRef.current.width + 400 + i * OBSTACLE_SPACING, height);
    }
    
    setScore(0);
    setGameState('PLAYING');
  };

  const addObstacle = (x: number, height: number) => {
      const minHeight = 50;
      const maxHeight = height - OBSTACLE_GAP - minHeight;
      const topHeight = Math.random() * (maxHeight - minHeight) + minHeight;
      stateRef.current.obstacles.push({ x, topHeight, passed: false });
  };

  const createExplosion = (x: number, y: number) => {
      soundRef.current?.playCrash();
      for (let i = 0; i < 20; i++) {
          const angle = Math.random() * Math.PI * 2;
          const speed = Math.random() * 5 + 2;
          stateRef.current.particles.push({
              x, y,
              vx: Math.cos(angle) * speed,
              vy: Math.sin(angle) * speed,
              life: 1.0,
              color: Math.random() > 0.5 ? '#f472b6' : '#818cf8' 
          });
      }
  };

  const jump = () => {
      if (stateRef.current.isActive) {
          stateRef.current.agentVelocity = JUMP_STRENGTH;
          soundRef.current?.playJump();
          
          const { agentY } = stateRef.current;
          for(let i=0; i<3; i++) {
             stateRef.current.particles.push({
                 x: 100, 
                 y: agentY, 
                 vx: (Math.random() - 0.5) * 2 - 2, 
                 vy: (Math.random() - 0.5) * 2 + 2,
                 life: 0.5,
                 color: '#ffffff'
             });
          }
      } else if (gameState !== 'PLAYING') {
          startGame();
      }
  };

  useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const handleResize = () => {
          if (containerRef.current && canvas) {
              canvas.width = containerRef.current.clientWidth;
              canvas.height = containerRef.current.clientHeight;
          }
      };
      
      window.addEventListener('resize', handleResize);
      handleResize();

      let animationId = 0;

      const loop = () => {
          if (stateRef.current.isActive) {
            update(canvas.width, canvas.height);
          }
          draw(ctx, canvas.width, canvas.height);
          animationId = requestAnimationFrame(loop);
      };

      loop();

      return () => {
          window.removeEventListener('resize', handleResize);
          cancelAnimationFrame(animationId);
      };
  }, [gameState]); 

  const update = (width: number, height: number) => {
      const state = stateRef.current;
      state.frame++;

      // Physics
      state.agentVelocity += GRAVITY;
      state.agentY += state.agentVelocity;

      // Floor/Ceiling Collision
      if (state.agentY > height - 10 || state.agentY < 10) {
          gameOver();
          return;
      }

      // Obstacles
      if (state.obstacles.length > 0 && state.obstacles[state.obstacles.length - 1].x < width - OBSTACLE_SPACING) {
          addObstacle(width, height);
      }

      state.obstacles.forEach(obs => {
          obs.x -= SPEED;
          
          if (obs.x < 115 && obs.x + OBSTACLE_WIDTH > 85) {
              if (state.agentY - 10 < obs.topHeight || state.agentY + 10 > obs.topHeight + OBSTACLE_GAP) {
                  gameOver();
              }
          }

          if (obs.x + OBSTACLE_WIDTH < 85 && !obs.passed) {
              obs.passed = true;
              state.score++;
              setScore(state.score);
          }
      });

      if (state.obstacles.length > 0 && state.obstacles[0].x < -OBSTACLE_WIDTH) {
          state.obstacles.shift();
      }

      for (let i = state.particles.length - 1; i >= 0; i--) {
          const p = state.particles[i];
          p.x += p.vx;
          p.y += p.vy;
          p.life -= 0.02;
          if (p.life <= 0) state.particles.splice(i, 1);
      }
  };

  const gameOver = () => {
      stateRef.current.isActive = false;
      createExplosion(100, stateRef.current.agentY);
      setGameState('GAMEOVER');
      soundRef.current?.stopMusic();
      
      if (stateRef.current.score > highScore) {
          setHighScore(stateRef.current.score);
          localStorage.setItem('ldc_high_score', stateRef.current.score.toString());
      }
  };

  const draw = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
      // Clear
      ctx.clearRect(0, 0, width, height);

      // Background Grid
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      const offset = (stateRef.current.frame * SPEED * 0.5) % 50;
      
      for(let x = -offset; x < width; x += 50) {
          ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
      }
      for(let y = 0; y < height; y += 50) {
          ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
      }

      // Obstacles
      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg-card').trim() || '#1e293b';
      ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--col-primary').trim() || '#3b82f6';
      ctx.lineWidth = 2;

      stateRef.current.obstacles.forEach(obs => {
          // Danger Color for Barriers
          const barrierColor = '#f43f5e'; // Rose-500
          
          // Top Bar
          ctx.strokeStyle = barrierColor;
          ctx.fillStyle = 'rgba(244, 63, 94, 0.1)';
          ctx.shadowColor = barrierColor;
          ctx.shadowBlur = 10;
          
          ctx.fillRect(obs.x, 0, OBSTACLE_WIDTH, obs.topHeight);
          ctx.strokeRect(obs.x, 0, OBSTACLE_WIDTH, obs.topHeight);
          
          // Bottom Bar
          const botY = obs.topHeight + OBSTACLE_GAP;
          ctx.fillRect(obs.x, botY, OBSTACLE_WIDTH, height - botY);
          ctx.strokeRect(obs.x, botY, OBSTACLE_WIDTH, height - botY);
          
          ctx.shadowBlur = 0;
          
          // Tech markings
          ctx.beginPath();
          ctx.moveTo(obs.x + 10, 0); ctx.lineTo(obs.x + 10, obs.topHeight - 20);
          ctx.moveTo(obs.x + 30, 0); ctx.lineTo(obs.x + 30, obs.topHeight - 40);
          ctx.stroke();
      });

      // Player Trail
      if (stateRef.current.isActive) {
        ctx.beginPath();
        ctx.moveTo(100, stateRef.current.agentY);
        ctx.lineTo(60, stateRef.current.agentY - stateRef.current.agentVelocity * 2);
        ctx.strokeStyle = 'rgba(59, 130, 246, 0.5)';
        ctx.lineWidth = 4;
        ctx.stroke();
      }

      // Player
      if (gameState !== 'GAMEOVER' || stateRef.current.particles.length > 0) {
          if (stateRef.current.isActive) {
            ctx.fillStyle = '#ffffff';
            ctx.shadowBlur = 15;
            ctx.shadowColor = '#3b82f6';
            ctx.beginPath();
            ctx.arc(100, stateRef.current.agentY, 12, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;
            
            // Eye
            ctx.fillStyle = '#000';
            ctx.beginPath();
            ctx.arc(104, stateRef.current.agentY - 2, 3, 0, Math.PI * 2);
            ctx.fill();
          }
      }

      // Particles
      stateRef.current.particles.forEach(p => {
          ctx.globalAlpha = p.life;
          ctx.fillStyle = p.color;
          ctx.beginPath();
          ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
          ctx.fill();
      });
      ctx.globalAlpha = 1.0;
  };

  return (
    <div 
        ref={containerRef} 
        className="w-full h-full relative bg-app-bg select-none overflow-hidden outline-none" 
        onMouseDown={jump}
        onTouchStart={jump}
        tabIndex={0}
        onKeyDown={(e) => { if (e.code === 'Space') jump(); }}
    >
      <canvas ref={canvasRef} className="block" />
      
      {/* HUD */}
      <div className="absolute top-6 left-6 flex gap-4 z-10">
          <div className="bg-panel-bg/80 backdrop-blur border border-border-col px-4 py-2 rounded-lg shadow-lg">
              <span className="text-xs text-text-muted uppercase font-bold block">Current Epoch</span>
              <span className="text-2xl font-bold text-white font-mono">{score}</span>
          </div>
          <div className="bg-panel-bg/80 backdrop-blur border border-border-col px-4 py-2 rounded-lg shadow-lg">
              <span className="text-xs text-text-muted uppercase font-bold block">Best Record</span>
              <span className="text-2xl font-bold text-primary font-mono">{highScore}</span>
          </div>
          
          <button 
             onClick={(e) => { e.stopPropagation(); toggleMute(); }}
             className="bg-panel-bg/80 backdrop-blur border border-border-col w-14 rounded-lg shadow-lg flex items-center justify-center hover:bg-card-bg transition-colors"
          >
             {isMuted ? '🔇' : '🔊'}
          </button>
      </div>

      {/* Start / Tutorial Screen */}
      {gameState === 'START' && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/70 backdrop-blur-sm z-20 animate-fade-in">
              <div className="bg-panel-bg border border-border-col p-8 rounded-2xl shadow-2xl text-center max-w-lg w-full mx-4">
                  {isFirstVisit ? (
                      <>
                        <h2 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-pink-500 mb-6">
                            Training Simulation
                        </h2>
                        
                        <div className="grid grid-cols-2 gap-4 mb-8">
                            <div className="bg-card-bg p-4 rounded-xl border border-border-col">
                                <div className="text-3xl mb-2">🖱️ / ⌨️</div>
                                <p className="text-sm font-bold text-text-main">Click or Space</p>
                                <p className="text-xs text-text-muted">Apply gradient (Jump)</p>
                            </div>
                            <div className="bg-card-bg p-4 rounded-xl border border-border-col">
                                <div className="text-3xl mb-2">🚧</div>
                                <p className="text-sm font-bold text-text-main">Avoid Barriers</p>
                                <p className="text-xs text-text-muted">Don't overfit!</p>
                            </div>
                        </div>

                        <div className="text-left text-sm text-text-muted mb-8 space-y-2 bg-card-bg/50 p-4 rounded-lg">
                            <p>• Your agent is prone to <span className="text-white font-bold">Underfitting</span> (Gravity).</p>
                            <p>• Keep the loss low by navigating through the gap.</p>
                            <p>• Survive as many <span className="text-white font-bold">Epochs</span> as possible.</p>
                        </div>
                      </>
                  ) : (
                      <>
                        <div className="w-20 h-20 bg-primary/20 rounded-full flex items-center justify-center mx-auto mb-6">
                            <span className="text-4xl">🚀</span>
                        </div>
                        <h2 className="text-3xl font-bold text-white mb-2">Neural Navigate</h2>
                        <p className="text-text-muted mb-8">Ready for another training run?</p>
                      </>
                  )}
                  
                  <button 
                      onClick={(e) => { e.stopPropagation(); startGame(); }}
                      className="w-full py-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold rounded-xl shadow-lg transition-transform transform hover:scale-105 active:scale-95 text-lg"
                  >
                      {isFirstVisit ? 'Initialize Agent' : 'Start Training'}
                  </button>
              </div>
          </div>
      )}

      {/* Game Over Screen */}
      {gameState === 'GAMEOVER' && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm z-20 animate-fade-in">
              <div className="bg-panel-bg border border-border-col p-8 rounded-2xl shadow-2xl text-center max-w-md w-full mx-4">
                  <div className="text-6xl mb-4">💥</div>
                  <h2 className="text-3xl font-bold text-white mb-2">Validation Failed!</h2>
                  <p className="text-text-muted mb-6">
                     You survived <span className="text-primary font-bold text-xl">{score}</span> epochs.
                  </p>
                  
                  <button 
                      onClick={(e) => { e.stopPropagation(); startGame(); }}
                      className="w-full py-4 bg-primary hover:bg-blue-600 text-white font-bold rounded-xl shadow-lg transition-transform transform hover:scale-105 active:scale-95"
                  >
                      Retry Optimization
                  </button>
              </div>
          </div>
      )}
    </div>
  );
};

export default Playground;