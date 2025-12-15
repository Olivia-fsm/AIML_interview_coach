import React, { useRef, useEffect, useState } from 'react';

const Playground: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const [gameState, setGameState] = useState<'START' | 'PLAYING' | 'GAMEOVER'>('START');
  const [score, setScore] = useState(0);
  const [highScore, setHighScore] = useState(0);

  // Game Constants
  const GRAVITY = 0.4;
  const JUMP_STRENGTH = -7;
  const SPEED = 3.5;
  const OBSTACLE_WIDTH = 60;
  const OBSTACLE_GAP = 200;
  const OBSTACLE_SPACING = 350;

  // Refs for game loop to avoid stale state in closures
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
    const saved = localStorage.getItem('ldc_high_score');
    if (saved) setHighScore(parseInt(saved));
  }, []);

  const startGame = () => {
    if (!canvasRef.current) return;
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
        addObstacle(canvasRef.current.width + i * OBSTACLE_SPACING, height);
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
      for (let i = 0; i < 20; i++) {
          const angle = Math.random() * Math.PI * 2;
          const speed = Math.random() * 5 + 2;
          stateRef.current.particles.push({
              x, y,
              vx: Math.cos(angle) * speed,
              vy: Math.sin(angle) * speed,
              life: 1.0,
              color: Math.random() > 0.5 ? '#f472b6' : '#818cf8' // Pink/Indigo
          });
      }
  };

  const jump = () => {
      if (stateRef.current.isActive) {
          stateRef.current.agentVelocity = JUMP_STRENGTH;
          
          // Add trail particles
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
  }, [gameState]); // Re-bind if game state changes significantly, though largely handled by refs

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
          
          // Collision Logic
          // Player is at x=100, radius=15
          // Obstacle is rect at obs.x, width=OBSTACLE_WIDTH
          
          if (obs.x < 115 && obs.x + OBSTACLE_WIDTH > 85) {
              // Inside horizontal area
              if (state.agentY - 10 < obs.topHeight || state.agentY + 10 > obs.topHeight + OBSTACLE_GAP) {
                  gameOver();
              }
          }

          // Score
          if (obs.x + OBSTACLE_WIDTH < 85 && !obs.passed) {
              obs.passed = true;
              state.score++;
              setScore(state.score);
          }
      });

      // Cleanup obstacles
      if (state.obstacles.length > 0 && state.obstacles[0].x < -OBSTACLE_WIDTH) {
          state.obstacles.shift();
      }

      // Particles
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
      
      if (stateRef.current.score > highScore) {
          setHighScore(stateRef.current.score);
          localStorage.setItem('ldc_high_score', stateRef.current.score.toString());
      }
  };

  const draw = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
      // Clear
      ctx.clearRect(0, 0, width, height);

      // Background Grid (Moving)
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
          // Top Bar
          ctx.fillRect(obs.x, 0, OBSTACLE_WIDTH, obs.topHeight);
          ctx.strokeRect(obs.x, 0, OBSTACLE_WIDTH, obs.topHeight);
          
          // Bottom Bar
          const botY = obs.topHeight + OBSTACLE_GAP;
          ctx.fillRect(obs.x, botY, OBSTACLE_WIDTH, height - botY);
          ctx.strokeRect(obs.x, botY, OBSTACLE_WIDTH, height - botY);
          
          // "Circuit" details on obstacles
          ctx.beginPath();
          ctx.moveTo(obs.x + 10, 0); ctx.lineTo(obs.x + 10, obs.topHeight - 20);
          ctx.moveTo(obs.x + 30, 0); ctx.lineTo(obs.x + 30, obs.topHeight - 40);
          ctx.stroke();
      });

      // Player Trail
      const trailLen = 10;
      if (stateRef.current.isActive) {
        ctx.beginPath();
        ctx.moveTo(100, stateRef.current.agentY);
        ctx.lineTo(60, stateRef.current.agentY - stateRef.current.agentVelocity * 2);
        ctx.strokeStyle = 'rgba(59, 130, 246, 0.5)';
        ctx.lineWidth = 4;
        ctx.stroke();
      }

      // Player
      if (gameState !== 'GAMEOVER' || stateRef.current.particles.length > 0) { // Hide player on death, show particles
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
      <div className="absolute top-6 left-6 flex gap-6 z-10">
          <div className="bg-panel-bg/80 backdrop-blur border border-border-col px-4 py-2 rounded-lg shadow-lg">
              <span className="text-xs text-text-muted uppercase font-bold block">Current Epoch</span>
              <span className="text-2xl font-bold text-white font-mono">{score}</span>
          </div>
          <div className="bg-panel-bg/80 backdrop-blur border border-border-col px-4 py-2 rounded-lg shadow-lg">
              <span className="text-xs text-text-muted uppercase font-bold block">Best Record</span>
              <span className="text-2xl font-bold text-primary font-mono">{highScore}</span>
          </div>
      </div>

      {/* Start / Game Over Screens */}
      {gameState !== 'PLAYING' && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm z-20 animate-fade-in">
              <div className="bg-panel-bg border border-border-col p-8 rounded-2xl shadow-2xl text-center max-w-md w-full mx-4">
                  <div className="w-20 h-20 bg-primary/20 rounded-full flex items-center justify-center mx-auto mb-6">
                      <span className="text-4xl">🚀</span>
                  </div>
                  
                  <h2 className="text-3xl font-bold text-white mb-2">
                      {gameState === 'START' ? 'Neural Navigate' : 'Validation Failed!'}
                  </h2>
                  
                  <p className="text-text-muted mb-8">
                      {gameState === 'START' 
                        ? 'Guide the learning agent through the hidden layers. Avoid overfitting barriers!' 
                        : `You survived ${score} epochs before the gradient vanished.`}
                  </p>
                  
                  <button 
                      onClick={startGame}
                      className="w-full py-4 bg-primary hover:bg-blue-600 text-white font-bold rounded-xl shadow-lg transition-transform transform hover:scale-105 active:scale-95"
                  >
                      {gameState === 'START' ? 'Start Training' : 'Retry Optimization'}
                  </button>
                  
                  <p className="mt-4 text-xs text-text-muted">
                      Press <span className="font-bold border border-gray-600 rounded px-1">Space</span> or <span className="font-bold border border-gray-600 rounded px-1">Click</span> to jump
                  </p>
              </div>
          </div>
      )}
    </div>
  );
};

export default Playground;