import React, { useEffect, useRef } from 'react';
import { ThemeId } from '../types';

interface Props {
  theme: ThemeId | null;
}

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  life: number;
  maxLife: number;
  color: string;
  rotation: number;
  rotationSpeed: number;
  // Specific for electricity
  segments?: {x: number, y: number}[];
}

const ClickEffects: React.FC<Props> = ({ theme }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const requestRef = useRef<number>(0);

  // Helper to draw a star
  const drawStar = (ctx: CanvasRenderingContext2D, x: number, y: number, r: number, color: string, rot: number) => {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(rot);
    ctx.beginPath();
    ctx.fillStyle = color;
    for (let i = 0; i < 5; i++) {
      ctx.lineTo(Math.cos((18 + i * 72) * 0.01745) * r, -Math.sin((18 + i * 72) * 0.01745) * r);
      ctx.lineTo(Math.cos((54 + i * 72) * 0.01745) * (r * 0.5), -Math.sin((54 + i * 72) * 0.01745) * (r * 0.5));
    }
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  };

  // Helper to draw a fish tail (simple fin shape)
  const drawFishTail = (ctx: CanvasRenderingContext2D, x: number, y: number, r: number, color: string, rot: number) => {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(rot);
    ctx.fillStyle = color;
    ctx.beginPath();
    // A simple crescent/tail shape
    ctx.moveTo(0, 0);
    ctx.quadraticCurveTo(-r, -r, -r * 1.5, -r * 0.5);
    ctx.quadraticCurveTo(-r * 0.5, 0, -r * 1.5, r * 0.5);
    ctx.quadraticCurveTo(-r, r, 0, 0);
    ctx.fill();
    ctx.restore();
  };

  // Helper to draw a flower (Forget-me-not style)
  const drawFlower = (ctx: CanvasRenderingContext2D, x: number, y: number, r: number, color: string, rot: number) => {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(rot);
    // Petals
    ctx.fillStyle = color;
    for (let i = 0; i < 5; i++) {
      const angle = (i * 2 * Math.PI) / 5;
      const px = Math.cos(angle) * (r * 0.5);
      const py = Math.sin(angle) * (r * 0.5);
      ctx.beginPath();
      ctx.arc(px, py, r * 0.4, 0, Math.PI * 2);
      ctx.fill();
    }
    // Center
    ctx.fillStyle = '#FFD700';
    ctx.beginPath();
    ctx.arc(0, 0, r * 0.25, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  };

  // Helper to draw a snowflake
  const drawSnowflake = (ctx: CanvasRenderingContext2D, x: number, y: number, r: number, color: string, rot: number) => {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(rot);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    // 3 intersecting lines
    for (let i = 0; i < 3; i++) {
        ctx.moveTo(0, -r);
        ctx.lineTo(0, r);
        ctx.rotate(Math.PI / 3);
    }
    ctx.stroke();
    ctx.restore();
  };

  // Helper to draw electricity spike
  const drawElectricity = (ctx: CanvasRenderingContext2D, p: Particle) => {
      ctx.save();
      ctx.translate(p.x, p.y);
      // Rotate towards velocity direction
      const angle = Math.atan2(p.vy, p.vx);
      ctx.rotate(angle);
      
      ctx.strokeStyle = p.color;
      ctx.lineWidth = 2;
      ctx.shadowBlur = 10;
      ctx.shadowColor = p.color;
      
      ctx.beginPath();
      ctx.moveTo(0, 0);
      
      // Draw a jagged line of length ~20-30
      let cx = 0;
      let cy = 0;
      for (let i = 0; i < 4; i++) {
          cx += 8;
          cy = (Math.random() - 0.5) * 10; // Jitter Y
          ctx.lineTo(cx, cy);
      }
      ctx.stroke();
      
      ctx.restore();
  };

  // Main animation loop
  const animate = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Update and draw particles
    for (let i = particlesRef.current.length - 1; i >= 0; i--) {
      const p = particlesRef.current[i];
      p.life--;
      p.x += p.vx;
      p.y += p.vy;
      
      if (theme !== 'gothic') {
        p.vy += 0.05; // Gravity for non-electricity
      }
      p.rotation += p.rotationSpeed;

      if (p.life <= 0) {
        particlesRef.current.splice(i, 1);
        continue;
      }

      const alpha = p.life / p.maxLife;
      ctx.globalAlpha = alpha;

      if (theme === 'cosmic') {
          drawStar(ctx, p.x, p.y, p.size, p.color, p.rotation);
      } else if (theme === 'sea') {
          const angle = Math.atan2(p.vy, p.vx);
          drawFishTail(ctx, p.x, p.y, p.size, p.color, angle);
      } else if (theme === 'flower') {
          drawFlower(ctx, p.x, p.y, p.size, p.color, p.rotation);
      } else if (theme === 'snow' || theme === 'christmas') {
          if (theme === 'christmas' && Math.random() > 0.5) {
               // Festive confetti circle for Christmas mix
              ctx.fillStyle = p.color;
              ctx.beginPath();
              ctx.arc(p.x, p.y, p.size / 2, 0, Math.PI * 2);
              ctx.fill();
          } else {
              drawSnowflake(ctx, p.x, p.y, p.size, p.color, p.rotation);
          }
      } else if (theme === 'gothic') {
          drawElectricity(ctx, p);
      } else {
          // Default circle
          ctx.fillStyle = p.color;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size / 2, 0, Math.PI * 2);
          ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    requestRef.current = requestAnimationFrame(animate);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }
    
    const handleResize = () => {
        if (canvasRef.current) {
            canvasRef.current.width = window.innerWidth;
            canvasRef.current.height = window.innerHeight;
        }
    };
    window.addEventListener('resize', handleResize);
    requestRef.current = requestAnimationFrame(animate);

    return () => {
        window.removeEventListener('resize', handleResize);
        cancelAnimationFrame(requestRef.current);
    };
  }, [theme]);

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      const count = theme === 'gothic' ? 4 : 6;
      for (let i = 0; i < count; i++) {
        const angle = Math.random() * Math.PI * 2;
        const speed = Math.random() * 3 + 1;
        
        let color = '#ffffff';
        let size = 10;
        let life = 40 + Math.random() * 20;
        
        if (theme === 'cosmic') {
            color = Math.random() > 0.5 ? '#FFD700' : '#FFFACD';
            size = Math.random() * 8 + 4;
        } else if (theme === 'sea') {
            color = Math.random() > 0.5 ? '#00FFFF' : '#7FFFD4';
            size = Math.random() * 10 + 5;
        } else if (theme === 'flower') {
            const colors = ['#87CEFA', '#FFB6C1', '#DDA0DD'];
            color = colors[Math.floor(Math.random() * colors.length)];
            size = Math.random() * 8 + 6;
        } else if (theme === 'snow') {
            color = '#F0F8FF';
            size = Math.random() * 6 + 4;
        } else if (theme === 'christmas') {
            const colors = ['#ef4444', '#22c55e', '#fbbf24', '#ffffff']; // Red, Green, Gold, White
            color = colors[Math.floor(Math.random() * colors.length)];
            size = Math.random() * 6 + 4;
        } else if (theme === 'gothic') {
            // Dark red / crimson / blood red for gothic electricity
            const colors = ['#dc2626', '#b91c1c', '#7f1d1d']; 
            color = colors[Math.floor(Math.random() * colors.length)];
            size = 2; // Line width essentially
            life = 10 + Math.random() * 10; // Short life for sparks
        } else {
            const defaults = ['#FFD700', '#FFA500', '#00FFFF', '#FF00FF', '#FFFFFF'];
            color = defaults[Math.floor(Math.random() * defaults.length)];
            size = Math.random() * 6 + 2;
        }

        particlesRef.current.push({
          x: e.clientX,
          y: e.clientY,
          vx: Math.cos(angle) * speed,
          vy: Math.sin(angle) * speed,
          size: size,
          life: life,
          maxLife: life,
          color: color,
          rotation: Math.random() * Math.PI * 2,
          rotationSpeed: (Math.random() - 0.5) * 0.2
        });
      }
    };

    window.addEventListener('mousedown', handleClick);
    return () => window.removeEventListener('mousedown', handleClick);
  }, [theme]);

  return (
    <canvas 
        ref={canvasRef} 
        className="fixed inset-0 pointer-events-none z-[9999]" 
    />
  );
};

export default ClickEffects;