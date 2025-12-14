import React, { useRef, useEffect } from 'react';

const GothicBackground: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = window.innerWidth;
    let height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;

    // Feathers
    const feathers = Array.from({ length: 40 }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      size: Math.random() * 15 + 10,
      speedY: Math.random() * 1 + 0.5,
      swayRange: Math.random() * 20 + 10,
      swaySpeed: Math.random() * 0.02 + 0.01,
      swayOffset: Math.random() * Math.PI * 2,
      rotation: Math.random() * Math.PI * 2,
      rotationSpeed: Math.random() * 0.02 - 0.01,
      color: Math.random() > 0.5 ? '#171717' : '#262626' // Very dark grey/black feathers
    }));

    // Fog particles
    const fog = Array.from({ length: 20 }, () => ({
      x: Math.random() * width,
      y: height - Math.random() * 300,
      width: Math.random() * 400 + 200,
      height: Math.random() * 100 + 50,
      speed: Math.random() * 0.2 + 0.1,
      opacity: Math.random() * 0.2
    }));

    let time = 0;

    const drawCastle = (ctx: CanvasRenderingContext2D, w: number, h: number) => {
        ctx.save();
        ctx.fillStyle = '#0a0a0a'; // Silhouette color
        
        // Base hill
        ctx.beginPath();
        ctx.moveTo(0, h);
        ctx.bezierCurveTo(w * 0.3, h - 100, w * 0.7, h - 50, w, h);
        ctx.fill();

        // Main Keep
        const cx = w * 0.2; // Left side positioning
        const cy = h - 80;
        
        // Towers
        const drawTower = (tx: number, ty: number, tw: number, th: number) => {
            ctx.fillRect(tx, ty - th, tw, th);
            // Spire
            ctx.beginPath();
            ctx.moveTo(tx - 5, ty - th);
            ctx.lineTo(tx + tw/2, ty - th - 60);
            ctx.lineTo(tx + tw + 5, ty - th);
            ctx.fill();
            // Windows
            ctx.fillStyle = '#f59e0b'; // Lit window
            if (Math.random() > 0.95) ctx.globalAlpha = 0.5 + Math.sin(time * 0.1) * 0.5; // Flicker
            ctx.fillRect(tx + tw/2 - 3, ty - th + 20, 6, 12);
            ctx.globalAlpha = 1;
            ctx.fillStyle = '#0a0a0a';
        };

        drawTower(cx, cy, 60, 200);
        drawTower(cx - 50, cy + 20, 40, 150);
        drawTower(cx + 50, cy + 10, 50, 180);

        // Bridge/Wall
        ctx.fillRect(cx + 80, cy - 60, 100, 40);
        
        // Right side ruins
        const rx = w * 0.8;
        const ry = h - 60;
        drawTower(rx, ry, 40, 100);
        
        ctx.restore();
    };

    const drawFeather = (ctx: CanvasRenderingContext2D, f: any, t: number) => {
        ctx.save();
        ctx.translate(f.x + Math.sin(t * f.swaySpeed + f.swayOffset) * f.swayRange, f.y);
        ctx.rotate(Math.sin(t * f.swaySpeed) * 0.5 + f.rotation);
        
        ctx.fillStyle = f.color;
        ctx.beginPath();
        // Quill spine
        ctx.moveTo(0, -f.size);
        ctx.quadraticCurveTo(2, 0, 0, f.size);
        ctx.quadraticCurveTo(-2, 0, 0, -f.size);
        ctx.fill();
        
        // Barbs
        ctx.beginPath();
        ctx.moveTo(0, -f.size * 0.8);
        ctx.quadraticCurveTo(f.size * 0.6, -f.size * 0.5, 0, f.size * 0.8);
        ctx.quadraticCurveTo(-f.size * 0.6, -f.size * 0.5, 0, -f.size * 0.8);
        ctx.fill();
        
        ctx.restore();
    };

    const animate = () => {
        time++;
        // Gothic Sky Gradient
        const bgGrad = ctx.createLinearGradient(0, 0, 0, height);
        bgGrad.addColorStop(0, '#0f0505'); // Very dark red/black
        bgGrad.addColorStop(0.5, '#1a0505'); 
        bgGrad.addColorStop(1, '#2a0a0a'); // Dark crimson bottom
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, width, height);

        // Moon
        ctx.save();
        ctx.fillStyle = '#e2e8f0';
        ctx.shadowColor = '#fef3c7';
        ctx.shadowBlur = 40;
        ctx.beginPath();
        ctx.arc(width * 0.8, 100, 40, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        // Draw Fog (Behind Castle)
        fog.forEach(cloud => {
            cloud.x += cloud.speed;
            if (cloud.x > width + cloud.width) cloud.x = -cloud.width;
            
            const grad = ctx.createRadialGradient(cloud.x, cloud.y, 0, cloud.x, cloud.y, cloud.width/2);
            grad.addColorStop(0, `rgba(60, 20, 20, ${cloud.opacity})`);
            grad.addColorStop(1, 'rgba(0,0,0,0)');
            
            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.ellipse(cloud.x, cloud.y, cloud.width/2, cloud.height/2, 0, 0, Math.PI * 2);
            ctx.fill();
        });

        // Draw Castle Silhouette
        drawCastle(ctx, width, height);

        // Draw Feathers (Foreground)
        feathers.forEach(f => {
            f.y += f.speedY;
            f.rotation += f.rotationSpeed;

            if (f.y > height + 20) {
                f.y = -20;
                f.x = Math.random() * width;
            }

            drawFeather(ctx, f, time);
        });

        requestAnimationFrame(animate);
    };

    const animId = requestAnimationFrame(animate);
    const handleResize = () => {
        width = window.innerWidth;
        height = window.innerHeight;
        canvas.width = width;
        canvas.height = height;
    };
    window.addEventListener('resize', handleResize);

    return () => {
        cancelAnimationFrame(animId);
        window.removeEventListener('resize', handleResize);
    };
  }, []);

  return <canvas ref={canvasRef} className="fixed inset-0 pointer-events-none z-0" />;
};

export default GothicBackground;