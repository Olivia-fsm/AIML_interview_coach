import React, { useRef, useEffect } from 'react';

const SnowBackground: React.FC = () => {
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

    const snowflakes = Array.from({ length: 200 }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      radius: Math.random() * 3 + 1,
      speedY: Math.random() * 2 + 1,
      speedX: Math.random() * 1 - 0.5,
      opacity: Math.random() * 0.5 + 0.3
    }));

    let mouseX = width / 2;
    let wind = 0;

    const handleMouseMove = (e: MouseEvent) => {
        mouseX = e.clientX;
        // Simple wind calculation based on mouse position relative to center
        wind = (mouseX - width / 2) * 0.005;
    };
    window.addEventListener('mousemove', handleMouseMove);

    const drawSnowman = (ctx: CanvasRenderingContext2D, w: number, h: number) => {
        const x = w * 0.85; // Position on right side
        const y = h + 20; // Anchor at bottom
        
        ctx.save();
        
        // Shadow
        ctx.fillStyle = 'rgba(0,0,0,0.1)';
        ctx.beginPath();
        ctx.ellipse(x, y - 40, 60, 10, 0, 0, Math.PI * 2);
        ctx.fill();

        // Bottom Body
        const r1 = 50;
        const y1 = y - r1 - 20;
        const grad1 = ctx.createRadialGradient(x - 10, y1 - 10, 0, x, y1, r1);
        grad1.addColorStop(0, '#ffffff');
        grad1.addColorStop(1, '#e2e8f0');
        ctx.fillStyle = grad1;
        ctx.beginPath();
        ctx.arc(x, y1, r1, 0, Math.PI * 2);
        ctx.fill();

        // Middle Body
        const r2 = 35;
        const y2 = y1 - r1 - r2 * 0.6;
        const grad2 = ctx.createRadialGradient(x - 5, y2 - 10, 0, x, y2, r2);
        grad2.addColorStop(0, '#ffffff');
        grad2.addColorStop(1, '#e2e8f0');
        ctx.fillStyle = grad2;
        ctx.beginPath();
        ctx.arc(x, y2, r2, 0, Math.PI * 2);
        ctx.fill();

        // Buttons
        ctx.fillStyle = '#1e293b';
        ctx.beginPath(); ctx.arc(x, y2 - 10, 3, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(x, y2 + 10, 3, 0, Math.PI * 2); ctx.fill();

        // Head
        const r3 = 25;
        const y3 = y2 - r2 - r3 * 0.7;
        const grad3 = ctx.createRadialGradient(x - 5, y3 - 5, 0, x, y3, r3);
        grad3.addColorStop(0, '#ffffff');
        grad3.addColorStop(1, '#e2e8f0');
        ctx.fillStyle = grad3;
        ctx.beginPath();
        ctx.arc(x, y3, r3, 0, Math.PI * 2);
        ctx.fill();

        // Eyes
        ctx.fillStyle = '#1e293b';
        ctx.beginPath(); ctx.arc(x - 8, y3 - 5, 3, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(x + 8, y3 - 5, 3, 0, Math.PI * 2); ctx.fill();

        // Nose (Carrot)
        ctx.fillStyle = '#fb923c';
        ctx.beginPath();
        ctx.moveTo(x, y3 + 2);
        ctx.lineTo(x + 15, y3 + 6);
        ctx.lineTo(x, y3 + 8);
        ctx.fill();

        // Arms (Twigs)
        ctx.strokeStyle = '#78350f'; // Brown
        ctx.lineWidth = 3;
        // Left arm
        ctx.beginPath();
        ctx.moveTo(x - r2 + 5, y2 - 5);
        ctx.lineTo(x - r2 - 25, y2 - 20);
        ctx.stroke();
        // Right arm
        ctx.beginPath();
        ctx.moveTo(x + r2 - 5, y2 - 5);
        ctx.lineTo(x + r2 + 25, y2 - 25);
        ctx.stroke();
        
        // Scarf (Red)
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 8;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(x - 15, y3 + r3 * 0.6);
        ctx.quadraticCurveTo(x, y3 + r3 + 5, x + 20, y2 - r2 + 5);
        ctx.stroke();
        // Scarf tail
        ctx.beginPath();
        ctx.moveTo(x + 10, y2 - r2 + 5);
        ctx.lineTo(x + 15, y2 + 10);
        ctx.stroke();

        ctx.restore();
    };

    const animate = () => {
        // Cold Winter Background
        const bgGrad = ctx.createLinearGradient(0, 0, 0, height);
        bgGrad.addColorStop(0, '#1e293b'); // Slate 800
        bgGrad.addColorStop(1, '#334155'); // Slate 700
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, width, height);

        // Draw Snowman (Behind the falling snow)
        drawSnowman(ctx, width, height);

        ctx.fillStyle = '#ffffff';

        snowflakes.forEach(flake => {
            flake.y += flake.speedY;
            flake.x += flake.speedX + wind;

            // Reset if out of bounds
            if (flake.y > height) {
                flake.y = -10;
                flake.x = Math.random() * width;
            }
            if (flake.x > width) flake.x = 0;
            if (flake.x < 0) flake.x = width;

            ctx.globalAlpha = flake.opacity;
            ctx.beginPath();
            ctx.arc(flake.x, flake.y, flake.radius, 0, Math.PI * 2);
            ctx.fill();
        });
        ctx.globalAlpha = 1;

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
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('resize', handleResize);
    };
  }, []);

  return <canvas ref={canvasRef} className="fixed inset-0 pointer-events-none z-0" />;
};

export default SnowBackground;