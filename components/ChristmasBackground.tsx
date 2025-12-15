import React, { useRef, useEffect } from 'react';

const ChristmasBackground: React.FC = () => {
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

    // Snowflakes
    const snowflakes = Array.from({ length: 150 }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      r: Math.random() * 3 + 1,
      speedY: Math.random() * 1 + 0.5,
      speedX: Math.random() * 0.5 - 0.25,
      opacity: Math.random() * 0.5 + 0.3
    }));

    let time = 0;

    const drawTree = (ctx: CanvasRenderingContext2D, x: number, y: number, scale: number) => {
        ctx.save();
        ctx.translate(x, y);
        ctx.scale(scale, scale);

        // Shadow
        ctx.fillStyle = 'rgba(0,0,0,0.3)';
        ctx.beginPath();
        ctx.ellipse(0, 10, 50, 15, 0, 0, Math.PI * 2);
        ctx.fill();

        // Trunk
        ctx.fillStyle = '#3e2723';
        ctx.fillRect(-10, 0, 20, 30);

        // Leaves (Layers) - Dark Green to Light Green
        const drawLayer = (yOffset: number, width: number, height: number, color: string) => {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.moveTo(-width, yOffset);
            ctx.lineTo(width, yOffset);
            ctx.lineTo(0, yOffset - height);
            ctx.fill();
        };

        drawLayer(0, 50, 60, '#1b5e20');
        drawLayer(-30, 40, 50, '#2e7d32');
        drawLayer(-60, 30, 40, '#388e3c');

        // Ornaments
        const ornaments = [
            {x: -20, y: -20, c: '#ef4444'}, {x: 15, y: -15, c: '#fbbf24'}, {x: 5, y: -45, c: '#3b82f6'},
            {x: -15, y: -50, c: '#ec4899'}, {x: 20, y: -70, c: '#ef4444'}, {x: -10, y: -80, c: '#fbbf24'}
        ];
        
        ornaments.forEach(o => {
            ctx.fillStyle = o.c;
            ctx.beginPath();
            ctx.arc(o.x, o.y, 4, 0, Math.PI * 2);
            ctx.fill();
            // Reflection
            ctx.fillStyle = 'rgba(255,255,255,0.6)';
            ctx.beginPath();
            ctx.arc(o.x - 1, o.y - 1, 1.5, 0, Math.PI * 2);
            ctx.fill();
        });

        // Tinsel (Garland)
        ctx.beginPath();
        ctx.moveTo(-30, -10);
        ctx.quadraticCurveTo(0, 0, 30, -15);
        ctx.moveTo(-20, -40);
        ctx.quadraticCurveTo(0, -30, 20, -45);
        ctx.strokeStyle = 'rgba(251, 191, 36, 0.4)'; // Gold
        ctx.lineWidth = 2;
        ctx.stroke();

        // Star
        ctx.fillStyle = '#fbbf24'; // Amber-400
        ctx.shadowColor = '#fbbf24';
        ctx.shadowBlur = 15;
        ctx.beginPath();
        ctx.translate(0, -100);
        // Draw Star Shape
        for(let i=0; i<5; i++){
             ctx.lineTo(Math.cos((18 + i*72)/180*Math.PI)*10, -Math.sin((18 + i*72)/180*Math.PI)*10);
             ctx.lineTo(Math.cos((54 + i*72)/180*Math.PI)*4, -Math.sin((54 + i*72)/180*Math.PI)*4);
        }
        ctx.closePath();
        ctx.fill();
        ctx.shadowBlur = 0;

        ctx.restore();
    };

    const drawDeer = (ctx: CanvasRenderingContext2D, x: number, y: number, scale: number) => {
        ctx.save();
        ctx.translate(x, y);
        ctx.scale(scale, scale);
        
        // Shadow
        ctx.fillStyle = 'rgba(0,0,0,0.3)';
        ctx.beginPath();
        ctx.ellipse(0, 30, 25, 8, 0, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#8d6e63'; // Brown

        // Body
        ctx.beginPath();
        ctx.ellipse(0, 15, 20, 12, 0, 0, Math.PI*2);
        ctx.fill();

        // Legs
        ctx.fillRect(-12, 20, 4, 20);
        ctx.fillRect(-4, 20, 4, 20);
        ctx.fillRect(4, 20, 4, 20);
        ctx.fillRect(12, 20, 4, 20);

        // Neck
        ctx.beginPath();
        ctx.moveTo(10, 10);
        ctx.lineTo(25, -15);
        ctx.lineTo(15, -15);
        ctx.lineTo(0, 10);
        ctx.fill();

        // Head
        ctx.beginPath();
        ctx.ellipse(22, -20, 9, 7, -0.2, 0, Math.PI*2);
        ctx.fill();

        // Ears
        ctx.beginPath();
        ctx.ellipse(14, -22, 3, 6, 0.5, 0, Math.PI*2);
        ctx.fill();

        // Antlers
        ctx.strokeStyle = '#5d4037'; // Dark brown
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(22, -25);
        ctx.lineTo(25, -35);
        ctx.lineTo(30, -40);
        ctx.moveTo(25, -35);
        ctx.lineTo(20, -40);
        ctx.stroke();

        // Nose (Red Glow)
        const glow = Math.sin(time * 0.1) * 5 + 10;
        ctx.shadowColor = '#ef4444';
        ctx.shadowBlur = glow;
        ctx.fillStyle = '#ef4444'; 
        ctx.beginPath();
        ctx.arc(31, -20, 2.5, 0, Math.PI*2);
        ctx.fill();
        ctx.shadowBlur = 0;

        // Eye
        ctx.fillStyle = '#1a1a1a';
        ctx.beginPath();
        ctx.arc(24, -22, 1, 0, Math.PI*2);
        ctx.fill();

        ctx.restore();
    };

    const animate = () => {
        time++;
        // Winter Night Gradient
        const grad = ctx.createLinearGradient(0, 0, 0, height);
        grad.addColorStop(0, '#022c22'); // Dark green/black top
        grad.addColorStop(1, '#064e3b'); // Emerald bottom
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, width, height);

        // Draw Snow (Background)
        ctx.fillStyle = "#ffffff";
        snowflakes.forEach(f => {
            f.y += f.speedY;
            f.x += f.speedX;

            if (f.y > height) {
                f.y = -10;
                f.x = Math.random() * width;
            }
            if (f.x > width) f.x = 0;
            if (f.x < 0) f.x = width;

            ctx.globalAlpha = f.opacity;
            ctx.beginPath();
            ctx.arc(f.x, f.y, f.r, 0, Math.PI * 2);
            ctx.fill();
        });
        ctx.globalAlpha = 1;

        // --- DRAW SCENE (Bottom Right) ---
        const treeX = width - 150;
        const treeY = height - 50;
        const deerX = width - 260;
        const deerY = height - 60;

        // Hill
        ctx.fillStyle = '#065f46'; // Lighter emerald for ground
        ctx.beginPath();
        ctx.moveTo(width - 400, height);
        ctx.quadraticCurveTo(width - 200, height - 80, width, height - 40);
        ctx.lineTo(width, height);
        ctx.lineTo(width - 400, height);
        ctx.fill();

        drawTree(ctx, treeX, treeY, 2.0);
        
        // Deer looks at tree, maybe bobbing head slightly
        const deerScale = 1.5;
        const headBob = Math.sin(time * 0.05) * 2;
        ctx.save();
        ctx.translate(0, headBob * 0.2); // Tiny body movement
        drawDeer(ctx, deerX, deerY, deerScale);
        ctx.restore();

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

export default ChristmasBackground;