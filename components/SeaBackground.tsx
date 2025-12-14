import React, { useRef, useEffect } from 'react';

const SeaBackground: React.FC = () => {
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

    // Bubbles
    const bubbles = Array.from({ length: 50 }, () => ({
      x: Math.random() * width,
      y: height + Math.random() * 200,
      radius: Math.random() * 8 + 2,
      speed: Math.random() * 1.5 + 0.5,
      wobbleSpeed: Math.random() * 0.05 + 0.02,
      wobbleOffset: Math.random() * Math.PI * 2,
      opacity: Math.random() * 0.4 + 0.1
    }));

    // Light Rays
    const rays = Array.from({ length: 5 }, () => ({
      x: Math.random() * width,
      width: Math.random() * 100 + 50,
      angle: Math.random() * 0.2 + 0.1, // Slight slant
      speed: Math.random() * 0.002
    }));

    let time = 0;

    const drawCoralReef = (ctx: CanvasRenderingContext2D, x: number, y: number, t: number) => {
        ctx.save();
        ctx.translate(x, y);
        const scale = 1.2;
        ctx.scale(scale, scale);

        // Sway calculation
        const sway = Math.sin(t * 0.02) * 5;

        // Base Rock formation
        ctx.fillStyle = '#1f2937'; // Gray-800
        ctx.beginPath();
        ctx.moveTo(-200, 0); 
        ctx.quadraticCurveTo(-100, -80, 0, -60);
        ctx.quadraticCurveTo(100, -40, 200, 0);
        ctx.fill();

        // Branching Coral (Staghorn) - Back Layer
        ctx.strokeStyle = '#f87171'; // Red-400
        ctx.lineWidth = 6;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(-50, -60);
        ctx.quadraticCurveTo(-60, -120, -80 + sway, -160);
        ctx.moveTo(-60, -100);
        ctx.quadraticCurveTo(-30, -130, -20 + sway, -150);
        ctx.stroke();

        // Fan Coral (Purple)
        ctx.fillStyle = 'rgba(168, 85, 247, 0.8)'; // Purple
        ctx.beginPath();
        ctx.moveTo(100, -30);
        ctx.bezierCurveTo(180, -120 + sway, 20, -120 + sway, 100, -30);
        ctx.fill();
        // Veins
        ctx.strokeStyle = 'rgba(147, 51, 234, 0.5)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for(let i=0; i<5; i++) {
             ctx.moveTo(100, -30);
             ctx.quadraticCurveTo(60 + i*20, -80, 40 + i*30 + sway, -110);
        }
        ctx.stroke();

        // Brain Coral (Orange)
        ctx.fillStyle = '#d97706';
        ctx.beginPath();
        ctx.arc(0, -30, 35, Math.PI, 0); // Half circle
        ctx.fill();
        // Brain texture details
        ctx.strokeStyle = '#92400e';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(-25, -30); ctx.quadraticCurveTo(-25, -55, -10, -40);
        ctx.moveTo(0, -30); ctx.quadraticCurveTo(0, -60, 15, -40);
        ctx.stroke();

        // Seaweed / Kelp (Green) - Front Layer
        ctx.strokeStyle = '#4ade80';
        ctx.lineWidth = 8;
        ctx.lineCap = 'round';
        // Strand 1
        ctx.beginPath();
        ctx.moveTo(-120, -10);
        ctx.bezierCurveTo(-120 + sway*2, -80, -150 - sway*2, -120, -130 + sway*3, -180);
        ctx.stroke();
        // Strand 2
        ctx.beginPath();
        ctx.moveTo(150, -5);
        ctx.bezierCurveTo(150 + sway*2, -60, 180 - sway*2, -100, 160 + sway*3, -150);
        ctx.stroke();

        // Small Fish School
        const fishOffset = Math.sin(t * 0.015) * 40;
        ctx.fillStyle = '#facc15'; // Yellow
        [
            {x: -50, y: -200}, {x: -20, y: -220}, {x: -80, y: -210}
        ].forEach(pos => {
             ctx.beginPath();
             // Simple fish shape
             ctx.ellipse(pos.x + fishOffset, pos.y, 8, 4, 0, 0, Math.PI*2);
             ctx.fill();
             // Tail
             ctx.beginPath();
             ctx.moveTo(pos.x + fishOffset - 8, pos.y);
             ctx.lineTo(pos.x + fishOffset - 12, pos.y - 4);
             ctx.lineTo(pos.x + fishOffset - 12, pos.y + 4);
             ctx.fill();
        });

        ctx.restore();
    };

    const animate = () => {
        time++;
        // Gradient Background (Deep Sea)
        const bgGrad = ctx.createLinearGradient(0, 0, 0, height);
        bgGrad.addColorStop(0, '#0f172a'); // Slightly lighter top
        bgGrad.addColorStop(1, '#020617'); // Dark bottom
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, width, height);

        // Draw Light Rays
        ctx.save();
        ctx.globalCompositeOperation = 'overlay';
        rays.forEach(ray => {
            const rayGrad = ctx.createLinearGradient(ray.x, 0, ray.x - Math.tan(ray.angle) * height, height);
            rayGrad.addColorStop(0, 'rgba(255, 255, 255, 0.08)');
            rayGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
            
            ctx.fillStyle = rayGrad;
            ctx.beginPath();
            ctx.moveTo(ray.x, 0);
            ctx.lineTo(ray.x + ray.width, 0);
            ctx.lineTo(ray.x + ray.width - Math.tan(ray.angle) * height, height);
            ctx.lineTo(ray.x - Math.tan(ray.angle) * height, height);
            ctx.closePath();
            ctx.fill();
            
            ray.x += Math.sin(time * ray.speed) * 0.5;
        });
        ctx.restore();

        // Draw Bubbles (Background)
        ctx.fillStyle = '#ffffff';
        bubbles.forEach(b => {
            b.y -= b.speed;
            const wobble = Math.sin(time * b.wobbleSpeed + b.wobbleOffset) * 20;
            const x = b.x + wobble;

            if (b.y < -50) {
                b.y = height + 50;
                b.x = Math.random() * width;
            }

            if (b.radius < 5) {
                ctx.globalAlpha = b.opacity;
                ctx.beginPath();
                ctx.arc(x, b.y, b.radius, 0, Math.PI * 2);
                ctx.fill();
            }
        });

        // --- DRAW CORAL REEF ---
        const reefX = width * 0.85;
        const reefY = height;
        
        // Add a subtle glow behind it
        ctx.save();
        ctx.translate(reefX, reefY - 50);
        const glow = ctx.createRadialGradient(0, 0, 50, 0, 0, 200);
        glow.addColorStop(0, 'rgba(45, 212, 191, 0.15)'); // Teal glow
        glow.addColorStop(1, 'transparent');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(0, 0, 200, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        // Draw the reef
        drawCoralReef(ctx, reefX, reefY, time);

        // Draw Bubbles (Foreground)
        ctx.fillStyle = '#ffffff';
        bubbles.forEach(b => {
             if (b.radius >= 5) {
                const wobble = Math.sin(time * b.wobbleSpeed + b.wobbleOffset) * 20;
                const x = b.x + wobble;
                
                ctx.globalAlpha = b.opacity;
                ctx.beginPath();
                ctx.arc(x, b.y, b.radius, 0, Math.PI * 2);
                ctx.fill();
             }
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
        window.removeEventListener('resize', handleResize);
    };
  }, []);

  return <canvas ref={canvasRef} className="fixed inset-0 pointer-events-none z-0" />;
};

export default SeaBackground;