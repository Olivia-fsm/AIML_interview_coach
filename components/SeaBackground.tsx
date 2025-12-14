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

    const drawRafayel = (ctx: CanvasRenderingContext2D, x: number, y: number, scale: number, t: number) => {
        ctx.save();
        ctx.translate(x, y);
        ctx.scale(scale, scale);

        // --- TAIL ---
        // Swaying motion
        const tailSway = Math.sin(t * 0.05) * 10;
        const finSway = Math.sin(t * 0.05 - 0.5) * 15;

        // Tail Body
        const tailGrad = ctx.createLinearGradient(0, 0, 0, 150);
        tailGrad.addColorStop(0, '#e0f2fe'); // Light skin transition
        tailGrad.addColorStop(0.3, '#8b5cf6'); // Purple start
        tailGrad.addColorStop(0.7, '#3b82f6'); // Blue mid
        tailGrad.addColorStop(1, '#06b6d4'); // Cyan end
        
        ctx.fillStyle = tailGrad;
        ctx.beginPath();
        ctx.moveTo(-25, 0); // Waist Left
        ctx.quadraticCurveTo(-35 + tailSway/2, 60, -10 + tailSway, 120); // Left curve
        ctx.lineTo(10 + tailSway, 120); // Tail end width
        ctx.quadraticCurveTo(35 + tailSway/2, 60, 25, 0); // Right curve
        ctx.closePath();
        ctx.fill();

        // Tail Fins (Glowing)
        ctx.save();
        ctx.translate(tailSway, 120);
        ctx.rotate(finSway * 0.02);
        
        ctx.shadowBlur = 15;
        ctx.shadowColor = '#06b6d4';
        ctx.fillStyle = 'rgba(6, 182, 212, 0.8)';
        
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.quadraticCurveTo(-40, 30, -50, 80); // Left fin
        ctx.quadraticCurveTo(-10, 50, 0, 40); // Center notch
        ctx.quadraticCurveTo(10, 50, 50, 80); // Right fin tip
        ctx.quadraticCurveTo(40, 30, 0, 0); // Right fin top
        ctx.fill();
        ctx.restore();

        // --- TORSO ---
        ctx.fillStyle = '#ffedd5'; // Skin tone
        ctx.beginPath();
        ctx.moveTo(-25, 0); // Waist
        ctx.lineTo(25, 0);
        ctx.lineTo(30, -60); // Shoulder Right
        ctx.lineTo(-30, -60); // Shoulder Left
        ctx.closePath();
        ctx.fill();

        // Arms (Simple down by side)
        ctx.strokeStyle = '#ffedd5';
        ctx.lineWidth = 14;
        ctx.lineCap = 'round';
        // Left Arm
        ctx.beginPath(); ctx.moveTo(-30, -55); ctx.quadraticCurveTo(-45, -20, -35, 10); ctx.stroke();
        // Right Arm
        ctx.beginPath(); ctx.moveTo(30, -55); ctx.quadraticCurveTo(45, -20, 35, 10); ctx.stroke();

        // --- HEAD ---
        ctx.fillStyle = '#ffedd5';
        ctx.beginPath();
        ctx.arc(0, -75, 28, 0, Math.PI * 2);
        ctx.fill();

        // Face
        ctx.fillStyle = '#4c1d95'; // Dark eyes
        ctx.beginPath(); ctx.arc(-10, -75, 3, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(10, -75, 3, 0, Math.PI * 2); ctx.fill();
        
        // Smile
        ctx.strokeStyle = '#9ca3af';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(0, -70, 5, 0.1, Math.PI - 0.1); ctx.stroke();

        // --- HAIR ---
        // Purple/Black messy hair
        ctx.fillStyle = '#2e1065'; 
        ctx.beginPath();
        ctx.arc(0, -80, 32, Math.PI, 0); // Top of head
        // Spikes/Bangs
        ctx.lineTo(32, -70); 
        ctx.quadraticCurveTo(35, -50, 20, -60);
        ctx.quadraticCurveTo(0, -55, -20, -60);
        ctx.quadraticCurveTo(-35, -50, -32, -70);
        ctx.closePath();
        ctx.fill();
        
        // Red Accent (Ember/Coral motif)
        ctx.fillStyle = '#f43f5e';
        ctx.beginPath();
        ctx.arc(20, -85, 4, 0, Math.PI * 2);
        ctx.fill();

        // --- SCARF/ACCESSORY ---
        ctx.strokeStyle = '#f43f5e'; // Rose/Red scarf floating
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(-15, -55);
        ctx.quadraticCurveTo(-40 + tailSway, -80, -60 + tailSway, -40 + finSway);
        ctx.stroke();

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

        // --- DRAW RAFAYEL (Character) ---
        // Position him in bottom right, floating
        const floatY = Math.sin(time * 0.03) * 15;
        const charX = width * 0.85;
        const charY = height - 150 + floatY;
        
        // Add a subtle glow behind him
        ctx.save();
        ctx.translate(charX, charY);
        const glow = ctx.createRadialGradient(0, -50, 50, 0, -50, 150);
        glow.addColorStop(0, 'rgba(139, 92, 246, 0.2)'); // Violet glow
        glow.addColorStop(1, 'transparent');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(0, -50, 150, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        // Draw the character
        drawRafayel(ctx, charX, charY, 0.8, time);

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