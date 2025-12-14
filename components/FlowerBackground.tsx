import React, { useRef, useEffect } from 'react';

const FlowerBackground: React.FC = () => {
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

    const petals = Array.from({ length: 60 }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      size: Math.random() * 5 + 3,
      speedY: Math.random() * 0.5 + 0.2,
      speedX: Math.random() * 0.4 - 0.2,
      rotation: Math.random() * Math.PI * 2,
      rotationSpeed: Math.random() * 0.02 - 0.01,
      color: Math.random() > 0.3 ? '#87CEFA' : '#E0FFFF' // Light sky blue & light cyan
    }));

    const flowers = Array.from({ length: 15 }, () => ({
      x: Math.random() * width,
      y: height - Math.random() * 200, // Bottom area
      size: Math.random() * 10 + 10,
      stemHeight: Math.random() * 100 + 50,
      swayOffset: Math.random() * Math.PI * 2,
      swaySpeed: Math.random() * 0.02 + 0.01
    }));
    
    // "Zzz" particles for the sleeping star
    const zzzs: {x: number, y: number, size: number, opacity: number}[] = [];

    let time = 0;

    const drawSleepingStar = (ctx: CanvasRenderingContext2D, x: number, y: number, t: number) => {
        const scale = 1.2 + Math.sin(t * 0.03) * 0.05; // Breathing
        const rot = Math.sin(t * 0.02) * 0.1; // Slight rocking

        ctx.save();
        ctx.translate(x, y);
        ctx.scale(scale, scale);
        ctx.rotate(rot);

        // --- STAR BODY ---
        // Draw a rounded 5-point star
        ctx.fillStyle = '#fde047'; // Yellow-300
        ctx.shadowColor = '#fcd34d'; // Glow
        ctx.shadowBlur = 20;
        
        ctx.beginPath();
        const outerRadius = 40;
        const innerRadius = 22; // Slightly chubby
        const spikes = 5;
        const offsetAngle = Math.PI / 10; 

        for (let i = 0; i < spikes * 2; i++) {
            const r = (i % 2 === 0) ? outerRadius : innerRadius;
            const angle = (Math.PI * i / spikes) - Math.PI / 2 + offsetAngle;
            const px = Math.cos(angle) * r;
            const py = Math.sin(angle) * r;
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.fill();
        
        // Remove shadow for details
        ctx.shadowBlur = 0;

        // --- FACE (Sleeping) ---
        // Eyes (Closed arcs)
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = '#854d0e'; // Brown
        ctx.lineCap = 'round';
        
        // Left Eye
        ctx.beginPath();
        ctx.arc(-12, 5, 7, 0.2, Math.PI - 0.2);
        ctx.stroke();

        // Right Eye
        ctx.beginPath();
        ctx.arc(12, 5, 7, 0.2, Math.PI - 0.2);
        ctx.stroke();
        
        // Mouth (Small 'o')
        ctx.fillStyle = '#ec4899'; // Pink
        ctx.beginPath();
        ctx.arc(0, 15, 3, 0, Math.PI * 2);
        ctx.fill();

        // Cheeks
        ctx.fillStyle = 'rgba(236, 72, 153, 0.3)';
        ctx.beginPath(); ctx.arc(-18, 12, 6, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(18, 12, 6, 0, Math.PI * 2); ctx.fill();

        // --- NIGHTCAP ---
        ctx.rotate(-0.25); // Tilt hat
        ctx.translate(-8, -35); // Position on top point

        // Cap body
        ctx.beginPath();
        ctx.moveTo(-16, 0);
        ctx.quadraticCurveTo(5, -45, 35, 15); // Droopy tail
        ctx.lineTo(16, 0);
        ctx.quadraticCurveTo(0, -10, -16, 0); // Brim curve
        ctx.fillStyle = '#93c5fd'; // Blue-300
        ctx.fill();

        // Brim
        ctx.fillStyle = '#eff6ff'; // White-ish
        ctx.beginPath();
        ctx.roundRect(-19, -6, 38, 12, 6);
        ctx.fill();

        // Pom-pom at end of tail
        ctx.beginPath();
        ctx.arc(35, 15, 9, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
    };

    const animate = () => {
        time++;
        // Soft Spring Background
        const bgGrad = ctx.createLinearGradient(0, 0, 0, height);
        bgGrad.addColorStop(0, '#f0fdf4'); // Very light green/white
        bgGrad.addColorStop(1, '#dcfce7'); // Light green
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, width, height);

        // Draw Standing Flowers (Background layer)
        flowers.forEach(f => {
            const sway = Math.sin(time * f.swaySpeed + f.swayOffset) * 10;
            const headX = f.x + sway;
            const headY = f.y - f.stemHeight;

            // Draw Stem
            ctx.beginPath();
            ctx.moveTo(f.x, f.y);
            ctx.quadraticCurveTo(f.x, f.y - f.stemHeight / 2, headX, headY);
            ctx.strokeStyle = '#4ade80';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Draw Flower Head (5 petals)
            ctx.fillStyle = '#60a5fa'; // Blue
            for (let i = 0; i < 5; i++) {
                const angle = (Math.PI * 2 * i) / 5;
                const px = headX + Math.cos(angle) * f.size;
                const py = headY + Math.sin(angle) * f.size;
                ctx.beginPath();
                ctx.arc(px, py, f.size * 0.6, 0, Math.PI * 2);
                ctx.fill();
            }
            // Center
            ctx.fillStyle = '#fef08a'; // Yellow center
            ctx.beginPath();
            ctx.arc(headX, headY, f.size * 0.4, 0, Math.PI * 2);
            ctx.fill();
        });

        // --- DRAW SLEEPING STAR ---
        const starX = width * 0.85;
        const starY = height - 80;
        
        // Shadow/Grounding for star
        ctx.fillStyle = 'rgba(20, 83, 45, 0.1)'; 
        ctx.beginPath();
        ctx.ellipse(starX, starY + 45, 45, 12, 0, 0, Math.PI * 2);
        ctx.fill();

        drawSleepingStar(ctx, starX, starY, time);

        // --- ZZZ ANIMATION ---
        if (time % 80 === 0) {
            zzzs.push({ x: starX + 25, y: starY - 40, size: 20, opacity: 1.0 });
        }
        
        for (let i = zzzs.length - 1; i >= 0; i--) {
            const z = zzzs[i];
            z.y -= 0.6; // Float up
            z.x += Math.sin(time * 0.05 + i) * 0.5; // Wiggle
            z.opacity -= 0.005; // Fade out
            z.size += 0.05; // Grow slightly
            
            if (z.opacity <= 0) {
                zzzs.splice(i, 1);
                continue;
            }

            ctx.fillStyle = `rgba(30, 58, 138, ${z.opacity})`; // Dark Blue text
            ctx.font = `bold ${Math.round(z.size)}px sans-serif`;
            ctx.fillText("Z", z.x, z.y);
        }

        // Draw Floating Petals (Foreground layer)
        petals.forEach(p => {
            p.y += p.speedY;
            p.x += Math.sin(time * 0.01) + p.speedX;
            p.rotation += p.rotationSpeed;

            if (p.y > height) {
                p.y = -20;
                p.x = Math.random() * width;
            }

            ctx.save();
            ctx.translate(p.x, p.y);
            ctx.rotate(p.rotation);
            ctx.fillStyle = p.color;
            // Draw simple petal shape
            ctx.beginPath();
            ctx.ellipse(0, 0, p.size, p.size * 0.6, 0, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
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

export default FlowerBackground;