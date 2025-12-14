import React, { useRef, useEffect } from 'react';

const CosmicBackground: React.FC = () => {
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

    // Initialize Stars
    const stars = Array.from({ length: 300 }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      size: Math.random() * 1.5 + 0.5,
      speed: Math.random() * 0.3 + 0.05,
      brightness: Math.random(),
      twinkleSpeed: Math.random() * 0.05
    }));

    // Initialize Planets
    const planets = [
        { x: width * 0.85, y: height * 0.2, radius: 60, color1: '#4c1d95', color2: '#2e1065', orbitSpeed: 0.0001, angle: 0 },
        { x: width * 0.15, y: height * 0.85, radius: 90, color1: '#1e3a8a', color2: '#172554', orbitSpeed: 0.0002, angle: Math.PI }
    ];

    let mouseX = 0;
    let mouseY = 0;

    const handleMouseMove = (e: MouseEvent) => {
        // Calculate parallax offset
        mouseX = (e.clientX - width/2) * 0.02;
        mouseY = (e.clientY - height/2) * 0.02;
    };

    window.addEventListener('mousemove', handleMouseMove);

    const animate = () => {
        // Deep space background
        const bgGrad = ctx.createLinearGradient(0, 0, 0, height);
        bgGrad.addColorStop(0, '#020617');
        bgGrad.addColorStop(1, '#0f172a');
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, width, height);

        // Draw Planets
        planets.forEach(p => {
             // Subtle orbit movement
             p.angle += p.orbitSpeed;
             const orbitX = p.x + Math.cos(p.angle) * 20; 
             const orbitY = p.y + Math.sin(p.angle) * 20;

             // Parallax applied to position
             const drawX = orbitX - mouseX * 2; // Planets move differently than stars
             const drawY = orbitY - mouseY * 2;

             // Planet Glow
             const glow = ctx.createRadialGradient(drawX, drawY, p.radius * 0.8, drawX, drawY, p.radius * 2);
             glow.addColorStop(0, p.color1);
             glow.addColorStop(1, 'transparent');
             ctx.fillStyle = glow;
             ctx.beginPath();
             ctx.arc(drawX, drawY, p.radius * 2, 0, Math.PI * 2);
             ctx.fill();

             // Planet Body
             const body = ctx.createRadialGradient(drawX - p.radius/3, drawY - p.radius/3, 0, drawX, drawY, p.radius);
             body.addColorStop(0, p.color1);
             body.addColorStop(1, p.color2);
             ctx.fillStyle = body;
             ctx.beginPath();
             ctx.arc(drawX, drawY, p.radius, 0, Math.PI * 2);
             ctx.fill();
        });

        // Draw Stars
        stars.forEach(star => {
            // Update brightness for twinkle
            star.brightness += star.twinkleSpeed;
            if (star.brightness > 1 || star.brightness < 0.3) {
                star.twinkleSpeed *= -1;
            }

            // Move stars upwards slowly
            star.y -= star.speed;
            if (star.y < 0) {
                star.y = height;
                star.x = Math.random() * width;
            }

            // Apply Parallax
            const x = star.x - mouseX * star.speed * 5; 
            const y = star.y - mouseY * star.speed * 5;

            ctx.globalAlpha = Math.max(0, Math.min(1, star.brightness));
            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.arc(x, y, star.size, 0, Math.PI * 2);
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

export default CosmicBackground;