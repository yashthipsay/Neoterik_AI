'use client'

import React, { useEffect, useRef } from 'react';

interface ParametricPixelsProps {
  density?: number;
  speed?: number;
  color?: string;
  opacity?: number;
  className?: string;
}

export const ParametricPixels: React.FC<ParametricPixelsProps> = ({
  density = 20,
  speed = 0.5,
  color = '#419D78',
  opacity = 0.1,
  className = ''
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Parametric pixel system
    const pixels: Array<{
      x: number;
      y: number;
      size: number;
      phase: number;
      frequency: number;
      amplitude: number;
    }> = [];

    // Initialize pixels
    for (let i = 0; i < density; i++) {
      pixels.push({
        x: Math.random() * canvas.width / window.devicePixelRatio,
        y: Math.random() * canvas.height / window.devicePixelRatio,
        size: Math.random() * 3 + 1,
        phase: Math.random() * Math.PI * 2,
        frequency: Math.random() * 0.02 + 0.01,
        amplitude: Math.random() * 20 + 10
      });
    }

    let time = 0;

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width / window.devicePixelRatio, canvas.height / window.devicePixelRatio);
      
      time += speed;

      pixels.forEach((pixel, index) => {
        // Parametric movement
        const offsetX = Math.sin(time * pixel.frequency + pixel.phase) * pixel.amplitude;
        const offsetY = Math.cos(time * pixel.frequency * 0.7 + pixel.phase) * pixel.amplitude * 0.5;
        
        const x = pixel.x + offsetX;
        const y = pixel.y + offsetY;

        // Dynamic opacity based on position and time
        const dynamicOpacity = opacity * (0.5 + 0.5 * Math.sin(time * 0.01 + index * 0.1));

        // Create gradient for each pixel
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, pixel.size * 2);
        gradient.addColorStop(0, `${color}${Math.floor(dynamicOpacity * 255).toString(16).padStart(2, '0')}`);
        gradient.addColorStop(1, `${color}00`);

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, pixel.size, 0, Math.PI * 2);
        ctx.fill();

        // Add connecting lines between nearby pixels
        pixels.forEach((otherPixel, otherIndex) => {
          if (index !== otherIndex) {
            const otherX = otherPixel.x + Math.sin(time * otherPixel.frequency + otherPixel.phase) * otherPixel.amplitude;
            const otherY = otherPixel.y + Math.cos(time * otherPixel.frequency * 0.7 + otherPixel.phase) * otherPixel.amplitude * 0.5;
            
            const distance = Math.sqrt((x - otherX) ** 2 + (y - otherY) ** 2);
            
            if (distance < 50) {
              const lineOpacity = (1 - distance / 50) * opacity * 0.3;
              ctx.strokeStyle = `${color}${Math.floor(lineOpacity * 255).toString(16).padStart(2, '0')}`;
              ctx.lineWidth = 0.5;
              ctx.beginPath();
              ctx.moveTo(x, y);
              ctx.lineTo(otherX, otherY);
              ctx.stroke();
            }
          }
        });
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [density, speed, color, opacity]);

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 pointer-events-none ${className}`}
      style={{ width: '100%', height: '100%' }}
    />
  );
};

export const StaticParametricPattern: React.FC<{
  size?: number;
  color?: string;
  className?: string;
}> = ({ size = 100, color = '#419D78', className = '' }) => {
  return (
    <div className={`relative ${className}`} style={{ width: size, height: size }}>
      <svg
        width={size}
        height={size}
        viewBox="0 0 100 100"
        className="absolute inset-0"
      >
        <defs>
          <pattern id="parametric-grid" x="0" y="0" width="10" height="10" patternUnits="userSpaceOnUse">
            <circle cx="5" cy="5" r="0.5" fill={color} opacity="0.3" />
          </pattern>
          <radialGradient id="parametric-glow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor={color} stopOpacity="0.2" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </radialGradient>
        </defs>
        
        <rect width="100" height="100" fill="url(#parametric-grid)" />
        <circle cx="50" cy="50" r="40" fill="url(#parametric-glow)" />
        
        {/* Parametric curves */}
        <path
          d="M20,50 Q35,30 50,50 T80,50"
          stroke={color}
          strokeWidth="0.5"
          fill="none"
          opacity="0.4"
        />
        <path
          d="M50,20 Q30,35 50,50 T50,80"
          stroke={color}
          strokeWidth="0.5"
          fill="none"
          opacity="0.4"
        />
        
        {/* Circuit-like nodes */}
        <circle cx="35" cy="35" r="1" fill={color} opacity="0.6" />
        <circle cx="65" cy="35" r="1" fill={color} opacity="0.6" />
        <circle cx="35" cy="65" r="1" fill={color} opacity="0.6" />
        <circle cx="65" cy="65" r="1" fill={color} opacity="0.6" />
      </svg>
    </div>
  );
};