'use client'

import React from 'react';
import { cn } from '@/lib/utils';
import { ParametricPixels, StaticParametricPattern } from './ParametricPixels';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  premium?: boolean;
  parametric?: boolean;
  parametricIntensity?: 'low' | 'medium' | 'high';
}

export const Card: React.FC<CardProps> = ({ 
  children, 
  className, 
  hover = true,
  premium = false,
  parametric = false,
  parametricIntensity = 'medium'
}) => {
  const intensityConfig = {
    low: { density: 10, opacity: 0.05, speed: 0.3 },
    medium: { density: 20, opacity: 0.1, speed: 0.5 },
    high: { density: 35, opacity: 0.15, speed: 0.8 }
  };

  const config = intensityConfig[parametricIntensity];

  return (
    <div
      className={cn(
        'relative bg-white rounded-2xl p-6 shadow-lg border border-gray-100 transition-all duration-300 overflow-hidden',
        hover && 'hover:shadow-xl hover:-translate-y-1',
        premium && 'bg-gradient-to-br from-purple-500 to-blue-600 text-white',
        parametric && 'border-[#419D78]/20',
        className
      )}
    >
      {parametric && (
        <>
          <ParametricPixels
            density={config.density}
            speed={config.speed}
            opacity={config.opacity}
            color="#419D78"
          />
          <div className="absolute top-2 right-2">
            <StaticParametricPattern size={24} color="#419D78" />
          </div>
        </>
      )}
      <div className="relative z-10">
        {children}
      </div>
    </div>
  );
};

interface CardHeaderProps {
  children: React.ReactNode;
  className?: string;
  parametric?: boolean;
}

export const CardHeader: React.FC<CardHeaderProps> = ({ 
  children, 
  className,
  parametric = false 
}) => {
  return (
    <div className={cn('mb-4 relative', className)}>
      {parametric && (
        <div className="absolute -left-2 top-0 bottom-0 w-1 bg-gradient-to-b from-[#419D78] to-[#E0A458] rounded-full opacity-60" />
      )}
      {children}
    </div>
  );
};

interface CardTitleProps {
  children: React.ReactNode;
  className?: string;
  parametric?: boolean;
}

export const CardTitle: React.FC<CardTitleProps> = ({ 
  children, 
  className,
  parametric = false 
}) => {
  return (
    <h3 className={cn(
      'text-lg font-semibold text-gray-900 relative',
      parametric && 'text-[#2D3047] font-bold tracking-wide',
      className
    )}>
      {parametric && (
        <span className="absolute -left-6 top-1/2 transform -translate-y-1/2 w-2 h-2 bg-[#419D78] rounded-full opacity-70" />
      )}
      {children}
    </h3>
  );
};

interface CardContentProps {
  children: React.ReactNode;
  className?: string;
}

export const CardContent: React.FC<CardContentProps> = ({ children, className }) => {
  return (
    <div className={cn('text-gray-600', className)}>
      {children}
    </div>
  );
};