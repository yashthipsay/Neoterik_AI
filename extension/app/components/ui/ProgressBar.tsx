'use client'

import React from 'react';
import { cn } from '@/lib/utils';

interface ProgressBarProps {
  value: number;
  max?: number;
  className?: string;
  showLabel?: boolean;
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error';
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  max = 100,
  className,
  showLabel = false,
  color = 'primary'
}) => {
  const percentage = Math.min((value / max) * 100, 100);
  
  const colors = {
    primary: 'from-[#419D78] to-[#37876A]',
    secondary: 'from-[#E0A458] to-[#C88F4B]',
    success: 'from-green-500 to-green-600',
    warning: 'from-yellow-500 to-yellow-600',
    error: 'from-red-500 to-red-600'
  };

  return (
    <div className={cn('w-full', className)}>
      {showLabel && (
        <div className="flex justify-between text-sm text-gray-600 mb-2">
          <span>Progress</span>
          <span>{Math.round(percentage)}%</span>
        </div>
      )}
      <div className="progress-bar">
        <div
          className={cn('progress-fill bg-gradient-to-r', colors[color])}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};