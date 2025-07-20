'use client'

import React from 'react';
import { Button } from './ui/Button';
import { Card } from './ui/Card';

interface PricingTier {
  name: string;
  price: string;
  period: string;
  description: string;
  features: string[];
  popular?: boolean;
  buttonText: string;
  buttonVariant?: 'primary' | 'secondary' | 'outline';
}

interface PricingCardProps {
  tier: PricingTier;
  onSelect: (tier: PricingTier) => void;
}

export const PricingCard: React.FC<PricingCardProps> = ({ tier, onSelect }) => {
  return (
    <Card 
      className={`relative text-center transition-all duration-300 bg-[#1a1a1a] border-gray-800 ${
        tier.popular 
          ? 'ring-2 ring-[#419D78] scale-105 shadow-2xl border-[#419D78]/50' 
          : 'hover:scale-105 hover:border-[#419D78]/30'
      }`}
      hover={!tier.popular}
    >
      {tier.popular && (
        <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
          <span className="bg-gradient-to-r from-[#419D78] to-[#37876A] text-white px-4 py-1 rounded-full text-sm font-semibold">
            Most Popular
          </span>
        </div>
      )}
      
      <div className="pt-6">
        <h3 className="text-xl font-bold text-gray-100 mb-2">{tier.name}</h3>
        <p className="text-gray-400 mb-4">{tier.description}</p>
        
        <div className="mb-6">
          <span className="text-4xl font-bold text-gray-100">{tier.price}</span>
          <span className="text-gray-400">/{tier.period}</span>
        </div>
        
        <ul className="space-y-3 mb-8 text-left">
          {tier.features.map((feature, index) => (
            <li key={index} className="flex items-center gap-3">
              <svg className="w-5 h-5 text-[#419D78] flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
              <span className="text-gray-300">{feature}</span>
            </li>
          ))}
        </ul>
        
        <Button
          onClick={() => onSelect(tier)}
          variant={tier.buttonVariant || (tier.popular ? 'primary' : 'outline')}
          className="w-full"
        >
          {tier.buttonText}
        </Button>
      </div>
    </Card>
  );
};