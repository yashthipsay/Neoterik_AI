'use client'

import React, { useState } from 'react';
import { Button } from './ui/Button';
import { Textarea } from './ui/Textarea';
import { Input } from './ui/Input';
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card';
import { ProgressBar } from './ui/ProgressBar';
import { StatusBadge } from './ui/StatusBadge';
import { Tooltip } from './ui/Tooltip';

interface CoverLetterGeneratorProps {
  onGenerate: (data: any) => void;
  isGenerating?: boolean;
}

export const CoverLetterGenerator: React.FC<CoverLetterGeneratorProps> = ({
  onGenerate,
  isGenerating = false
}) => {
  const [formData, setFormData] = useState({
    jobDescription: '',
    resumeHighlights: '',
    companyName: '',
    jobTitle: '',
    customInstructions: ''
  });
  
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [progress, setProgress] = useState(0);

  const validateForm = () => {
    const newErrors: Record<string, string> = {};
    
    if (!formData.jobDescription.trim()) {
      newErrors.jobDescription = 'Job description is required';
    }
    
    if (!formData.companyName.trim()) {
      newErrors.companyName = 'Company name is required';
    }
    
    if (!formData.jobTitle.trim()) {
      newErrors.jobTitle = 'Job title is required';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) return;
    
    // Simulate progress
    setProgress(0);
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) {
          clearInterval(interval);
          return 90;
        }
        return prev + 10;
      });
    }, 200);
    
    onGenerate(formData);
  };

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  return (
    <Card className="animate-fadeIn">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <span className="text-2xl">✨</span>
            Generate Your Cover Letter
          </CardTitle>
          <Tooltip content="AI-powered cover letter generation based on your resume and job requirements">
            <div className="w-5 h-5 bg-gray-200 rounded-full flex items-center justify-center text-xs cursor-help">
              ?
            </div>
          </Tooltip>
        </div>
        {isGenerating && (
          <div className="mt-4">
            <ProgressBar value={progress} showLabel color="primary" />
            <div className="flex items-center gap-2 mt-2">
              <div className="animate-spin w-4 h-4 border-2 border-[#419D78] border-t-transparent rounded-full" />
              <span className="text-sm text-gray-600">Generating your personalized cover letter...</span>
            </div>
          </div>
        )}
      </CardHeader>
      
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Input
              label="Company Name"
              placeholder="e.g., Google, Microsoft, Apple"
              value={formData.companyName}
              onChange={(e) => handleInputChange('companyName', e.target.value)}
              error={errors.companyName}
              icon={<span className="text-gray-400">🏢</span>}
            />
            
            <Input
              label="Job Title"
              placeholder="e.g., Software Engineer, Product Manager"
              value={formData.jobTitle}
              onChange={(e) => handleInputChange('jobTitle', e.target.value)}
              error={errors.jobTitle}
              icon={<span className="text-gray-400">💼</span>}
            />
          </div>
          
          <Textarea
            label="Job Description"
            placeholder="Paste the complete job description here..."
            value={formData.jobDescription}
            onChange={(e) => handleInputChange('jobDescription', e.target.value)}
            error={errors.jobDescription}
            helper="Include requirements, responsibilities, and company information for best results"
            rows={6}
          />
          
          <Textarea
            label="Resume Highlights (Optional)"
            placeholder="Key achievements, skills, or experiences you want to emphasize..."
            value={formData.resumeHighlights}
            onChange={(e) => handleInputChange('resumeHighlights', e.target.value)}
            helper="Highlight your most relevant experiences for this specific role"
            rows={4}
          />
          
          <Textarea
            label="Custom Instructions (Optional)"
            placeholder="Any specific tone, style, or points you want to include..."
            value={formData.customInstructions}
            onChange={(e) => handleInputChange('customInstructions', e.target.value)}
            helper="e.g., 'Keep it formal', 'Mention my passion for sustainability', etc."
            rows={3}
          />
          
          <div className="flex items-center justify-between pt-4 border-t">
            <div className="flex items-center gap-2">
              <StatusBadge status="info">AI-Powered</StatusBadge>
              <StatusBadge status="success">Personalized</StatusBadge>
            </div>
            
            <Button
              type="submit"
              loading={isGenerating}
              disabled={isGenerating}
              className="min-w-[140px]"
            >
              {isGenerating ? 'Generating...' : 'Generate Cover Letter'}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
};