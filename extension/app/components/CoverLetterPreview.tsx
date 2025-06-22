'use client'

import React, { useState } from 'react';
import { Button } from './ui/Button';
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card';
import { StatusBadge } from './ui/StatusBadge';

interface CoverLetterPreviewProps {
  coverLetter: string;
  onEdit?: () => void;
  onSave?: () => void;
  onCopy?: () => void;
}

export const CoverLetterPreview: React.FC<CoverLetterPreviewProps> = ({
  coverLetter,
  onEdit,
  onSave,
  onCopy
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(coverLetter);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      onCopy?.();
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const wordCount = coverLetter.split(/\s+/).length;
  const readingTime = Math.ceil(wordCount / 200); // Average reading speed

  return (
    <Card className="animate-fadeIn" parametric={true} parametricIntensity="low">
      <CardHeader parametric={true}>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2" parametric={true}>
            <span className="text-2xl">📄</span>
            Your Cover Letter
          </CardTitle>
          <div className="flex items-center gap-2">
            <StatusBadge status="success">Generated</StatusBadge>
            <span className="text-sm text-gray-500 bg-gradient-to-r from-[#419D78]/10 to-[#E0A458]/10 px-2 py-1 rounded-full">
              {wordCount} words • {readingTime} min read
            </span>
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        <div className="bg-gradient-to-br from-gray-50 to-[#419D78]/5 rounded-lg p-6 mb-6 max-h-96 overflow-y-auto scrollbar-thin border border-[#419D78]/10">
          <div className="whitespace-pre-wrap text-gray-800 leading-relaxed">
            {coverLetter}
          </div>
        </div>
        
        <div className="flex flex-wrap gap-3">
          <Button
            onClick={handleCopy}
            variant="outline"
            className="flex items-center gap-2 border-[#419D78]/30 hover:bg-[#419D78]/5"
          >
            {copied ? (
              <>
                <svg className="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                Copied!
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                Copy to Clipboard
              </>
            )}
          </Button>
          
          {onEdit && (
            <Button
              onClick={onEdit}
              variant="outline"
              className="flex items-center gap-2 border-[#E0A458]/30 hover:bg-[#E0A458]/5"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
              Edit
            </Button>
          )}
          
          {onSave && (
            <Button
              onClick={onSave}
              className="flex items-center gap-2 bg-gradient-to-r from-[#419D78] to-[#37876A] hover:shadow-lg hover:shadow-[#419D78]/25"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
              </svg>
              Save to Library
            </Button>
          )}
        </div>
        
        <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-[#419D78]/5 rounded-lg border border-blue-200/50">
          <div className="flex items-start gap-3">
            <div className="text-blue-500 text-xl">💡</div>
            <div>
              <h4 className="font-medium text-blue-900 mb-1">Pro Tips</h4>
              <ul className="text-sm text-blue-800 space-y-1">
                <li>• Review and customize the letter to match your voice</li>
                <li>• Ensure all company and role details are accurate</li>
                <li>• Consider adding specific examples from your experience</li>
              </ul>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};