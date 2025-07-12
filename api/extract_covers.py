#!/usr/bin/env python3
import os
import sys
from docx import Document
import json
import re

def extract_cover_letter_content(file_path):
    try:
        doc = Document(file_path)
        content = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                content.append(paragraph.text.strip())
        
        full_text = '\n\n'.join(content)
        
        # Find the actual cover letter content (skip promotional content)
        lines = full_text.split('\n')
        
        # Look for the start of the actual cover letter
        start_idx = 0
        for i, line in enumerate(lines):
            # Look for date pattern or Dear
            if re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d+,\s+\d+', line) or line.strip().startswith('Dear'):
                start_idx = i
                break
        
        # If we found a date, look for the Dear line after it
        if start_idx > 0:
            for i in range(start_idx, len(lines)):
                if lines[i].strip().startswith('Dear'):
                    start_idx = i
                    break
        
        # Extract the main cover letter content
        cover_letter_lines = lines[start_idx:]
        return '\n\n'.join(cover_letter_lines)
    except Exception as e:
        print(f'Error reading {file_path}: {e}')
        return None

def main():
    cover_letters_dir = '/home/yasht/Neoterik_AI/api/agents/style_rag_agent/data/cover-letters'
    
    # Get all docx files
    docx_files = [f for f in os.listdir(cover_letters_dir) if f.endswith('.docx')]
    print(f'Found {len(docx_files)} docx files')
    
    # Process first few files
    for i, file in enumerate(docx_files[:5]):
        print(f'\n=== {file} ===')
        file_path = os.path.join(cover_letters_dir, file)
        content = extract_cover_letter_content(file_path)
        if content:
            print(content[:1500] + '...' if len(content) > 1500 else content)
        print('\n' + '='*80)

if __name__ == "__main__":
    main()
