"""
Models for Cover Letter Generation Agent

Defines the Pydantic models used for input, output, and validation in the cover letter generation workflow.
These models ensure structured data exchange between the agent, workflow, and any external consumers.
"""

from pydantic import BaseModel
from typing import List, Optional

class CoverLetterInput(BaseModel):
    """
    Input schema for generating a cover letter.
    Contains all relevant job, applicant, and context information.
    """
    job_title: str
    hiring_company: str
    applicant_name: str
    job_description: Optional[str] = None
    preferred_qualifications: Optional[str] = None
    working_experience: Optional[str] = None  # e.g., "ML Engineer at VisionTech for 2 years; Research Scientist at AI Labs for 3 years"
    qualifications: Optional[str] = None      # e.g., "Master's in Artificial Intelligence"
    skillsets: Optional[str] = None           # e.g., "Python, TensorFlow, PyTorch, cloud deployment (AWS, GCP), computer vision, NLP, MLOps, scalable model deployment, data-driven decision making"
    company_culture_notes: Optional[str] = None  # e.g., "Innovative, collaborative, mission-driven"
    github_username: Optional[str] = None        # For enrichment if needed

class CoverLetterOutput(BaseModel):
    """
    Output schema for a generated cover letter.
    Includes the letter, optional summary, and metadata about used highlights and GitHub info.
    """
    cover_letter: str
    summary: Optional[str] = None
    used_highlights: Optional[List[str]] = None
    used_github_info: Optional[dict] = None

class CoverLetterValidationResult(BaseModel):
    """
    Result of validating a generated cover letter.
    Indicates validity and lists any issues found.
    """
    is_valid: bool
    issues: Optional[List[str]] = None

class CoverLetterGenerationResult(BaseModel):
    """
    Full result of a cover letter generation run, including input, output, validation, and errors.
    """
    input: CoverLetterInput
    output: CoverLetterOutput
    validation: Optional[CoverLetterValidationResult] = None
    error: Optional[str] = None