"""
Models for Cover Letter Generation Agent

Defines the Pydantic models used for input, output, and validation in the cover letter generation workflow.
These models ensure structured data exchange between the agent, workflow, and any external consumers.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

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
    resume_data: Optional[Dict[str, Any]] = None
    github_data: Optional[Dict[str, Any]] = None
    github_username: Optional[str] = None        # For enrichment if needed
    applicant_experience_level: Optional[str] = None
    desired_tone: Optional[str] = None  # e.g., "formal", "friendly"

class CoverLetterOutput(BaseModel):
    cover_letter: str = Field(description="Generated cover letter text")
    summary: Optional[str] = Field(description="Summary of the generation process")
    used_highlights: Optional[List[str]] = Field(description="Resume highlights used in generation")
    used_github_info: Optional[Dict[str, Any]] = Field(description="GitHub information used in generation")  # Change to Dict

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