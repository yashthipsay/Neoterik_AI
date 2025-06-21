from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

class StyleSelectionInput(BaseModel):
    job_title: str
    hiring_company: str
    job_description: Optional[str]
    preferred_qualifications: Optional[str]
    company_culture_notes: Optional[str]
    applicant_experience_level: Optional[str]  # e.g., "entry-level", "senior"
    desired_tone: Optional[str]  # e.g., "formal", "friendly"

class StyleSelectionOutput(BaseModel):
    selected_template: Dict[str, Any] = Field(description="Selected template details")
    tone: str = Field(description="Selected tone")
    style: str = Field(description="Selected style")  # Make sure this is required
    industry: str = Field(description="Selected industry")
    level: str = Field(description="Selected level")
    retrieved_documents: List[Dict[str, Any]] = Field(description="Documents retrieved from RAG")  # Make sure this is required

class CoverLetterTemplate(BaseModel):
    tone: str  # e.g., "formal", "friendly"
    industry: str  # e.g., "tech", "blockchain", "general"
    level: str  # e.g., "entry-level", "senior"
    style: str  # e.g., "best_candidate", "skills_focused"
    content: str  # Full template text
    keywords: Optional[List[str]]
    metadata: Optional[Dict[str, str]] = Field(default_factory=dict)

class ToneGuideline(BaseModel):
    tone: str
    description: str
    keywords: List[str]
    example: str

class IndustryPhrase(BaseModel):
    industry: str
    phrases: List[str]

class SkillSnippet(BaseModel):
    skill: str
    description: str

class CompanyValue(BaseModel):
    value: str
    example_sentence: str

class GlobalRAGData(BaseModel):
    templates: List[CoverLetterTemplate]
    tones: List[ToneGuideline]
    phrases: List[IndustryPhrase]
    skills: List[SkillSnippet]
    values: List[CompanyValue]