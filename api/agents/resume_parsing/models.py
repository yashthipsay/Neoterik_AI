from pydantic import BaseModel
from typing import List, Optional

class Education(BaseModel):
    degree: str
    institution: str
    year: str
    gpa: Optional[float] = None

class Experience(BaseModel):
    title: str
    company: str
    duration: str
    description: List[str]

class Skill(BaseModel):
    category: str
    skills: List[str]

class ResumeData(BaseModel):
    name: str
    email: Optional[str]
    phone: Optional[str]
    education: List[Education]
    experience: List[Experience]
    skills: List[Skill]
    summary: Optional[str]