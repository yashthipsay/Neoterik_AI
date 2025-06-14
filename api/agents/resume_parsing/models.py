"""
Models for Resume Parsing Agent

Defines the Pydantic models for representing education, experience, skills, and the overall resume data structure.
These models are used to structure the output of the resume parsing agent and ensure consistent data exchange.
"""

from pydantic import BaseModel
from typing import List, Optional

class Education(BaseModel):
    """
    Model representing an education entry in a resume.

    Attributes:
        degree (str): The degree obtained or pursued.
        institution (str): The name of the educational institution.
        year (str): The year of graduation or attendance.
        gpa (Optional[float]): The grade point average, if available.
    """
    degree: str
    institution: str
    year: str
    gpa: Optional[float] = None

class Experience(BaseModel):
    """
    Model representing a work experience entry in a resume.

    Attributes:
        title (str): The job title held.
        company (str): The name of the company or organization.
        duration (str): The duration of employment.
        description (List[str]): A list of responsibilities and achievements.
    """
    title: str
    company: str
    duration: str
    description: List[str]

class Skill(BaseModel):
    """
    Model representing a skill category and its associated skills.

    Attributes:
        category (str): The category of the skill (e.g., programming languages, tools).
        skills (List[str]): A list of specific skills within the category.
    """
    category: str
    skills: List[str]

class ResumeData(BaseModel):
    """
    Model representing the full structured data extracted from a resume.

    Attributes:
        name (str): The name of the individual.
        email (Optional[str]): The email address.
        phone (Optional[str]): The phone number.
        education (List[Education]): A list of educational experiences.
        experience (List[Experience]): A list of work experiences.
        skills (List[Skill]): A list of skill categories and skills.
        summary (Optional[str]): A brief summary or objective statement.
    """
    name: str
    email: Optional[str]
    phone: Optional[str]
    education: List[Education]
    experience: List[Experience]
    skills: List[Skill]
    summary: Optional[str]