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
        program (str): The program or degree pursued.
        institution (str): The name of the educational institution.
        grade (str): The grade or CGPA achieved.
        year (str): The year or range of attendance.
    """
    program: str
    institution: str
    grade: str
    year: str

class Experience(BaseModel):
    """
    Model representing a work experience entry in a resume.

    Attributes:
        title (str): The job title held.
        company (str): The name of the company or organization.
        duration (str): The duration of employment.
        responsibilities (List[str]): A list of responsibilities and achievements.
        tools (List[str]): A list of tools and technologies used.
    """
    title: str
    company: str
    duration: str
    responsibilities: List[str]
    tools: List[str]

class Skill(BaseModel):
    """
    Model representing a skill category and its associated skills.

    Attributes:
        category (str): The category of the skill (e.g., Programming Languages, Frameworks).
        skills (List[str]): A list of specific skills within the category.
    """
    category: str
    skills: List[str]

class Project(BaseModel):
    """
    Model representing a project entry in a resume.

    Attributes:
        name (str): The name of the project.
        description (List[str]): A list of descriptions or key points about the project.
        tech_stack (List[str]): A list of technologies used in the project.
    """
    name: str
    description: List[str]
    tech_stack: List[str]

class ResumeData(BaseModel):
    """
    Model representing the full structured data extracted from a resume.

    Attributes:
        name (str): The name of the individual.
        education (List[Education]): A list of educational experiences.
        experience (List[Experience]): A list of work experiences.
        skills (List[Skill]): A list of skill categories and skills.
        projects (List[Project]): A list of projects.
        achievements (List[str]): A list of achievements.
    """
    name: str
    education: List[Education]
    experience: List[Experience]
    skills: List[Skill]
    projects: List[Project]
    achievements: List[str]