"""
Models for GitHub Repo Parsing Agent

Defines the Pydantic models for representing GitHub repository and profile data.
These models are used to structure the output of the GitHub parsing agent and ensure
consistent data exchange within the workflow.
"""

from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class GitHubRepo(BaseModel):
    """
    Model representing a GitHub repository.
    """
    name: str
    description: Optional[str] = None
    url: HttpUrl
    language: Optional[str] = None
    stars: Optional[int] = None

class GitHubProfileData(BaseModel):
    """
    Model representing a GitHub user's profile data, including top repositories and inferred skills.
    """
    username: str
    name: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    public_repos_count: Optional[int] = None
    followers: Optional[int] = None
    following: Optional[int] = None
    top_repositories: Optional[List[GitHubRepo]] = [] # Example: Extract top N repos
    skills_tags: Optional[List[str]] = [] # Languages, topics etc. inferred by LLM

