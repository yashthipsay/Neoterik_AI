from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class GitHubRepo(BaseModel):
    name: str
    description: Optional[str] = None
    url: HttpUrl
    language: Optional[str] = None
    stars: Optional[int] = None

class GitHubProfileData(BaseModel):
    username: str
    name: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    public_repos_count: Optional[int] = None
    followers: Optional[int] = None
    following: Optional[int] = None
    top_repositories: Optional[List[GitHubRepo]] = [] # Example: Extract top N repos
    skills_tags: Optional[List[str]] = [] # Languages, topics etc. inferred by LLM
    
