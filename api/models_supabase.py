from pydantic import BaseModel

class UserIn(BaseModel):
    id: str
    email: str
    name: str
    avatar_url: str = ""
    github_username: str = ""

class DocumentIn(BaseModel):
    user_id: str
    type: str  # e.g. "resume" or "github"
    raw_input: dict
    parsed_data: dict