from pathlib import Path
from ..resume_parsing.agent import resume_agent
from ..repo_parsing.agent import github_agent
from ..cover_letter_generator.agent import cover_letter_agent, build_prompt, generate_with_style # Make sure to import the agent and build_prompt
from ..cover_letter_generator.models import CoverLetterInput, CoverLetterOutput # Import CoverLetterOutput
from typing import TypedDict, Dict, Optional
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
import json
import asyncio

from langgraph.graph import StateGraph, END, START


# Define the state schema
class AppState(TypedDict):
    context: Dict
    resume_path: Optional[str]
    github_username: Optional[str]
    
    
# --- Node 1: Resume Parsing ---
async def resume_node(state):
    print(">>> Entered resume_node")
    context = state["context"]
    resume_path = state["resume_path"]
    print("Running Resume Parser...")

    resume_text = ""
    if resume_path:
        suffix = Path(resume_path).suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(resume_path)
            docs = loader.load()
            resume_text = "\n".join([doc.page_content for doc in docs])
        elif suffix in [".docx", ".doc"]:
            loader = Docx2txtLoader(resume_path)
            docs = loader.load()
            resume_text = "\n".join([doc.page_content for doc in docs])
        else:
            raise ValueError("Unsupported resume file type.")
    else:
        resume_text = ""  # Or load a sample resume text

    # Call the agent with the resume text
    result = await resume_agent.run(resume_text)
    context["resume"] = result
    print(">>> Exiting resume_node with state:", {
        "context": context,
        "resume_path": resume_path,
        "github_username": state["github_username"]
    })
    return {"context": context, "resume_path": resume_path, "github_username": state["github_username"]}

# --- Node 2: GitHub Repo Parsing ---
async def github_node(state):
    context = state["context"]
    github_username = state["github_username"]
    print("Running GitHub Parser...")
    if not github_username:
        context["github"] = {}
        return {"context": context, "resume_path": state["resume_path"], "github_username": github_username}
    github_result = await github_agent.run(github_username)
    # If your tool returns a .data attribute, use that; otherwise, use the result directly
    github_data = github_result.data if hasattr(github_result, "data") else github_result
    context["github"] = github_data
    return {"context": context, "resume_path": state["resume_path"], "github_username": github_username}

def clean_json_string(s):
    """Remove markdown fences, description text, and whitespace from a string."""
    s = s.strip()
    
    # Find the JSON block within the string
    json_start = s.find("```json")
    if json_start != -1:
        # Find the start of actual JSON after the markdown fence
        json_content_start = s.find("\n", json_start) + 1
        # Find the end of the JSON block
        json_end = s.find("```", json_content_start)
        if json_end != -1:
            return s[json_content_start:json_end].strip()
    
    # If no markdown fences found, try to find JSON object directly
    json_start = s.find("{")
    json_end = s.rfind("}")
    if json_start != -1 and json_end != -1 and json_end > json_start:
        return s[json_start:json_end+1].strip()
    
    # Return original if no JSON pattern found
    return s

async def cover_letter_node(state):
    context = state["context"]
    resume_result = context.get("resume", {})
    github_result = context.get("github", {})

    print("\n--- DEBUG: resume_result ---")
    print(repr(resume_result))
    print("--- /DEBUG ---")

    print("\n--- DEBUG: github_result ---")
    print(repr(github_result))
    print("--- /DEBUG ---")

    # Extract data from AgentRunResult if needed
    resume = resume_result.data if hasattr(resume_result, "data") else resume_result
    github = github_result.data if hasattr(github_result, "data") else github_result

    print("\n--- DEBUG: resume (after .data if present) ---")
    print(repr(resume))
    print("--- /DEBUG ---")

    print("\n--- DEBUG: github (after .data if present) ---")
    print(repr(github))
    print("--- /DEBUG ---")

    # --- FIX: Parse JSON if resume/github are strings ---
    if isinstance(resume, str):
        try:
            resume = json.loads(clean_json_string(resume))
        except Exception as e:
            print("Error parsing resume JSON:", e)
            resume = {}

    if isinstance(github, str):
        try:
            github = json.loads(clean_json_string(github))
        except Exception as e:
            print("Error parsing github JSON:", e)
            github = {}

    # FIX: Ensure github is a proper dictionary, not just the username string
    if not isinstance(github, dict):
        print(f"Warning: github data is not a dict, got {type(github)}: {github}")
        github = {}

    # FIX: Extract GitHub username properly and create github info dict
    github_username = github.get("username", context.get("github_username", ""))
    github_info_dict = github if isinstance(github, dict) and github else {}

    cover_letter_input_model = CoverLetterInput(
        job_title=context.get("job_title", ""),
        hiring_company=context.get("hiring_company", ""),
        applicant_name=resume.get("name", ""), # resume is now a dict
        job_description=context.get("job_description", ""),
        preferred_qualifications=context.get("preferred_qualifications", ""),
        working_experience="; ".join(
            f"{exp.get('title', '')} at {exp.get('company', '')} for {exp.get('duration', '')}"
            for exp in resume.get("experience", [])
        ) if resume.get("experience") else "",
        qualifications="; ".join(
            f"{edu.get('degree', '')} from {edu.get('institution', '')}"
            for edu in resume.get("education", [])
        ) if resume.get("education") else "",
        skillsets=", ".join(resume.get("skills", [])) if resume.get("skills") else "",
        company_culture_notes=context.get("company_culture_notes", ""),
        github_username=github_username, # Use extracted username
        applicant_experience_level=context.get("applicant_experience_level", "mid"),  # Default to mid-level
        desired_tone=context.get("desired_tone", "professional"), 
    )
    
    # FIX: Convert github dict to string properly for the prompt
    github_info_str = json.dumps(github_info_dict, indent=2) if github_info_dict else ""
    resume_highlights_str = context.get("resume_highlights", "")
    prompt_str = build_prompt(
        cover_letter_input_model, 
        github_info=github_info_str, 
        resume_highlights=resume_highlights_str
    )
    
    try:
        # Call the agent with the properly structured input
        # Use the generate_with_style tool which handles RAG internally
        agent_result = await cover_letter_agent.run(prompt_str, deps=cover_letter_input_model)
        
        # Extract the result
        output_data = agent_result.data if hasattr(agent_result, "data") else agent_result
        
        # Store the structured output as a dictionary in the context
        if hasattr(output_data, "model_dump"):
            context["cover_letter"] = output_data.model_dump()
        elif hasattr(output_data, "dict"):
            context["cover_letter"] = output_data.dict()
        else:
            # Fallback for plain text or other formats
            context["cover_letter"] = {
                "cover_letter": str(output_data),
                "summary": None,
                "used_highlights": None,
                "used_github_info": github_info_dict  # FIX: Use the github dict
            }
    
    except Exception as e:
        print(f"Error in cover letter generation: {e}")
        # Provide a fallback response
        context["cover_letter"] = {
            "cover_letter": f"Error generating cover letter: {str(e)}",
            "summary": None,
            "used_highlights": None,
            "used_github_info": github_info_dict  # FIX: Use the github dict
        }
    
    return {"context": context}

# --- Define the LangGraph graph ---
def build_graph():
    workflow = StateGraph(AppState)
    workflow.add_node("resume", resume_node)
    workflow.add_node("github", github_node)
    workflow.add_node("cover_letter", cover_letter_node)

    # Edges: resume -> github -> cover_letter -> END
    workflow.add_edge("resume", "github")
    workflow.add_edge("github", "cover_letter")
    workflow.add_edge("cover_letter", END)

    workflow.set_entry_point("resume")
    return workflow.compile()