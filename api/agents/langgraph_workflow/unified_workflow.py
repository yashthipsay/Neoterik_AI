from pathlib import Path
from ..resume_parsing.agent import resume_agent, parse_resume_from_pdf
from ..repo_parsing.agent import github_agent
from ..cover_letter_generator.agent import cover_letter_agent
from ..cover_letter_generator.models import CoverLetterInput, CoverLetterOutput # Import CoverLetterOutput
from typing import TypedDict, Dict, Optional
from ..resume_parsing.models import ResumeData
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
import json
from supabase_client import supabase
import asyncio

from langgraph.graph import StateGraph, END, START

# Define the state schema
class AppState(TypedDict):
    context: Dict
    user_id: Optional[str]  # Optional user ID for tracking
    # resume_path: Optional[str]
    # github_username: Optional[str]
    
    
# --- Node 1: Resume Parsing ---
async def resume_node(state):
    print(f">>> Entered resume_node for user_id={state['user_id']}")
    context = state["context"]
    user_id = state["user_id"]
                        
    # Fetch parsed resume from Supabase
    res = supabase.table("documents").select("*").eq("user_id", user_id).eq("type", "resume").single().execute()
    docs = res.data
    if not docs or not docs.get("parsed_data"):
    # Fallback: Try to get uploaded resume file path for this user
    # You need to store the file path in raw_input when uploading
        resume_path = docs.get("raw_input", {}).get("resume_path") if docs else None
        if resume_path and Path(resume_path).exists():
            print(f"[WORKFLOW] Fallback: Parsing resume from file path for user_id={user_id}: {resume_path}")
            from agents.resume_parsing.agent import parse_resume_from_pdf
            parsed_data = await parse_resume_from_pdf(Path(resume_path))
            context["resume"] = parsed_data
            return {"context": context, "user_id": user_id}
    # If no file path or file not found, raise error
        raise Exception("No parsed resume found for this user. Please upload your resume first.")
    
    # If parsed data exists, use it directly
    print(f"[WORKFLOW] Fetched parsed resume data from Supabase for user_id={user_id}")
    context["resume"] = docs["parsed_data"]
    return {"context": context, "user_id": user_id}

    #  print(">>> Entered resume_node")
    # context = state["context"]
    # resume_path = state["resume_path"]
    
    # print("Running Resume Parser using parse_resume_from_pdf...")
    # # Convert resume_path to Path object if provided
    # path_obj = Path(resume_path) if resume_path else None
    
    # # Call the resume parsing agent tool and capture the parsed data
    # parsed_resume = await parse_resume_from_pdf(path_obj)
    
    # # Store the parsed resume in the context
    # context["resume"] = parsed_resume
    
    # return {"context": context, "resume_path": resume_path, "github_username": state["github_username"]}

    # Call the agent with the resume text
    # result = await resume_agent.run(resume_text)
    # context["resume"] = result
    # print(">>> Exiting resume_node with state:", {
    #     "context": context,
    #     "resume_path": resume_path,
    #     "github_username": state["github_username"]
    # })
    # return {"context": context, "resume_path": resume_path, "github_username": state["github_username"]}

# --- Node 2: GitHub Repo Parsing ---
async def github_node(state):
    print(f">>> Entered github_node for user_id={state['user_id']}")
    context = state["context"]
    user_id = state["user_id"]
    # Fetch parsed github from Supabase
    res = supabase.table("documents").select("*").eq("user_id", user_id).eq("type", "github").single().execute()
    docs = res.data
    if not docs or not docs.get("parsed_data"):
    # Fallback: Try to get github_username from raw_input and re-parse
        github_username = docs.get("raw_input", {}).get("github_username") if docs else None
        if github_username:
            print(f"[WORKFLOW] Fallback: Parsing GitHub data for user_id={user_id} using username={github_username}")
            from agents.repo_parsing.agent import github_agent
            parsed_data = await github_agent.run(github_username)
            context["github"] = parsed_data.data if hasattr(parsed_data, "data") else parsed_data
            return {"context": context, "user_id": user_id}
        # If no username or parsed data, raise error
        raise Exception("No parsed GitHub data found for this user. Please submit your GitHub username first.")
    
    print(f"[WORKFLOW] Fetched parsed GitHub data from Supabase for user_id={user_id}")
    context["github"] = docs["parsed_data"]
    return {"context": context, "user_id": user_id}

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

    SYSTEM_PROMPT = """
    You are a cover letter assistant. Your sole responsibility is to use the provided tools to generate a cover letter.
    **Crucial Instruction:** You must invoke the `retrieve_styles_and_generate_letter` tool to fulfill the user's request. Do not attempt to write the cover letter yourself.
    """
    print(f"[WORKFLOW] Entered cover_letter_node for user_id={state['user_id']}")
    context = state["context"]
    resume_result = context.get("resume", {})
    github_result = context.get("github", {})

    # print("\n--- DEBUG: resume_result ---")
    # print(repr(resume_result))
    # print("--- /DEBUG ---")

    # print("\n--- DEBUG: github_result ---")
    # print(repr(github_result))
    # print("--- /DEBUG ---")

    # Extract data from AgentRunResult if needed
    resume_dict = resume_result.data if hasattr(resume_result, "data") else resume_result
    github_dict = github_result.data if hasattr(github_result, "data") else github_result

    # print("\n--- DEBUG: resume (after .data if present) ---")
    # print(repr(resume))
    # print("--- /DEBUG ---")

    # print("\n--- DEBUG: github (after .data if present) ---")
    # print(repr(github))
    # print("--- /DEBUG ---")

    # --- FIX: Parse JSON if resume/github are strings ---
    if isinstance(resume_dict, str):
        try:
            resume_dict = json.loads(clean_json_string(resume_dict))
        except Exception as e:
            print("Error parsing resume JSON:", e)
            resume_dict = {}

    if isinstance(github_dict, str):
        try:
            github_dict = json.loads(clean_json_string(github_dict))
        except Exception as e:
            print(f"Error parsing github JSON: {e}")
            github_dict = {}

    # FIX: Ensure github is a proper dictionary, not just the username string
    if not isinstance(github_dict, dict):
        print(f"Warning: github data is not a dict, got {type(github_dict)}: {github_dict}")
        github_dict = {}

    # --- Create Pydantic models from the dictionaries ---
    # try:
    #     resume_data_model = ResumeData(**resume_dict)
    # except Exception as e:
    #     print(f"Error validating resume data with Pydantic model: {e}")
    #     resume_data_model = ResumeData() # Fallback to empty model

    # FIX: Extract GitHub username properly and create github info dict
    github_username = github_dict.get("username", context.get("github_username", ""))
    github_info_dict = github_dict if isinstance(github_dict, dict) and github_dict else {}

    cover_letter_input_model = CoverLetterInput(
        job_title=context.get("job_title", ""),
        hiring_company=context.get("hiring_company", ""),
        applicant_name=resume_dict.get("name", ""), # resume is now a dict
        job_description=context.get("job_description", ""),
        preferred_qualifications=context.get("preferred_qualifications", ""),
        working_experience="; ".join(
            f"{exp.get('title', '')} at {exp.get('company', '')} for {exp.get('duration', '')}"
            for exp in resume_dict.get("experience", [])
        ) if resume_dict.get("experience") else "",
        qualifications="; ".join(
            f"{edu.get('degree', '')} from {edu.get('institution', '')}"
            for edu in resume_dict.get("education", [])
            if isinstance(edu, dict)
        ) if resume_dict.get("education") else "",
        # Combine all individual skills from each skill group into a comma-separated string.
        skillsets=", ".join(
            skill for s in resume_dict.get("skills", []) for skill in s.get("skills", [])
        ) if resume_dict.get("skills") else "",
        company_culture_notes=context.get("company_culture_notes", ""),
        github_username=github_username,
        applicant_experience_level=context.get("applicant_experience_level", "mid"),
        desired_tone=context.get("desired_tone", "professional"),
        resume_data=resume_dict,
        github_data=github_dict,
    )
    
    
    try:
        # Call the agent with the properly structured input
        # Use the generate_with_style tool which handles RAG internally
        agent_result = await cover_letter_agent.run(SYSTEM_PROMPT, deps=cover_letter_input_model)
        
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

def show_graph():
    from IPython.display import Image, display
    graph = build_graph()
    img_data = graph.get_graph().draw_mermaid_png()
    with open("graph_visualization.png", "wb") as f:
        f.write(img_data)
    print("✅ Graph visualization saved as graph_visualization.png")
    display(Image(img_data))

