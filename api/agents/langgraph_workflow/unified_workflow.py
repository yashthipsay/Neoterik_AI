from pathlib import Path
from ..resume_parsing.agent import resume_agent
from ..repo_parsing.agent import github_agent
from ..cover_letter_generator.agent import cover_letter_agent, build_prompt_for_gemini
from ..cover_letter_generator.models import CoverLetterInput, CoverLetterOutput
from typing import TypedDict, Dict, Optional
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
import json

from langgraph.graph import StateGraph, END

# --- Define the state schema ---
class AppState(TypedDict):
    context: Dict
    resume_data: Optional[dict]
    github_data: Optional[dict]
    resume_path: Optional[str]  # legacy/testing only
    github_username: Optional[str]  # legacy/testing only

# --- Node 1: Resume Parsing or Use Parsed Data ---
async def resume_node(state):
    print(">>> Entered resume_node")
    context = state["context"]
    # Use parsed resume_data if present (from Supabase)
    if "resume_data" in state and state["resume_data"]:
        print("Using parsed resume_data from state (Supabase).")
        context["resume"] = state["resume_data"]
        return {"context": context, "resume_data": state["resume_data"], "github_data": state.get("github_data")}
    # Legacy: parse from file if resume_data not present
    resume_path = state.get("resume_path")
    print(f"Running Resume Parser... resume_path={resume_path}")
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
        print("No resume_path provided, using empty resume_text.")
        resume_text = ""
    # Call the agent with the resume text
    result = await resume_agent.run(resume_text)
    print("Resume agent result:", result)
    context["resume"] = result
    print(">>> Exiting resume_node with state:", {
        "context": context,
        "resume_data": result,
        "github_data": state.get("github_data")
    })
    return {"context": context, "resume_data": result, "github_data": state.get("github_data")}

# --- Node 2: GitHub Repo Parsing or Use Parsed Data ---
async def github_node(state):
    print(">>> Entered github_node")
    context = state["context"]
    # Use parsed github_data if present (from Supabase)
    if "github_data" in state and state["github_data"]:
        print("Using parsed github_data from state (Supabase).")
        context["github"] = state["github_data"]
        return {"context": context, "resume_data": state.get("resume_data"), "github_data": state["github_data"]}
    # Legacy: parse from username if github_data not present
    github_username = state.get("github_username")
    print(f"Running GitHub Parser... github_username={github_username}")
    if not github_username:
        print("No github_username provided, returning empty github dict.")
        context["github"] = {}
        return {"context": context, "resume_data": state.get("resume_data"), "github_data": {}}
    github_result = await github_agent.run(github_username)
    github_data = github_result.data if hasattr(github_result, "data") else github_result
    print("GitHub agent result:", github_data)
    context["github"] = github_data
    print(">>> Exiting github_node with state:", {
        "context": context,
        "resume_data": state.get("resume_data"),
        "github_data": github_data
    })
    return {"context": context, "resume_data": state.get("resume_data"), "github_data": github_data}

def clean_json_string(s):
    """Remove markdown fences, description text, and whitespace from a string."""
    s = s.strip()
    json_start = s.find("```json")
    if json_start != -1:
        json_content_start = s.find("\n", json_start) + 1
        json_end = s.find("```", json_content_start)
        if json_end != -1:
            return s[json_content_start:json_end].strip()
    json_start = s.find("{")
    json_end = s.rfind("}")
    if json_start != -1 and json_end != -1 and json_end > json_start:
        return s[json_start:json_end+1].strip()
    return s

# --- Node 3: Cover Letter Generation ---
async def cover_letter_node(state):
    print(">>> Entered cover_letter_node")
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

    print("\n--- DEBUG: github_info_dict ---")
    print(repr(github_info_dict))
    print("--- /DEBUG ---")

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
            if isinstance(edu, dict)
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
    prompt_str = build_prompt_for_gemini(
        cover_letter_input_model, 
        github_info=github_info_str, 
        resume_data=resume_highlights_str
    )
    print("\n--- DEBUG: Prompt for Gemini ---")
    print(prompt_str)
    print("--- /DEBUG ---")
    
    try:
        # Call the agent with the properly structured input
        # Use the generate_with_style tool which handles RAG internally
        agent_result = await cover_letter_agent.run(prompt_str, deps=cover_letter_input_model)
        print("Cover letter agent result:", agent_result)
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
                "used_github_info": github_info_dict
            }
    except Exception as e:
        print(f"Error in cover letter generation: {e}")
        # Provide a fallback response
        context["cover_letter"] = {
            "cover_letter": f"Error generating cover letter: {str(e)}",
            "summary": None,
            "used_highlights": None,
            "used_github_info": github_info_dict
        }
    print(">>> Exiting cover_letter_node with context keys:", list(context.keys()))
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

