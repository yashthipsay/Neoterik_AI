from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import BaseModel, HttpUrl
from supabase_client import supabase
from models_supabase import UserIn, DocumentIn
import re
from pathlib import Path
import shutil
import tempfile
from fastapi.middleware.cors import CORSMiddleware
from agents.resume_parsing.workflow import create_resume_parsing_workflow
from agents.repo_parsing.workflow import create_github_parsing_workflow 
from agents.resume_parsing.agent import SAMPLE_RESUME_PATH
# from agents.cover_letter_generator.agent import CoverLetterAgent, build_prompt
from agents.cover_letter_generator.models import CoverLetterInput
from agents.langgraph_workflow.unified_workflow import build_graph, show_graph
from company_research_graph import build_detect_only_graph, DetectOnlyState,  run_job_research  # Import the job research function
import asyncio
from fastapi.responses import JSONResponse


app = FastAPI(
    title="Job URL Detector API",
    description="API to parse resumes and GitHub profiles using AI agents."
)

# Add CORS middleware to allow extension to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your extension's origin
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class RegisterUserRequest(BaseModel):
    id: str
    email: str
    name: str
    avatar_url: str = ""
    github_username: str = ""


@app.post("/register-user")
async def register_user(user: RegisterUserRequest):
    try:
        data = {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "avatar_url": user.avatar_url,
            "github_username": user.github_username,
        }
        print("Registering user with data:", data)
        result = supabase.table("users").upsert(data, on_conflict="id").execute()
        print("Supabase result:", result)
        # No .error attribute; just return the data
        return {"success": True, "user": result.data}
    except Exception as e:
        import traceback
        print("Register user error:", e)
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to register user: {str(e)}")

@app.get("/test-supabase")
def test_supabase():
    result = supabase.table("users").select("*").limit(1).execute()
    return result.data

class URLCheckRequest(BaseModel):
    url: str
    
class GitHubParseRequest(BaseModel): # Model for the new endpoint's input
    github_username: str

# Comprehensive list of job board URL patterns
KNOWN_JOB_BOARD_PATTERNS = [
    r"job-boards\.greenhouse\.io/.+/jobs/\d+",
    r"jobs\.lever\.co/.+/\d+",
    r"boards\.greenhouse\.io/.+/#application_form",
    r"apply\.workable\.com/.+/j/.+",
    r"jobs\.ashbyhq\.com/.+/.+", 
    r"careers\.smartrecruiters\.com/.+/job/.+",
    r"workday\..+/careers/.+/job/.+",
    r"linkedin\.com/jobs/view/.+",
    r"indeed\.com/.+/viewjob",
    r"wellfound\.com/jobs/.+",
    # Add more patterns as needed
]

@app.post("/check-url")
async def check_url(data: URLCheckRequest):
    try:
        # Run detect_node only
        graph = build_detect_only_graph()
        initial_state = DetectOnlyState(
            job_url=data.url,
            is_job_page=False,
            scraped_html="",
            job_title="",
            company_name=""
        )
        result = await graph.ainvoke(initial_state)

        # Return is_job_application only based on detect_node
        return JSONResponse(content={
            "is_job_application": result.get("is_job_page", False),
            "job_title": result.get("job_title", ""),
            "company_name": result.get("company_name", "")
        }, status_code=200)
        
    except Exception as e:
        return JSONResponse(content={
            "is_job_application": False,
            "error": str(e)
        }, status_code=500)
    
# Endpoint to run the full job research graph
@app.post("/run-agent")
async def run_agent_api(data: URLCheckRequest):
    try:
        parsed_output = await run_job_research(data.url)  # full graph
        return parsed_output.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# def is_job_application_url(url: str) -> bool:
#     """Check if URL matches any known job board patterns"""
#     return any(re.search(pattern, url) for pattern in KNOWN_JOB_BOARD_PATTERNS)

# @app.post("/check-url")
# async def check_url(data: URLCheckRequest):
#     try:
#         result = await run_job_research(data.url)
#         if result:
#             return JSONResponse(content={
#                 "is_job_application": True,  # ✅ always true if graph succeeded
#                 "parsed_output": result.model_dump()
#             }, status_code=200)
#         return JSONResponse(content={
#             "is_job_application": False,
#             "parsed_output": None
#         }, status_code=204)
#     except Exception as e:
#         return JSONResponse(content={
#             "is_job_application": False,
#             "error": str(e)
#         }, status_code=500)
    # """Check if the provided URL is a job application page"""
    # is_job_url = is_job_application_url(data.url)
    # print(f"Checking URL: {data.url} - Result: {'✓' if is_job_url else '✗'}")
    # return {"is_job_application": is_job_url, "checked_url": data.url}

@app.post("/parse-resume")
async def parse_resume(file: UploadFile = None):
    """Parse resume from uploaded PDF file or use sample"""
    # Initialize the workflow
    workflow = create_resume_parsing_workflow()

    input_data = {}
    temp_file_path = None

    try:
        if file:
            # Save uploaded file to temporary location
            # Ensure the suffix matches the file type if needed, or handle different types
            suffix = Path(file.filename).suffix if file.filename else '.tmp'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_file_path = Path(tmp.name)
                input_data = {"resume_path": temp_file_path}
        else:
            # Use sample resume path directly if no file uploaded
            input_data = {"resume_path": None} # Or pass SAMPLE_RESUME_PATH if preferred

        # Run workflow asynchronously
        result = await workflow.ainvoke(input_data) # Use ainvoke for async workflow

    finally:
        # Clean up temp file if it was created
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()

    # Extract results safely using .get()
    parsed_data = result.get("parsed_data", {})
    validation_result = result.get("validation_result", {})
    error = result.get("error")

    if error:
        # Handle potential errors during the workflow
        # You might want to return a different status code or error message
        return {"error": error, "parsed_data": parsed_data, "validation_result": validation_result}

    return {
        "parsed_data": parsed_data,
        "validation_result": validation_result,
        # "enriched_data": result.get("enriched_data", {}) # Add back if you have an enrichment step
    }

# --- Endpoint to upload resume and parse it ---
@app.post("/upload-resume")
async def upload_resume(user_id: str = Form(...), file: UploadFile = File(...)):
    workflow = create_resume_parsing_workflow()
    input_data = {}
    temp_file_path = None
    try:
        # Save uploaded file to temp location
        suffix = Path(file.filename).suffix if file.filename else '.tmp'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file_path = Path(tmp.name)
            input_data = {"resume_path": temp_file_path}
        # Run workflow
        result = await workflow.ainvoke(input_data)
        parsed = result.get("parsed_data", {})
        doc = {
            "user_id": user_id,
            "type": "resume",
            "raw_input": {"filename": file.filename},
            "parsed_data": parsed,
        }
        supabase.table("documents").upsert(doc, on_conflict=["user_id", "type"]).execute()
        return {"status": "ok", "parsed_data": parsed}
    except Exception as e:
        import traceback
        print(f"Unhandled exception in /upload-resume endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") 
    finally:
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()
   

# --- New Endpoint for GitHub Parsing ---
@app.post("/parse-github")
async def parse_github(request: GitHubParseRequest):
    """Parse GitHub profile using the dedicated GitHub workflow."""
    workflow = create_github_parsing_workflow()
    input_data = {"github_username": request.github_username}

    try:
        print(f"Invoking GitHub workflow with input: {input_data}")
        result = await workflow.ainvoke(input_data)
        print(f"GitHub workflow result: {result}")

        error = result.get("error")
        if error:
            # Determine appropriate status code based on error type if possible
            status_code = 404 if "not found" in error.lower() else 500
            raise HTTPException(status_code=status_code, detail={"error": error, "parsed_github_data": result.get("parsed_github_data")})

        return {
            "parsed_github_data": result.get("parsed_github_data")
        }
    except Exception as e:
        import traceback
        print(f"Unhandled exception in /parse-github endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# --- Endpoint to submit GitHub username, parse, and store in Supabase ---
@app.post("/submit-github")
async def submit_github(user_id: str = Form(...), github_username: str = Form(...)):
    workflow = create_github_parsing_workflow()
    input_data = {"github_username": github_username}
    try:
        print(f"Invoking GitHub workflow with input: {input_data}")
        result = await workflow.ainvoke(input_data)
        print(f"GitHub workflow result: {result}")

        error = result.get("error")
        parsed = result.get("parsed_github_data", {})
        if error:
            status_code = 404 if "not found" in error.lower() else 500
            raise HTTPException(status_code=status_code, detail={"error": error, "parsed_github_data": parsed})

        # Store in Supabase
        doc = {
            "user_id": user_id,
            "type": "github",
            "raw_input": {"github_username": github_username},
            "parsed_data": parsed,
        }
        supabase.table("documents").upsert(doc, on_conflict=["user_id", "type"]).execute()

        return {"status": "ok", "parsed_data": parsed}
    except Exception as e:
        import traceback
        print(f"Unhandled exception in /submit-github endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# --- Endpoint to generate cover letter using unified workflow ---
@app.post("/generate-cover-letter")
async def generate_cover_letter(input: CoverLetterInput):
    """Generate a cover letter using the unified workflow."""

    # Fetch parsed resume and github from Supabase
    try:
        resume_doc = supabase.table("documents").select("*").eq("user_id", input.user_id).eq("type", "resume").single().execute()
        github_doc = supabase.table("documents").select("*").eq("user_id", input.user_id).eq("type", "github").single().execute()
    except Exception as e:
        import traceback
        print(f"Supabase fetch error in /generate-cover-letter: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to fetch user documents from Supabase.")

    parsed_resume = resume_doc.data["parsed_data"] if resume_doc.data else None
    parsed_github = github_doc.data["parsed_data"] if github_doc.data else None

    if not parsed_resume or not parsed_github:
        missing = []
        if not parsed_resume:
            missing.append("resume")
        if not parsed_github:
            missing.append("GitHub")
        raise HTTPException(
            status_code=400,
            detail=f"No parsed {', '.join(missing)} info found for user. Please upload your resume and/or GitHub profile first."
        )

    # Prepare initial state for the workflow
    initial_state = {
        "resume_data": parsed_resume,
        "github_data": parsed_github,
        "context": {
            "job_title": input.job_title,
            "hiring_company": input.hiring_company,
            "job_description": input.job_description,
            "preferred_qualifications": input.preferred_qualifications,
            "company_culture_notes": input.company_culture_notes,
            "resume_highlights": "",
            "github_username": input.github_username or "",
            "applicant_name": input.applicant_name,
            "desired_tone": input.desired_tone,
        }
    }

    # Build and run the workflow
    try:
        graph = build_graph()
        result = await graph.ainvoke(initial_state)
        cover_letter = result["context"]["cover_letter"]
    except Exception as e:
        import traceback
        print(f"Error in cover letter workflow: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Error generating cover letter.")

    if isinstance(cover_letter, str):
        return {
            "cover_letter": cover_letter,
            "summary": None,
            "used_highlights": None,
            "used_github_info": None,
        }
    else:
        return {
            "cover_letter": cover_letter.get("cover_letter"),
            "summary": cover_letter.get("summary"),
            "used_highlights": cover_letter.get("used_highlights"),
            "used_github_info": cover_letter.get("used_github_info"),
        }

@app.get("/")
async def root():
    """Root endpoint for API status check"""
    return {"status": "online", "service": "Job URL Detector API"}