from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import BaseModel, HttpUrl
from supabase_client import supabase
from models_supabase import UserIn, DocumentIn
import re
from pathlib import Path
import shutil
import tempfile
from fastapi.middleware.cors import CORSMiddleware
from agents.resume_parsing.agent import resume_agent
from agents.repo_parsing.agent import github_agent 
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
        # Check if user already exists
        existing = supabase.table("users").select("*").eq("id", user.id).single().execute()
        if existing.data:
            print(f"User already exists: {user.id}")
            return {"success": True, "user": existing.data, "message": "User already registered"}

        # Insert new user
        data = {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "avatar_url": user.avatar_url,
            "github_username": user.github_username,
        }
        print("Registering new user with data:", data)
        result = supabase.table("users").insert(data).execute()
        print("Supabase insert result:", result)
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

# Add this test endpoint after the existing test-supabase endpoint
@app.get("/test-documents-table")
def test_documents_table():
    """Test endpoint to check the documents table structure"""
    try:
        # Try to get the table structure
        result = supabase.table("documents").select("*").limit(1).execute()
        return {"status": "success", "data": result.data, "columns": list(result.data[0].keys()) if result.data else []}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/test-db-connection")
def test_db_connection():
    """Test endpoint to check database connection and table existence"""
    try:
        # Test basic connection
        users_result = supabase.table("users").select("*").limit(1).execute()
        
        # Test documents table
        try:
            docs_result = supabase.table("documents").select("*").limit(1).execute()
            docs_status = "exists"
        except Exception as docs_error:
            docs_status = f"error: {str(docs_error)}"
        
        return {
            "status": "connected",
            "users_table": "exists" if users_result.data is not None else "error",
            "documents_table": docs_status,
            "supabase_url": SUPABASE_URL if SUPABASE_URL else "not_set",
            "supabase_key": "set" if SUPABASE_KEY else "not_set"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

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

# @app.post("/parse-resume")
# async def parse_resume(file: UploadFile = None):
#     """Parse resume from uploaded PDF file or use sample"""
#     # Initialize the workflow
#     workflow = create_resume_parsing_workflow()

#     input_data = {}
#     temp_file_path = None

#     try:
#         if file:
#             # Save uploaded file to temporary location
#             # Ensure the suffix matches the file type if needed, or handle different types
#             suffix = Path(file.filename).suffix if file.filename else '.tmp'
#             with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#                 shutil.copyfileobj(file.file, tmp)
#                 temp_file_path = Path(tmp.name)
#                 input_data = {"resume_path": temp_file_path}
#         else:
#             # Use sample resume path directly if no file uploaded
#             input_data = {"resume_path": None} # Or pass SAMPLE_RESUME_PATH if preferred

#         # Run workflow asynchronously
#         result = await workflow.ainvoke(input_data) # Use ainvoke for async workflow

#     finally:
#         # Clean up temp file if it was created
#         if temp_file_path and temp_file_path.exists():
#             temp_file_path.unlink()

#     # Extract results safely using .get()
#     parsed_data = result.get("parsed_data", {})
#     validation_result = result.get("validation_result", {})
#     error = result.get("error")

#     if error:
#         # Handle potential errors during the workflow
#         # You might want to return a different status code or error message
#         return {"error": error, "parsed_data": parsed_data, "validation_result": validation_result}

#     return {
#         "parsed_data": parsed_data,
#         "validation_result": validation_result,
#         # "enriched_data": result.get("enriched_data", {}) # Add back if you have an enrichment step
#     }

# --- Endpoint to upload resume and parse it ---
@app.post("/upload-resume")
async def upload_resume(user_id: str = Form(...), file: UploadFile = File(...)):
    print(f"[API] Received upload-resume for user_id={user_id}, filename={file.filename}")
    temp_file_path = None
    try:
        # Validate inputs
        if not user_id or not file:
            raise HTTPException(status_code=400, detail="user_id and file are required")
        
        # Save uploaded file to a temporary location
        suffix = Path(file.filename).suffix if file.filename else '.tmp'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file_path = Path(tmp.name)
        print(f"[API] Saved resume to temp path: {temp_file_path}")

        # Parse resume using the agent (pass the temp file path directly)
        from agents.resume_parsing.agent import parse_resume_from_pdf
        parsed_data = await parse_resume_from_pdf(temp_file_path)
        print(f"[API] Parsed resume data for user_id={user_id}: {parsed_data}")
        
        # Validate parsed data
        if not parsed_data or isinstance(parsed_data, dict) and parsed_data.get("error"):
            raise HTTPException(status_code=400, detail=f"Failed to parse resume: {parsed_data.get('error') if isinstance(parsed_data, dict) else 'Unknown error'}")
        
        # Store in Supabase - Fixed upsert operation
        doc = {
            "user_id": user_id,
            "type": "resume",
            "raw_input": {"filename": file.filename, "resume_path": str(temp_file_path)},
            "parsed_data": parsed_data,
        }
        
        # First try to delete existing record for this user and type
        try:
            supabase.table("documents").delete().eq("user_id", user_id).eq("type", "resume").execute()
            print(f"[API] Deleted existing resume record for user_id={user_id}")
        except Exception as delete_error:
            print(f"[API] Warning: Could not delete existing record: {delete_error}")
        
        # Then insert the new record
        try:
            supabase.table("documents").insert(doc).execute()
            print(f"[API] Successfully inserted resume data for user_id={user_id}")
        except Exception as insert_error:
            print(f"[API] Database insert error: {insert_error}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(insert_error)}")
        
        return {"status": "ok", "parsed_data": parsed_data}
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        import traceback
        print(f"Unhandled exception in /upload-resume endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Clean up temp file if it was created
        if temp_file_path and Path(temp_file_path).exists():
            Path(temp_file_path).unlink()
   
# --- New Endpoint for GitHub Parsing ---
# @app.post("/parse-github")
# async def parse_github(request: GitHubParseRequest):
#     """Parse GitHub profile using the dedicated GitHub workflow."""
#     workflow = create_github_parsing_workflow()
#     input_data = {"github_username": request.github_username}

#     try:
#         print(f"Invoking GitHub workflow with input: {input_data}")
#         result = await workflow.ainvoke(input_data)
#         print(f"GitHub workflow result: {result}")

#         error = result.get("error")
#         if error:
#             # Determine appropriate status code based on error type if possible
#             status_code = 404 if "not found" in error.lower() else 500
#             raise HTTPException(status_code=status_code, detail={"error": error, "parsed_github_data": result.get("parsed_github_data")})

#         return {
#             "parsed_github_data": result.get("parsed_github_data")
#         }
#     except Exception as e:
#         import traceback
#         print(f"Unhandled exception in /parse-github endpoint: {e}")
#         print(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# --- Endpoint to submit GitHub username, parse, and store in Supabase ---
@app.post("/submit-github")
async def submit_github(user_id: str = Form(...), github_username: str = Form(...)):
    print(f"[API] Received submit-github for user_id={user_id}, github_username={github_username}")
    try:
        # Validate inputs
        if not user_id or not github_username:
            raise HTTPException(status_code=400, detail="user_id and github_username are required")
        
        # Parse GitHub using the agent
        github_result = await github_agent.run(github_username)
        parsed_data = github_result.data if hasattr(github_result, "data") else github_result
        print(f"[API] Parsed GitHub data for user_id={user_id}: {parsed_data}")
        
        # Validate parsed data
        if not parsed_data or isinstance(parsed_data, dict) and parsed_data.get("error"):
            raise HTTPException(status_code=400, detail=f"Failed to parse GitHub profile: {parsed_data.get('error') if isinstance(parsed_data, dict) else 'Unknown error'}")
        
        # Store in Supabase - Fixed upsert operation
        doc = {
            "user_id": user_id,
            "type": "github",
            "raw_input": {"github_username": github_username},
            "parsed_data": parsed_data,
        }
        
        # First try to delete existing record for this user and type
        try:
            supabase.table("documents").delete().eq("user_id", user_id).eq("type", "github").execute()
            print(f"[API] Deleted existing GitHub record for user_id={user_id}")
        except Exception as delete_error:
            print(f"[API] Warning: Could not delete existing record: {delete_error}")
        
        # Then insert the new record
        try:
            supabase.table("documents").insert(doc).execute()
            print(f"[API] Successfully inserted GitHub data for user_id={user_id}")
        except Exception as insert_error:
            print(f"[API] Database insert error: {insert_error}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(insert_error)}")
        
        return {"status": "ok", "parsed_data": parsed_data}
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        import traceback
        print(f"Unhandled exception in /submit-github endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# --- Endpoint to generate cover letter using unified workflow ---
@app.post("/generate-cover-letter")
async def generate_cover_letter(input: CoverLetterInput):
    """Generate a cover letter using the unified workflow."""

    # Fetch user id to pass in unified workflow to get parsed resume and GitHub data
    # try:
    #     supabase.table("users").select("*").eq("user_id", input.user_id).single().execute()
    #     print(f"[API] Received generate-cover-letter for user_id={input.user_id}")
    # except Exception as e:
    #     import traceback
    #     print(f"Supabase fetch error in /generate-cover-letter: {e}")
    #     print(traceback.format_exc())
    #     raise HTTPException(status_code=500, detail="Failed to fetch user from Supabase.")

     # Prepare initial state for the workflow
    initial_state = {
        "user_id": input.user_id,
        # "github_username":  input.github_username,
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
        print(f"[API] Starting workflow for user_id={input.user_id}")
        result = await graph.ainvoke(initial_state)
        cover_letter = result["context"]["cover_letter"]
        print(f"[API] Generated cover letter for user_id={input.user_id}")
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
