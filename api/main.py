from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, HttpUrl
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
    
@app.post("/generate-cover-letter")
async def generate_cover_letter(input: CoverLetterInput):
    """Generate a cover letter using the unified workflow."""

    # Prepare initial state for the workflow
    initial_state = {
        "resume_path": str(SAMPLE_RESUME_PATH),  # Or set to a path if you want to use a sample resume
        "github_username": input.github_username or "",
        "context": {
            "job_title": input.job_title,
            "hiring_company": input.hiring_company,
            "job_description": input.job_description,
            "preferred_qualifications": input.preferred_qualifications,
            "company_culture_notes": input.company_culture_notes,
            "resume_highlights": "",  # You can add this to your input/model if needed
            "github_username": input.github_username or "",
            "applicant_name": input.applicant_name,
            "desired_tone": input.desired_tone,   
      }
    }

    # Build and run the workflow
    graph = build_graph()
    result = await graph.ainvoke(initial_state)
    cover_letter = result["context"]["cover_letter"]

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