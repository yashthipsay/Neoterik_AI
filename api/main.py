from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, HttpUrl
import re
from pathlib import Path
import shutil
import tempfile
from fastapi.middleware.cors import CORSMiddleware
from agents.resume_parsing.workflow import create_resume_parsing_workflow
from agents.resume_parsing.agent import SAMPLE_RESUME_PATH

app = FastAPI(
    title="Job URL Detector API",
    description="API to detect job application URLs using regex pattern matching"
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

def is_job_application_url(url: str) -> bool:
    """Check if URL matches any known job board patterns"""
    return any(re.search(pattern, url) for pattern in KNOWN_JOB_BOARD_PATTERNS)

@app.post("/check-url")
async def check_url(data: URLCheckRequest):
    """Check if the provided URL is a job application page"""
    is_job_url = is_job_application_url(data.url)
    print(f"Checking URL: {data.url} - Result: {'✓' if is_job_url else '✗'}")
    return {"is_job_application": is_job_url, "checked_url": data.url}

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

@app.get("/")
async def root():
    """Root endpoint for API status check"""
    return {"status": "online", "service": "Job URL Detector API"}