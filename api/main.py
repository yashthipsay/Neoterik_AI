from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
import re
from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/")
async def root():
    """Root endpoint for API status check"""
    return {"status": "online", "service": "Job URL Detector API"}