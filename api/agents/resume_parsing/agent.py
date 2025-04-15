from pydantic_ai import Agent, RunContext
from pathlib import Path
from PyPDF2 import PdfReader
import docx2txt
import json
import spacy
from spacy.matcher import PhraseMatcher
from .models import ResumeData, Education, Experience, Skill
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.deepseek import DeepSeekProvider
from dotenv import load_dotenv
load_dotenv()

# Define base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SAMPLE_RESUME_PATH = BASE_DIR / "sample_resume" / "Resume-Yash-Thipsay.pdf"

model = OpenAIModel(
    model_name='deepseek-chat',  # or whichever DeepSeek model variant you need
    provider=DeepSeekProvider(api_key='sk-2a45174c07cd486f81d82cbf49f8bae2')
)

# Initialize the agent
resume_agent = Agent(
    model="google-vertex:gemini-exp-1206",
    deps_type=str,
    system_prompt=(
        "You are a resume parsing expert. Your role is to extract and structure "
        "information from resumes, ensuring all key details are captured accurately. "
        "Extract the following information from the resume: name, email, phone number, "
        "skills, education (degree, institution, year), experience (title, company, duration, "
        "description), and a brief summary. Return your response as a structured JSON."
    ),
)

# Initialize spaCy model (load once ideally, or share across calls)
nlp = spacy.load("en_core_web_sm")

def extract_text_from_file(file_path):
    """Extract text from PDF or DOCX file"""
    file_path = str(file_path)
    if file_path.endswith('.pdf'):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file_path.endswith('.docx'):
        return docx2txt.process(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

@resume_agent.tool_plain
async def parse_resume_from_pdf(resume_path: Path = None) -> dict:
    """
    Parse resume from a PDF or DOCX file.
    If no path is provided, uses a sample resume.
    This version uses spaCy to enhance extraction.
    """
    from tempfile import NamedTemporaryFile
    from pydantic import ValidationError

    # Use sample resume if no path provided
    if resume_path is None:
        # Assume SAMPLE_RESUME_PATH is defined globally
        from .agent import SAMPLE_RESUME_PATH
        resume_path = SAMPLE_RESUME_PATH

    # Extract raw text from the file
    resume_text = extract_text_from_file(resume_path)

    # Process the text with spaCy for enhanced extraction
    doc = nlp(resume_text)
    
    # Example: Use spaCy NER to extract the first PERSON entity as the candidate's name
    name = ""
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break

    # Example: Use PhraseMatcher to extract skills from the resume
    skill_list = ["Python", "Django", "React", "Blockchain", "Solidity", "SQL"]
    matcher = PhraseMatcher(nlp.vocab)
    patterns = [nlp.make_doc(skill) for skill in skill_list]
    matcher.add("SKILLS", patterns)
    matched_skills = set()
    matches = matcher(doc)
    for match_id, start, end in matches:
        matched_skills.add(doc[start:end].text)

    # Now, create a prompt that includes the resume text (or part of it)
    # This prompt will be sent to your LLM (DeepSeek in your case) for further parsing,
    # while you already have some structured data from spaCy.
    prompt = f"""
    Parse the following resume text and extract structured information.
    Use the pre-extracted candidate name and skills if possible.
    If any field is ambiguous, use null or an empty list.
    
    Candidate Name (from spaCy NER): {name}
    Extracted Skills (from spaCy Matcher): {list(matched_skills)}
    
    Resume Text (first 4000 characters):
    ---
    {resume_text[:3000]}
    ---
    
    Return a JSON object matching this schema:
    {{
        "name": "Full Name or null",
        "email": "email@example.com or null",
        "phone": "phone number or null",
        "skills": ["skill1", "skill2", ...],
        "education": [
            {{
                "degree": "Degree Name or null",
                "institution": "Institution Name or null",
                "year": "Graduation Year/Range or null"
            }}
        ],
        "experience": [
            {{
                "title": "Job Title or null",
                "company": "Company Name or null",
                "duration": "Duration or null",
                "description": ["responsibility1", "responsibility2", ...]
            }}
        ],
        "summary": "Brief professional summary or null"
    }}
    Respond ONLY with a valid JSON object matching the structure below. 
    Do NOT include any other text, explanation, markdown formatting, or comments.
    """

    # Call the LLM agent tool to further process the resume text (using DeepSeek)
    agent_result = await resume_agent.run(prompt)
    
    if not isinstance(agent_result.data, str):
        print(f"Error: LLM returned {type(agent_result.data)} - {agent_result.data}")
        return {"error": f"LLM did not return a string, got type {type(agent_result.data)}"}
    
    parsed_data_json_string = agent_result.data
    
    try:
        parsed_data = json.loads(parsed_data_json_string)
        
        print("--- RAW LLM OUTPUT START ---")
        print(agent_result.data)
        print("--- RAW LLM OUTPUT END ---")

        # Merge spaCy extracted fields (if present) with LLM output
        # For instance, ensure that the candidate name and skills include what spaCy found
        if not parsed_data.get("name"):
            parsed_data["name"] = name
        if parsed_data.get("skills") is None or len(parsed_data.get("skills")) == 0:
            parsed_data["skills"] = list(matched_skills)

        structured_data = {
            "name": parsed_data.get("name", ""),
            "email": parsed_data.get("email", ""),
            "phone": parsed_data.get("phone", ""),
            "skills": [Skill(category="Technical", skills=parsed_data.get("skills", []))],
            "education": [
                Education(
                    degree=edu.get("degree", ""),
                    institution=edu.get("institution", ""),
                    year=edu.get("year", "")
                ) for edu in parsed_data.get("education", [])
            ],
            "experience": [
                Experience(
                    title=exp.get("title", ""),
                    company=exp.get("company", ""),
                    duration=exp.get("duration", ""),
                    description=exp.get("description", []) if isinstance(exp.get("description"), list) else
                                exp.get("description", "").split("\n") if exp.get("description") else []
                ) for exp in parsed_data.get("experience", [])
            ],
            "summary": parsed_data.get("summary", "")
        }
        return ResumeData(**structured_data).model_dump()
    except json.JSONDecodeError:
        return ResumeData(
            name="", email="", phone="",
            skills=[Skill(category="Technical", skills=[])],
            education=[], experience=[], summary="Error parsing resume"
        ).model_dump()

@resume_agent.tool_plain
def validate_parsed_data(parsed_data: dict) -> dict: # Remove ctx, change return type hint
    """
    Validate the parsed resume data and suggest improvements
    """
    validation_result = {
        "is_valid": True,
        "suggestions": [],
        "confidence_score": 0.0 # You might want to calculate this based on checks
    }

    # Add validation logic here
    # Example validation checks:
    if not parsed_data.get("name"):
        validation_result["suggestions"].append("Name is missing.")
        validation_result["is_valid"] = False

    if not parsed_data.get("email") and not parsed_data.get("phone"):
        validation_result["suggestions"].append("Contact information (email/phone) is missing.")
        validation_result["is_valid"] = False

    if not parsed_data.get("skills"):
        validation_result["suggestions"].append("No skills detected.")
        # Decide if this makes it invalid, maybe just a suggestion
        # validation_result["is_valid"] = False

    if not parsed_data.get("experience"):
        validation_result["suggestions"].append("No work experience detected.")
        # Decide if this makes it invalid
        # validation_result["is_valid"] = False

    # Example confidence calculation (very basic)
    num_fields = 6 # name, email, phone, skills, education, experience, summary
    filled_fields = sum(1 for field in ["name", "email", "phone", "skills", "education", "experience", "summary"] if parsed_data.get(field))
    validation_result["confidence_score"] = round(filled_fields / num_fields, 2) if num_fields > 0 else 0.0


    return validation_result # Return the dictionary