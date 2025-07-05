from pydantic_ai import Agent, RunContext
import json
from .models import ResumeData, Education, Experience, Skill
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.deepseek import DeepSeekProvider
from dotenv import load_dotenv
from pathlib import Path
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
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
    model="gemini-1.5-pro-latest",
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
# nlp = spacy.load("en_core_web_sm")

def extract_text_from_file(file_path):
    """
    Use LangChain document loaders to extract text from PDF or DOCX.
    """
    file_str = str(file_path)
    if file_str.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_str)
    elif file_str.lower().endswith(".docx"):
        loader = Docx2txtLoader(file_str)
    else:
        raise ValueError(f"Unsupported file format: {file_str}")

    docs = loader.load()
    # concatenate all pages
    return "\n".join(doc.page_content for doc in docs)

@resume_agent.tool_plain
async def parse_resume_from_pdf(resume_path: Path = None) -> dict:
    """
    Parse resume from a PDF or DOCX file.
    If no path is provided, uses a sample resume.
    This version uses LangChain document loaders.
    """
    # Handle the case where no path is provided, use the sample resume
    if resume_path is None:
        resume_path = SAMPLE_RESUME_PATH
        print(f"No resume path provided, using sample: {resume_path}")

    # Ensure resume_path is not None before proceeding
    if not resume_path:
         return {"error": "Resume path is missing and sample path could not be determined."}

    try:
        resume_text = extract_text_from_file(resume_path)
    except ValueError as e:
        # Catch the specific error from extract_text_from_file if format is still wrong
        return {"error": str(e)}
    except Exception as e:
        # Catch other potential errors during text extraction
        return {"error": f"Failed to extract text from {resume_path}: {str(e)}"}

    # You can still pre‑seed name/skills via simple regex or skip
    name = None
    matched_skills = []

    # Now, create a prompt that includes the resume text (or part of it)
    # This prompt will be sent to your LLM (DeepSeek in your case) for further parsing,
    # while you already have some structured data from spaCy.
    prompt = f"""
    Parse the following resume text and extract structured information.
    Use the pre-extracted candidate name and skills if possible.
    If any field is ambiguous, use null or an empty list.

    Candidate Name (from SpaCy NER): {name}
    Extracted Skills (from SpaCy Matcher): {list(matched_skills)}

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
    try:
        agent_result = await resume_agent.run(prompt)
    except Exception as e:
        return {"error": f"LLM agent run failed: {str(e)}"}

    if not isinstance(agent_result.data, str):
        print(f"Error: LLM returned {type(agent_result.data)} - {agent_result.data}")
        return {"error": f"LLM did not return a string, got type {type(agent_result.data)}"}

    parsed_data_json_string = agent_result.data

    try:
        
                # Clean the string: remove markdown code fences and strip whitespace
        cleaned_json_string = parsed_data_json_string.strip()
        if cleaned_json_string.startswith("```json"):
            cleaned_json_string = cleaned_json_string[len("```json"):].strip()
        if cleaned_json_string.endswith("```"):
            cleaned_json_string = cleaned_json_string[:-len("```")].strip()
            
        parsed_data = json.loads(parsed_data_json_string)

        print("--- RAW LLM OUTPUT START ---")
        print(agent_result.data) # Still print the original raw output for debugging
        print("--- RAW LLM OUTPUT END ---")
        print("--- CLEANED JSON STRING START ---")
        print(cleaned_json_string)
        print("--- CLEANED JSON STRING END ---")

        # Merge any fallback name/skills if LLM left them null/empty
        if not parsed_data.get("name") and name:
            parsed_data["name"] = name
        if (not parsed_data.get("skills")) and matched_skills:
            parsed_data["skills"] = matched_skills

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
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        # Print the cleaned string that failed to parse
        print(f"Invalid JSON string received (after cleaning): {cleaned_json_string}")
        return ResumeData(
            name="", email="", phone="",
            skills=[Skill(category="Technical", skills=[])],
            education=[], experience=[], summary="Error parsing LLM JSON response"
        ).model_dump()
    except Exception as e:
        # Catch potential errors during Pydantic model validation
        print(f"Error creating ResumeData model: {e}")
        return ResumeData(
            name="", email="", phone="",
            skills=[Skill(category="Technical", skills=[])],
            education=[], experience=[], summary=f"Error processing parsed data: {str(e)}"
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