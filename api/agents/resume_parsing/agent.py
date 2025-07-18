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
SAMPLE_RESUME_PATH = BASE_DIR / "sample_resume" / "Parag_cv.pdf"

# model = OpenAIModel(
#     model_name='deepseek-chat',  # or whichever DeepSeek model variant you need
#     provider=DeepSeekProvider(api_key='sk-2a45174c07cd486f81d82cbf49f8bae2')
# )
#groq:deepseek-r1-distill-llama-70b

# Initialize the agent
resume_agent = Agent(
    model="gemini-1.5-flash",
    deps_type=str,
    system_prompt=(
        "You are a resume parsing expert. Extract structured information from resumes based on the provided format. "
        "Focus on capturing name, education, experience, skills, projects, and achievements accurately. "
        "Return the response as a structured JSON matching the schema below."
    ),
)

# Initialize spaCy model (load once ideally, or share across calls)
# nlp = spacy.load("en_core_web_sm")

def extract_text_from_file(file_path):
    """
    Use LangChain document loaders to extract text from PDF or DOCX.
    """
    print(f"Extracting text from file: {file_path}")
    file_str = str(file_path)
    if file_str.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_str)
    elif file_str.lower().endswith(".docx"):
        loader = Docx2txtLoader(file_str)
    else:
        raise ValueError(f"Unsupported file format: {file_str}")

    docs = loader.load()
    # concatenate all pages
    print(f"Loaded {len(docs)} document(s) from {file_path}")
    return "\n".join(doc.page_content for doc in docs)

@resume_agent.tool_plain
async def parse_resume_from_pdf(resume_path: Path = None) -> dict:
    """
    Parse resume from a PDF or DOCX file.
    If no path is provided, uses a sample resume.
    This version uses LangChain document loaders.
    """
    print(f"Starting resume parsing for path: {resume_path}")
    # Handle the case where no path is provided, use the sample resume
    if resume_path is None:
        resume_path = SAMPLE_RESUME_PATH
        print(f"No resume path provided, using sample: {resume_path}")

    # Ensure resume_path is not None before proceeding
    if not resume_path:
         return {"error": "Resume path is missing and sample path could not be determined."}

    try:
        resume_text = extract_text_from_file(resume_path)
        print(f"Extracted text (first 100 chars): {resume_text[:100]}...")
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
    Focus on the following fields: name, education, experience, skills, projects, achievements.
    If any field is ambiguous, use null or an empty list.

    Resume Text (first 4000 characters):
    ---
    {resume_text[:4000]}
    ---

    Return a JSON object matching this schema:
    {{
        "name": "Full Name or null",
        "education": [
            {{
                "program": "Program Name or null",
                "institution": "Institution Name or null",
                "grade": "Grade/CGPA or null",
                "year": "Year/Range or null"
            }}
        ],
        "experience": [
            {{
                "title": "Job Title or null",
                "company": "Company Name or null",
                "duration": "Duration or null",
                "responsibilities": ["responsibility1", "responsibility2", ...],
                "tools": ["tool1", "tool2", ...]
            }}
        ],
        "skills": [
            {{
                "category": "Category Name",
                "skills": ["skill1", "skill2", ...]
            }}
        ],
        "projects": [
            {{
                "name": "Project Name or null",
                "description": ["description1", "description2", ...],
                "tech_stack": ["tech1", "tech2", ...]
            }}
        ],
        "achievements": ["achievement1", "achievement2", ...]
    }}
    Respond ONLY with a valid JSON object.
    """

    # Call the LLM agent tool to further process the resume text (using DeepSeek)
    try:
        agent_result = await resume_agent.run(prompt)
        print(f"LLM response received, type: {type(agent_result.data)}")
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
            
        parsed_data = json.loads(cleaned_json_string)

        # print("--- RAW LLM OUTPUT START ---")
        # print(agent_result.data) # Still print the original raw output for debugging
        # print("--- RAW LLM OUTPUT END ---")
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
            "education": [
                Education(
                    program=edu.get("program", ""),
                    institution=edu.get("institution", ""),
                    grade=edu.get("grade", ""),
                    year=edu.get("year", "")
                ) for edu in parsed_data.get("education", [])
            ],
            "experience": [
                Experience(
                    title=exp.get("title", ""),
                    company=exp.get("company", ""),
                    duration=exp.get("duration", ""),
                    responsibilities=exp.get("responsibilities", []) if isinstance(exp.get("responsibilities"), list) else
                                    exp.get("responsibilities", "").split("\n") if exp.get("responsibilities") else [],
                    tools=exp.get("tools", [])
                ) for exp in parsed_data.get("experience", [])
            ],
            "skills": [
                Skill(
                    category=skill.get("category", "Technical"),
                    skills=skill.get("skills", [])
                ) for skill in parsed_data.get("skills", [])
            ],
            "projects": parsed_data.get("projects", []),
            "achievements": parsed_data.get("achievements", [])
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
        "confidence_score": 0.0
    }

    if not parsed_data.get("name"):
        validation_result["suggestions"].append("Name is missing.")
        validation_result["is_valid"] = False

    if not parsed_data.get("education"):
        validation_result["suggestions"].append("Education section is missing.")
        validation_result["is_valid"] = False

    if not parsed_data.get("experience"):
        validation_result["suggestions"].append("Experience section is missing.")

    if not parsed_data.get("skills"):
        validation_result["suggestions"].append("Skills section is missing.")

    if not parsed_data.get("projects"):
        validation_result["suggestions"].append("Projects section is missing.")

    num_fields = 5  # name, education, experience, skills, projects
    filled_fields = sum(1 for field in ["name", "education", "experience", "skills", "projects"] if parsed_data.get(field))
    validation_result["confidence_score"] = round(filled_fields / num_fields, 2) if num_fields > 0 else 0.0

    return validation_result