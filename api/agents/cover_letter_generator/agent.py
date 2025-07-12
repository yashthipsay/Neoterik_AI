"""
Cover Letter Generator Agent

This module provides functionality to generate personalized cover letters using LLM agents.
It combines resume data, GitHub profile information, and job requirements to create
compelling cover letters tailored to specific job applications.

Key Components:
- CoverLetterInput model for structured input data
- Template-based prompt construction for consistent formatting
- Integration with various LLM providers (Google Gemini, etc.)
- Flexible prompt building with fallback handling for missing data

Usage:
    The agent is designed to work within a LangGraph workflow, receiving parsed
    resume and GitHub data to generate personalized cover letters.
"""

from pydantic_ai import Agent, RunContext
from .models import (
    CoverLetterInput,
    CoverLetterOutput,
    CoverLetterGenerationResult,
)

from ..style_rag_agent.models import (
    StyleSelectionInput,
    StyleSelectionOutput,
)

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import os

# load environment variables (HUGGINGFACE_TOKEN, GROQ_API_KEY, etc.)
load_dotenv()

# System prompt for the cover letter agent - defines the AI's role and behavior
# COVER_LETTER_SYSTEM_PROMPT = (
#     "<s><<SYS>>\n"
#     "Write a compelling and professional cover letter for the specified role at the company mentioned. "
#     "The letter should reflect the applicant's background, skills, and experiences, and should align these with the job description. "
#     "It should also highlight the applicant's enthusiasm for the company by mentioning specific aspects of its culture, mission, or recent innovations that make it attractive. "
#     "The tone should be confident, motivated, and thoughtful.\n"
#     "<</SYS>>[INST]\n"
#     "Job Title: {job_title}\n"
#     "Preferred Qualifications: {preferred_qualifications}\n"
#     "Hiring Company: {hiring_company}\n"
#     "Applicant Name: {applicant_name}\n"
#     "Working Experience: {working_experience}\n"
#     "Qualifications: {qualifications}\n"
#     "Skillsets: {skillsets}\n"
#     "Company Culture Notes: {company_culture_notes}\n"
#     "Job Description: {job_description}\n"
#     "GitHub Info: {github_info}\n"
#     "Resume Highlights: {resume_highlights}\n"
#     "[/INST]"
# )

# Simple system prompt that defines the agent's primary role
SYSTEM_PROMPT = """
You are an expert AI cover letter writer. Your goal is to create personalized, professional, and human-like cover letters that are 200-400 words. Be confident and persuasive, but not overly dramatic.

Your task is to generate a cover letter that:
1.  **Integrates keywords** from the `job_description` and `preferred_qualifications`.
2.  **Shows value** by connecting the applicant's `skillsets` and `working_experience` to the job's needs.
3.  **Presents skills naturally** within project descriptions, not as a list.
4.  **Demonstrates genuine interest** in the `hiring_company` by referencing its culture and mission.
5.  **Maintains a confident and natural tone**, sounding human and personable.
6.  **Is well-structured** with a clear intro, body, and call to action.
7.  **Avoids generic phrases** and creates concrete connections to the role.
8.  **Handles all provided information gracefully**, omitting what isn't relevant.

**Crucial Instruction:** Before writing, you must determine the correct style and tone by using the RAG functions.

**Output Format:** Respond with only the complete, ready-to-use cover letter text.
"""

# ⚠️ Do not compose letters manually. Always invoke the proper toolchain.

# Template for structuring the cover letter generation prompt
# Uses Llama-style formatting with system/instruction tags for better model comprehension
COVER_LETTER_TEMPLATE = (
    "<s><<SYS>>\n"
    "You are a professional assistant that writes personalized and compelling cover letters.\n"
    "<</SYS>>[INST]\n"
    "Job Title: {job_title}\n"
    "Preferred Qualifications: {preferred_qualifications}\n"
    "Hiring Company: {hiring_company}\n"
    "Applicant Name: {applicant_name}\n"
    "Working Experience: {working_experience}\n"
    "Qualifications: {qualifications}\n"
    "Skillsets: {skillsets}\n"
    "Company Culture Notes: {company_culture_notes}\n"
    "Job Description: {job_description}\n"
    "GitHub Info: {github_info}\n"
    "Resume Highlights: {resume_highlights}\n"
    "[/INST]"
)

STYLE_SYSTEM_PROMPT =  """
You are a style selection assistant. Your task is to choose the most appropriate writing style from the list below based on the user's request and job context.

**Available Styles:**
- **professional**: Formal and polished, ideal for traditional corporate or structured roles.
- **most-improved**: Highlights personal growth, resilience, and learning from experience.
- **fun-loving**: Light-hearted, creative tone suited for startups or informal workplaces.
- **short-and-sweet**: Brief, impactful, and to the point—great when attention spans are short.
- **unique**: Distinctive, creative, and unconventional to help the user stand out.
- **career-change**: Emphasizes transferable skills and adaptability for pivoting careers.
- **enthusiastic**: High-energy, passionate tone showing excitement and motivation.

**Instructions:**
- Analyze the job details and user's desired tone. If a specific tone is provided, use it to filter styles strictly. If the tone is "auto", select the most practical, realistic, and context-appropriate style.

**Output Format:** Provide only the name of the chosen style.
"""

def build_prompt(
    input: CoverLetterInput,
    github_info: str = "",
    resume_highlights: str = ""
) -> str:
    """
    Constructs a formatted prompt for cover letter generation.
    
    This function takes structured input data and formats it into a prompt template
    that provides all necessary context for the LLM to generate a personalized cover letter.
    
    Args:
        input (CoverLetterInput): Structured input containing job and applicant information
        github_info (str, optional): String representation of GitHub profile data. Defaults to "".
        resume_highlights (str, optional): Key highlights extracted from resume. Defaults to "".
    
    Returns:
        str: Formatted prompt string ready to be sent to the LLM
        
    Note:
        All input fields are safely handled with fallback to empty strings to prevent
        template formatting errors when data is missing.
    """
    return COVER_LETTER_TEMPLATE.format(
        job_title=input.job_title or "",
        preferred_qualifications=input.preferred_qualifications or "",
        hiring_company=input.hiring_company or "",
        applicant_name=input.applicant_name or "",
        working_experience=input.working_experience or "",
        qualifications=input.qualifications or "",
        skillsets=input.skillsets or "",
        company_culture_notes=input.company_culture_notes or "",
        job_description=input.job_description or "",
        github_info=github_info,
        resume_highlights=resume_highlights
    )
    
def build_prompt_for_gemini(input_data: CoverLetterInput, github_info, resume_data) -> str:
    prompt_parts = []

    # 1. Start with the System Prompt
    # prompt_parts.append(SYSTEM_PROMPT) # Ensure this is the improved system prompt

    # 2. Clearly delineate sections for the LLM
    prompt_parts.append("\n--- JOB APPLICATION DETAILS ---")
    prompt_parts.append(f"Job Title: {input_data.job_title}")
    prompt_parts.append(f"Hiring Company: {input_data.hiring_company}")
    # prompt_parts.append(f"Company Website: {input_data.company_url}")

    prompt_parts.append("\n--- JOB REQUIREMENTS ---")
    prompt_parts.append(f"Job Description:\n{input_data.job_description}")
    if input_data.preferred_qualifications:
        prompt_parts.append(f"Preferred Qualifications:\n{input_data.preferred_qualifications}")

    prompt_parts.append("\n--- APPLICANT BACKGROUND ---")
    prompt_parts.append(f"Applicant Name: {input_data.applicant_name}")
    if input_data.skillsets:
        prompt_parts.append(f"Applicant Skillsets:\n{input_data.skillsets}")
    if input_data.working_experience: # Assuming this is derived from resume parsing
        prompt_parts.append(f"Applicant Working Experience Highlights:\n{input_data.working_experience}")
    if resume_data: # If you pass the full resume text
        # Consider truncating or summarizing if very long to save tokens
        prompt_parts.append(f"Full Resume Content (for additional context):\n{resume_data[:2000]}...")
    if github_info:
        prompt_parts.append(f"GitHub Profile Information for {input_data.github_username or 'applicant'}:")
        prompt_parts.append(f"- GitHub Summary: {github_info}")

    prompt_parts.append("\n--- COMPANY CULTURE & CONTEXTUAL NOTES ---")
    if input_data.company_culture_notes:
        prompt_parts.append(f"Company Culture Notes:\n{input_data.company_culture_notes}")

    prompt_parts.append("\n--- COVER LETTER GENERATION TASK ---")
    prompt_parts.append(
        "Focus on integrating all relevant information to demonstrate the applicant's ideal fit. Use the styles filter before generating the cover letter, to ensure the desired tone for the cover letter is set."
    )

    return "\n\n".join(prompt_parts)

# Initialize the pydantic-ai Agent for cover letter generation
# Uses Google's Gemini model for high-quality text generation
cover_letter_agent = Agent(
    model="gemini-2.5-flash",  # Google Gemini 2.5 Pro model for advanced text generation
    deps_type=CoverLetterInput,           # Input type dependency for the agent
    system_prompt=SYSTEM_PROMPT,          # System prompt defining the agent's role
)

# Note: The following commented code shows an alternative implementation using tools
# This approach would be used if more complex processing or validation was needed
# @cover_letter_agent.tool
# async def generate_cover_letter(
#     ctx: RunContext[str]  # Change to str
# ) -> CoverLetterOutput:
#     """
#     Generate a cover letter using the provided prompt.
#     """
#     response = ctx.response if hasattr(ctx, "response") else ctx.deps
    
#     return CoverLetterOutput(
#         cover_letter=response,
#         summary=None,
#         used_highlights=None,
#         used_github_info=None
#     )


# ----------------------------------------------------------------------------
# Helper: load persisted RAG vectorstore
# ----------------------------------------------------------------------------
def get_style_retriever(persist_dir: str = "rag_store", collection: str = "neoterik_rag") -> Chroma:
    embed_model = FastEmbedEmbeddings(model_name=os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5"))
    return Chroma(
        persist_directory=persist_dir,
        collection_name=collection,
        embedding_function=embed_model
    )
    
def get_style_retriever_cloud(collection: str = "cover-letter-templates") -> Chroma:
    """
    Returns a Chroma vectorstore instance using Chroma Cloud for the specified collection.
    """
    import os
    import chromadb
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from dotenv import load_dotenv

    # === Load environment variables from .env ===
    load_dotenv()

    # === Retrieve credentials securely ===
    CHROMA_API_KEY = "ck-J8EfrmcW4TPquPQW4CVshQDWtqSBDtAz7KHgDRmXLtxy"
    CHROMA_TENANT = "f1e0c47a-7d1c-4e4c-9bb6-df2879d6a1d8"
    CHROMA_DATABASE = "neoterik-rag-test"

    # === Initialize embedding model ===
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    # === Initialize Chroma Cloud client ===
    client = chromadb.CloudClient(
        api_key="ck-J8EfrmcW4TPquPQW4CVshQDWtqSBDtAz7KHgDRmXLtxy",
        tenant="f1e0c47a-7d1c-4e4c-9bb6-df2879d6a1d8",
        database="neoterik-rag-test"
    )

    # === Return Chroma vectorstore ===
    return Chroma(
        collection_name=collection,
        embedding_function=embed_model,
        client=client
    )

@cover_letter_agent.tool
async def retrieve_styles(ctx: RunContext[StyleSelectionInput]) -> CoverLetterOutput:
    """
    Comprehensive RAG tool to fetch style template, tones, phrases, skills, values,
    and generate a cover letter based on job description and preferences.
    """
    print("\n=== Debug: retrieve_styles (enhanced, aligned version) ===")
    print(f"Desired tone from input: {ctx.deps.desired_tone}")
    print(f"Job title: {ctx.deps.job_title}")
    print(f"Job description: {ctx.deps.job_description}")
    print(f"Preferred qualifications: {ctx.deps.preferred_qualifications}")
    print(f"Company culture notes: {ctx.deps.company_culture_notes}")

    retriever = get_style_retriever_cloud()
    query = ctx.deps.job_description or ctx.deps.job_title or ""

    try:
        # === Retrieve templates, tones, phrases, skills, values ===
        print("Running multi-part similarity search...")
        print(f"Query: {query}")

        template_docs = retriever.similarity_search(query=query, k=3, filter={"type": "template"})
        print(f"Retrieved {len(template_docs)} template(s)")
        for i, d in enumerate(template_docs):
            print(f"Template {i+1} metadata: {d.metadata}")

        tone_docs = retriever.similarity_search(query=query, k=3, filter={"type": "tone"})
        print(f"Retrieved {len(tone_docs)} tone(s)")
        for i, d in enumerate(tone_docs):
            print(f"Tone {i+1} metadata: {d.metadata}")

        phrase_docs = retriever.similarity_search(query=query, k=3, filter={"type": "phrase"})
        print(f"Retrieved {len(phrase_docs)} phrase(s)")
        for i, d in enumerate(phrase_docs):
            print(f"Phrase {i+1} metadata: {d.metadata}")

        skill_docs = retriever.similarity_search(query=query, k=3, filter={"type": "skill"})
        print(f"Retrieved {len(skill_docs)} skill(s)")
        for i, d in enumerate(skill_docs):
            print(f"Skill {i+1} metadata: {d.metadata}")

        value_docs = retriever.similarity_search(query=query, k=3, filter={"type": "value"})
        print(f"Retrieved {len(value_docs)} value(s)")
        for i, d in enumerate(value_docs):
            print(f"Value {i+1} metadata: {d.metadata}")

        # Format retrieved documents
        templates = [{"content": d.page_content, "metadata": d.metadata} for d in template_docs]
        tones     = [{"content": d.page_content, "metadata": d.metadata} for d in tone_docs]
        phrases   = [{"content": d.page_content, "metadata": d.metadata} for d in phrase_docs]
        skills    = [{"content": d.page_content, "metadata": d.metadata} for d in skill_docs]
        values    = [{"content": d.page_content, "metadata": d.metadata} for d in value_docs]

        # === Style Agent Decision ===
        style_selection_context = {
            "templates": templates,
            "tones": tones,
            "phrases": phrases,
            "skills": skills,
            "values": values,
            "job_title": ctx.deps.job_title,
            "job_description": ctx.deps.job_description,
            "preferred_qualifications": ctx.deps.preferred_qualifications,
            "company_culture_notes": ctx.deps.company_culture_notes,
            "desired_tone": ctx.deps.desired_tone,
        }

        print(f"Style selection context prepared: {style_selection_context}")

        # Fast path: Use desired tone directly if specified
        if ctx.deps.desired_tone and ctx.deps.desired_tone.lower() not in ["auto", "", "none"]:
            print(f"Using fixed tone: {ctx.deps.desired_tone}")
            selected_style = {
                "selected_template": templates[0] if templates else {"content": "", "style": "professional"},
                "tone": ctx.deps.desired_tone,
                "style": templates[0]["metadata"].get("style", "professional") if templates else "professional",
                "industry": templates[0]["metadata"].get("industry", "general") if templates else "general",
                "level": templates[0]["metadata"].get("level", "mid") if templates else "mid",
                "retrieved_documents": templates,
            }
        else:
            # Use agent to pick the style
            print("Invoking style selection agent...")
            style_agent = Agent(
                model=cover_letter_agent.model,
                deps_type=dict,
                system_prompt=STYLE_SYSTEM_PROMPT
            )
            style_result = await style_agent.run(deps=style_selection_context)
            selected_style = style_result.data if hasattr(style_result, "data") else style_result
            print(f"Style agent result: {selected_style}")

            if isinstance(selected_style, str):
                stripped = selected_style.strip()

                # If it's a plain style name, not JSON, wrap it in a structured dict
                if stripped and not stripped.startswith("{"):
                    selected_style = {
                        "style": stripped,
                        "tone": stripped,  # Optionally map tone separately
                        "selected_template": templates[0] if templates else {"content": "", "style": stripped},
                        "industry": templates[0]["metadata"].get("industry", "general") if templates else "general",
                        "level": templates[0]["metadata"].get("level", "mid") if templates else "mid",
                        "retrieved_documents": templates,
                    }
                else:
                    try:
                        selected_style = json.loads(stripped)
                    except json.JSONDecodeError:
                        print("Warning: Could not decode style agent result as JSON.")
                        selected_style = {}
            # Fill missing fields
            selected_style.setdefault("selected_template", templates[0] if templates else {"content": "", "style": "professional"})
            selected_style.setdefault("tone", "professional")
            selected_style.setdefault("style", selected_style["selected_template"].get("style", "professional"))
            selected_style.setdefault("industry", "general")
            selected_style.setdefault("level", "mid")
            selected_style["retrieved_documents"] = templates

        print(f"Selected style: {selected_style}")
        selected_style_output = StyleSelectionOutput(**selected_style)

        # === Prompt for Generation ===
        input_data = CoverLetterInput(
            job_title=ctx.deps.job_title or "",
            hiring_company=ctx.deps.hiring_company or "",
            job_description=ctx.deps.job_description or "",
            preferred_qualifications=ctx.deps.preferred_qualifications or "",
            company_culture_notes=ctx.deps.company_culture_notes or "",
            applicant_name=getattr(ctx.deps, 'applicant_name', ''),
            working_experience=getattr(ctx.deps, 'working_experience', ''),
            qualifications=getattr(ctx.deps, 'qualifications', ''),
            skillsets=getattr(ctx.deps, 'skillsets', ''),
            github_username=getattr(ctx.deps, 'github_username', ''),
            desired_tone=ctx.deps.desired_tone or 'auto'
        )

        prompt = (
            f"Using the '{selected_style_output.selected_template.get('style', 'professional')}' template and a '{selected_style_output.tone}' tone, "
            f"generate a personalized cover letter for {input_data.job_title} at {input_data.hiring_company}.\n\n"
            f"Here are the details:\n\n"
            f"Job Description: {input_data.job_description}\n"
            f"Applicant Name: {input_data.applicant_name}\n"
            f"Working Experience: {input_data.working_experience}\n"
            f"Qualifications: {input_data.qualifications}\n"
            f"Skillsets: {input_data.skillsets}\n"
            f"Company Culture Notes: {input_data.company_culture_notes}\n"
            f"\nTemplate Content:\n{selected_style_output.selected_template.get('content', '')}"
        )

        print("Prompt for generation prepared.")
        print("Invoking generation agent...")

        generation_agent = Agent(
            model=cover_letter_agent.model,
            deps_type=str,
            system_prompt=SYSTEM_PROMPT
        )
        llm_result = await generation_agent.run(prompt, deps=prompt)
        text = llm_result.data if hasattr(llm_result, "data") else str(llm_result)

        print("Cover letter generated.")

        return CoverLetterOutput(
            cover_letter=text,
            summary=f"Generated cover letter for {input_data.job_title} at {input_data.hiring_company}",
            used_highlights=input_data.working_experience.split("; ") if input_data.working_experience else [],
            used_github_info={"username": input_data.github_username} if input_data.github_username else {}
        )

    except Exception as e:
        print(f"[ERROR] retrieve_styles failed: {e}")
        fallback_text = (
            f"Dear Hiring Manager,\n\n"
            f"I am writing to express my interest in the {ctx.deps.job_title or 'position'} at "
            f"{ctx.deps.hiring_company or 'your company'}.\n\n"
            f"With my background in {ctx.deps.skillsets or 'relevant skills'} and "
            f"experience in {ctx.deps.working_experience or 'related roles'}, "
            f"I believe I can make a meaningful contribution to your team.\n\n"
            f"Sincerely,\n{ctx.deps.applicant_name or 'Applicant'}"
        )
        return CoverLetterOutput(
            cover_letter=fallback_text,
            summary="Fallback cover letter generated due to processing error",
            used_highlights=[],
            used_github_info={"username": ctx.deps.github_username} if hasattr(ctx.deps, 'github_username') else {}
        )


