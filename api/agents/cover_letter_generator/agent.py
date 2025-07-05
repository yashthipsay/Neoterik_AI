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
You are an AI-powered expert cover letter writer and a strategic career advisor. Your primary objective is to craft personalized, professional, practical and humanized cover letters.
The Cover Letter length should be not too long, and not too short, ideally around 200-400 words. DO NOT USE VERY DRAMATIC LANGUAGE, but be confident and persuasive.
Your task is to analyze the provided job details, applicant's information (including resume highlights and GitHub data), and insights into the hiring company's culture. Based on this analysis, you must generate a cover letter that:

1.  **Directly addresses and integrates** specific keywords and requirements from the `job_description` and `preferred_qualifications`.
2.  **Articulates the applicant's value proposition** by clearly showing *how* their `skillsets` and `working_experience` align with, and will contribute to, the job's demands. Focus on impact and outcomes.
3.  **Skill Presentation (Crucial!):**
    * **Avoid lengthy, sequential lists of technologies.** Instead, seamlessly integrate relevant technologies into descriptions of projects, accomplishments, or specific problem-solving scenarios.
    * **Prioritize relevance:** Only highlight technologies most pertinent to the job description.
4.  **Demonstrates a genuine understanding** for the `hiring_company`'s mission, values, and `company_culture_notes`, weaving these elements naturally into the narrative to create an authentic connection. But don't be too dramatic or verbose.
5.  **Maintains a confident, persuasive, and naturally flowing tone** throughout. The letter should sound like it was written by a human, conveying personality and interest.
6.  **Is structured logically** with a concise introduction, compelling body paragraphs showcasing relevant experience and company fit (with technical skills integrated contextually), and a strong, polite call to action.
7.  **Avoids generic phrases** and instead focuses on concrete connections between the applicant and the role/company, making every sentence purposeful. Do not directly copy phrases from the job description. It's instantly noticable and unprofessional. Do it subtly if required.
8.  **Handles all provided information gracefully**, integrating it where relevant and omitting or generalizing if a specific piece of information is missing from the input.

**Output Format:** Your response must be the complete text of the cover letter, ready for immediate use, without any additional conversational text, placeholders, or introductory/concluding remarks from you.

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

STYLE_SYSTEM_PROMPT = """
You are a style selection assistant for a cover letter generator. Your task is to identify the most appropriate writing style based on the user's request or job context.

**Available Styles:**
- **Professional & Formal:** Emphasizes traditional business language, clarity, and precision.
- **Enthusiastic & Dynamic:** Conveys high energy, passion, and a proactive attitude.
- **Concise & Direct:** Focuses on brevity, impactful statements, and getting straight to the point.
- **Innovative & Creative:** Highlights unique problem-solving, forward-thinking, and unconventional approaches (use with caution for formal roles).

**Instructions:**
1.  Analyze the provided `job_description`, `company_culture_notes`, and any explicit `style_preference` from the user.
2.  Select the single most suitable style from the "Available Styles" list that best aligns with the job's requirements and company's expected tone.
3.  If no specific style is indicated or implied, default to "Professional & Formal."

**Output Format:** Provide only the name of the chosen style (e.g., "Professional & Formal", "Enthusiastic & Dynamic").
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
    prompt_parts.append(SYSTEM_PROMPT) # Ensure this is the improved system prompt

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
        "Focus on integrating all relevant information to demonstrate the applicant's ideal fit."
    )

    return "\n\n".join(prompt_parts)

# Initialize the pydantic-ai Agent for cover letter generation
# Uses Google's Gemini model for high-quality text generation
cover_letter_agent = Agent(
    model="gemini-2.5-pro",  # Google Gemini 2.0 Flash model for fast, quality generation
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
    
    
# ----------------------------------------------------------------------------
# Tool: retrieve style templates via RAG
# ----------------------------------------------------------------------------
@cover_letter_agent.tool
async def retrieve_styles(
    ctx: RunContext[StyleSelectionInput]
) -> StyleSelectionOutput:
    """
    RAG tool to fetch top style templates based on job description and preferences.
    """
    retriever = get_style_retriever()
    # similarity search on job description
    docs = retriever.similarity_search(
        query=ctx.deps.job_description or ctx.deps.job_title,
        k=3,
        filter={"type": "template"}
    )
    
    retrieved_docs = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
    # collect raw page_contents
    retrieved_texts = [d.page_content for d in docs]

    # call a separate style selector agent for structured output
    style_agent = Agent(
        model=cover_letter_agent.model,
        deps_type=StyleSelectionInput,
        system_prompt=STYLE_SYSTEM_PROMPT,
    )
    rag_input = StyleSelectionInput(
        job_title=ctx.deps.job_title,
        hiring_company=ctx.deps.hiring_company,
        job_description=ctx.deps.job_description,
        preferred_qualifications=ctx.deps.preferred_qualifications,
        company_culture_notes=ctx.deps.company_culture_notes,
        applicant_experience_level=ctx.deps.applicant_experience_level,
        desired_tone=ctx.deps.desired_tone,
        retrieved_documents=retrieved_docs
    )
    style_result = await style_agent.run(deps=rag_input)
    structured = style_result.data if hasattr(style_result, "data") else style_result
    
    # Fix: Better JSON parsing with error handling
    if isinstance(structured, str):
        try:
            if structured.strip():  # Check if string is not empty
                structured = json.loads(structured)
            else:
                # Return default structure for empty responses
                structured = {
                    "selected_template": {"style": "professional", "content": ""},
                    "tone": "professional",
                    "style": "professional", 
                    "industry": "general",
                    "level": "mid",
                    "retrieved_documents": retrieved_docs
                }
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in retrieve_styles: {e}")
            # Return default structure for invalid JSON
            structured = {
                "selected_template": {"style": "professional", "content": ""},
                "tone": "professional", 
                "style": "professional",
                "industry": "general",
                "level": "mid",
                "retrieved_documents": retrieved_docs
            }
            
    # Ensure structured dict has all required fields
    if isinstance(structured, dict):
        structured.setdefault("style", "professional")
        structured.setdefault("retrieved_documents", retrieved_docs)
    
    return StyleSelectionOutput(**structured)

# ----------------------------------------------------------------------------
# Main tool: generate cover letter using chosen style
# ----------------------------------------------------------------------------
@cover_letter_agent.tool
async def generate_with_style(
    ctx: RunContext[CoverLetterInput]
) -> CoverLetterOutput:
    """
    Generates a cover letter after retrieving the best style using RAG.
    """
    input_data = ctx.deps

    try:
        # Get RAG retriever
        retriever = get_style_retriever()
        
        # If RAG is available, perform similarity search for templates
        retrieved_texts = []
        retrieved_docs = []
        if retriever:
            try:
                docs = retriever.similarity_search(
                    query=input_data.job_description or input_data.job_title or "professional cover letter",
                    k=3,
                    filter={"type": "template"}
                )
                retrieved_texts = [d.page_content for d in docs]
                retrieved_docs = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
            except Exception as e:
                print(f"RAG search error: {e}")
                retrieved_texts = []
                retrieved_docs = []

        # Create style selection input
        style_input = StyleSelectionInput(
            job_title=input_data.job_title or "",
            hiring_company=input_data.hiring_company or "",
            job_description=input_data.job_description or "",
            preferred_qualifications=input_data.preferred_qualifications or "",
            company_culture_notes=input_data.company_culture_notes or "",
            applicant_experience_level=getattr(input_data, 'applicant_experience_level', 'mid'),
            desired_tone=getattr(input_data, 'desired_tone', 'professional'),
            retrieved_documents=retrieved_docs,  # Use retrieved_docs instead of retrieved_texts
        )

        # If we have retrieved documents, use style agent for selection
        selected_style = None
        if retrieved_texts:
            try:
                style_agent = Agent(
                    model="gemini-2.5-pro",
                    deps_type=StyleSelectionInput,
                    system_prompt=STYLE_SYSTEM_PROMPT,
                )
                
                style_result = await style_agent.run(
                    "Select the best template and style based on the job description and retrieved documents.", deps=style_input
                )
                
                structured = style_result.data if hasattr(style_result, "data") else str(style_result)
                if isinstance(structured, str):
                    try:
                        structured = json.loads(structured)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, use default structure
                        structured = {
                            "selected_template": {"style": "professional", "content": ""},
                            "tone": "professional",
                            "style": "professional",  # Add required field
                            "industry": "general",
                            "level": "mid",
                            "retrieved_documents": retrieved_docs  # Add required field
                        }
                
                # Ensure all required fields are present
                if isinstance(structured, dict):
                    structured.setdefault("style", "professional")
                    structured.setdefault("retrieved_documents", retrieved_docs)
                    structured.setdefault("selected_template", {"style": "professional", "content": ""})
                    structured.setdefault("tone", "professional")
                    structured.setdefault("industry", "general")
                    structured.setdefault("level", "mid")
                
                selected_style = StyleSelectionOutput(**structured) if isinstance(structured, dict) else None
            except Exception as e:
                print(f"Style selection error: {e}")
                selected_style = None

        # Build the generation prompt
        if selected_style and selected_style.selected_template:
            prompt = (
                f"Using the '{selected_style.selected_template.get('style', 'professional')}' template and a '{selected_style.tone}' tone, "
                f"generate a personalized cover letter for {input_data.job_title} at {input_data.hiring_company}. "
                f"Here are the details:\n\n"
                f"Job Description: {input_data.job_description}\n"
                f"Applicant Name: {input_data.applicant_name}\n"
                f"Working Experience: {input_data.working_experience}\n"
                f"Qualifications: {input_data.qualifications}\n"
                f"Skillsets: {input_data.skillsets}\n"
                f"Company Culture Notes: {input_data.company_culture_notes}\n"
            )
            if selected_style.selected_template.get('content'):
                prompt += f"\nTemplate Content for reference:\n{selected_style.selected_template['content']}"
        else:
            # Fallback prompt without RAG
            prompt = (
                f"Generate a professional cover letter for {input_data.job_title} at {input_data.hiring_company}. "
                f"Use the following information:\n\n"
                f"Job Description: {input_data.job_description}\n"
                f"Applicant Name: {input_data.applicant_name}\n"
                f"Working Experience: {input_data.working_experience}\n"
                f"Qualifications: {input_data.qualifications}\n"
                f"Skillsets: {input_data.skillsets}\n"
                f"Company Culture Notes: {input_data.company_culture_notes}\n"
                f"\nWrite a compelling, professional cover letter that highlights the candidate's relevant experience and enthusiasm for the role."
            )

        # Generate the letter using a simple text generation approach
        generation_agent = Agent(
            model=cover_letter_agent.model,
            deps_type=str,
            system_prompt="You are a professional cover letter writer. Generate clear, compelling cover letters based on the provided information."
        )
        # Add prompt from the context here
        llm_result = await generation_agent.run(prompt, deps=prompt)
        text = llm_result.data if hasattr(llm_result, "data") else str(llm_result)

        # FIX: Ensure used_github_info is always a dictionary
        github_info_dict = {}
        if hasattr(input_data, 'github_username') and input_data.github_username:
            github_info_dict = {"username": input_data.github_username}

        return CoverLetterOutput(
            cover_letter=text,
            summary=f"Generated cover letter for {input_data.job_title} at {input_data.hiring_company}",
            used_highlights=input_data.working_experience.split("; ") if input_data.working_experience else [],
            used_github_info=github_info_dict  # Always pass a dictionary
        )

    except Exception as e:
        print(f"Error in generate_with_style: {e}")
        # Return a basic fallback cover letter
        fallback_text = (
            f"Dear Hiring Manager,\n\n"
            f"I am writing to express my interest in the {input_data.job_title or 'position'} "
            f"at {input_data.hiring_company or 'your company'}.\n\n"
            f"With my background in {input_data.skillsets or 'relevant technologies'} "
            f"and experience in {input_data.working_experience or 'the field'}, "
            f"I believe I would be a valuable addition to your team.\n\n"
            f"Thank you for considering my application.\n\n"
            f"Sincerely,\n{input_data.applicant_name or 'Applicant'}"
        )
        
        # FIX: Ensure fallback also uses dictionary for used_github_info
        github_info_dict = {}
        if hasattr(input_data, 'github_username') and input_data.github_username:
            github_info_dict = {"username": input_data.github_username}
        
        return CoverLetterOutput(
            cover_letter=fallback_text,
            summary="Fallback cover letter generated due to processing error",
            used_highlights=[],
            used_github_info=github_info_dict  # Always pass a dictionary
        )