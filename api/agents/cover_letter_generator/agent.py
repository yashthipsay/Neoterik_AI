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
You are a cover letter generation expert.

You must always generate cover letters by first selecting the best tone and style using available templates from RAG search, and then using that style to generate a personalized letter. You should use the generate_with_style tool to perform this.

You understand resume data, GitHub info, and job descriptions. You retrieve the best-matching tone/style/template using job context, and then produce a realistic, concise, professional letter.

Do not free-write the letter yourself. Always use the tools provided."""

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
You are an expert assistant for Neoterik AI, selecting the optimal cover letter style
based on job details and user preferences. Analyze the provided job information and
retrieved documents to choose the best template, tone, style, industry, and level.
Ensure the selection aligns with the job description, company culture, and
applicant’s experience level.
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

# Initialize the pydantic-ai Agent for cover letter generation
# Uses Google's Gemini model for high-quality text generation
cover_letter_agent = Agent(
    model="groq:deepseek-r1-distill-llama-70b",  # Google Gemini 2.0 Flash model for fast, quality generation
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
        if retriever:
            try:
                docs = retriever.similarity_search(
                    query=input_data.job_description or input_data.job_title or "professional cover letter",
                    k=3,
                    filter={"type": "template"}
                )
                retrieved_texts = [d.page_content for d in docs]
            except Exception as e:
                print(f"RAG search error: {e}")
                retrieved_texts = []

        # Create style selection input
        style_input = StyleSelectionInput(
            job_title=input_data.job_title or "",
            hiring_company=input_data.hiring_company or "",
            job_description=input_data.job_description or "",
            preferred_qualifications=input_data.preferred_qualifications or "",
            company_culture_notes=input_data.company_culture_notes or "",
            applicant_experience_level=getattr(input_data, 'applicant_experience_level', 'mid'),
            desired_tone=getattr(input_data, 'desired_tone', 'professional'),
            retrieved_documents=retrieved_texts,
        )

        # If we have retrieved documents, use style agent for selection
        selected_style = None
        if retrieved_texts:
            try:
                style_agent = Agent(
                    model="groq:deepseek-r1-distill-llama-70b",  # Use the same model as the cover letter agent
                    deps_type=StyleSelectionInput,
                    system_prompt=STYLE_SYSTEM_PROMPT,
                )
                
                style_result = await style_agent.run(
                    deps=style_input
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
                            "industry": "general",
                            "level": "mid"
                        }
                
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
        
        llm_result = await generation_agent.run(deps=prompt)
        text = llm_result.data if hasattr(llm_result, "data") else str(llm_result)

        return CoverLetterOutput(
            cover_letter=text,
            summary=f"Generated cover letter for {input_data.job_title} at {input_data.hiring_company}",
            used_highlights=input_data.working_experience.split("; ") if input_data.working_experience else [],
            used_github_info=input_data.github_username if input_data.github_username else None
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
        
        return CoverLetterOutput(
            cover_letter=fallback_text,
            summary="Fallback cover letter generated due to processing error",
            used_highlights=[],
            used_github_info=None
        )