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
You are an expert AI cover letter writer. Your goal is to create personalized, professional, and human-like cover letters that are 200-300 words. Be confident and persuasive, but not overly dramatic.

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
    
# def build_prompt_for_gemini(input_data: CoverLetterInput, github_info, resume_data) -> str:
#     prompt_parts = []

#     # 1. Start with the System Prompt
#     # prompt_parts.append(SYSTEM_PROMPT) # Ensure this is the improved system prompt

#     # 2. Clearly delineate sections for the LLM
#     prompt_parts.append("\n--- JOB APPLICATION DETAILS ---")
#     prompt_parts.append(f"Job Title: {input_data.job_title}")
#     prompt_parts.append(f"Hiring Company: {input_data.hiring_company}")
#     # prompt_parts.append(f"Company Website: {input_data.company_url}")

#     prompt_parts.append("\n--- JOB REQUIREMENTS ---")
#     prompt_parts.append(f"Job Description:\n{input_data.job_description}")
#     if input_data.preferred_qualifications:
#         prompt_parts.append(f"Preferred Qualifications:\n{input_data.preferred_qualifications}")

#     prompt_parts.append("\n--- APPLICANT BACKGROUND ---")
#     prompt_parts.append(f"Applicant Name: {input_data.applicant_name}")
#     if input_data.skillsets:
#         prompt_parts.append(f"Applicant Skillsets:\n{input_data.skillsets}")
#     if input_data.working_experience: # Assuming this is derived from resume parsing
#         prompt_parts.append(f"Applicant Working Experience Highlights:\n{input_data.working_experience}")
#     if resume_data: # If you pass the full resume text
#         # Consider truncating or summarizing if very long to save tokens
#         prompt_parts.append(f"Full Resume Content (for additional context):\n{resume_data[:2000]}...")
#     if github_info:
#         prompt_parts.append(f"GitHub Profile Information for {input_data.github_username or 'applicant'}:")
#         prompt_parts.append(f"- GitHub Summary: {github_info}")

#     prompt_parts.append("\n--- COMPANY CULTURE & CONTEXTUAL NOTES ---")
#     if input_data.company_culture_notes:
#         prompt_parts.append(f"Company Culture Notes:\n{input_data.company_culture_notes}")

#     prompt_parts.append("\n--- COVER LETTER GENERATION TASK ---")
#     prompt_parts.append(
#         "Focus on integrating all relevant information to demonstrate the applicant's ideal fit. Use the styles filter before generating the cover letter, to ensure the desired tone for the cover letter is set."
#     )

#     return "\n\n".join(prompt_parts)

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

def format_github_info(info):
    if not info:
        return "No GitHub information available."
    lines = []
    if info.get("name"):
        lines.append(f"Name: {info['name']}")
    if info.get("username"):
        lines.append(f"Username: {info['username']}")
    if info.get("bio"):
        lines.append(f"Bio: {info['bio']}")
    if info.get("location"):
        lines.append(f"Location: {info['location']}")
    if info.get("repo_count") is not None:
        lines.append(f"Public Repos: {info['repo_count']}")
    if info.get("followers") is not None:
        lines.append(f"Followers: {info['followers']}")
    if info.get("following") is not None:
        lines.append(f"Following: {info['following']}")
    if info.get("key_skills"):
        lines.append(f"Key Skills: {', '.join(info['key_skills'])}")
    if info.get("notable_repositories"):
        lines.append("Notable Repositories:")
        for repo in info["notable_repositories"]:
            repo_line = f"  - {repo['name']}"
            if repo.get("description"):
                repo_line += f": {repo['description']}"
            if repo.get("language"):
                repo_line += f" [{repo['language']}]"
            lines.append(repo_line)
    return "\n".join(lines)

@cover_letter_agent.tool
async def retrieve_styles(ctx: RunContext[StyleSelectionInput]) -> CoverLetterOutput:
    """
    RAG tool to fetch top style templates and generate cover letter based on job description and preferences.
    """
    print("\n=== Debug: retrieve_styles (enhanced version) ===")
    print(f"Desired tone from input: {ctx.deps.desired_tone}")
    print(f"Job title: {ctx.deps.job_title}")
    print(f"Hiring company: {ctx.deps.hiring_company}")
    print(f"Job description length: {len(ctx.deps.job_description or '')}")

    retriever = get_style_retriever_cloud()
    
    # Retrieve multiple types of data for comprehensive cover letter generation
    all_retrieved_data = {
        "templates": [],
        "tones": [],
        "phrases": [],
        "skills": [],
        "values": []
    }

    # If desired_tone is specified and not auto/none/empty, use tone-specific filter
    if ctx.deps.desired_tone and ctx.deps.desired_tone.lower() not in ["auto", "none", ""]:
        print(f"\n--- TONE-SPECIFIC RETRIEVAL ---")
        
        # 1. Get templates with specific tone
        template_filter = {"tone": ctx.deps.desired_tone}
        print(f"Searching templates with filter: {template_filter}")
        
        template_docs = retriever.similarity_search(
            query=ctx.deps.job_description or ctx.deps.job_title,
            k=2,
            filter=template_filter
        )
        all_retrieved_data["templates"] = [{"content": d.page_content, "metadata": d.metadata} for d in template_docs]
        print(f"Retrieved {len(all_retrieved_data['templates'])} template documents")
        
        # 2. Get tone guidelines
        tone_filter = {"tone": ctx.deps.desired_tone}
        print(f"Searching tones with filter: {tone_filter}")
        
        tone_docs = retriever.similarity_search(
            query=f"tone {ctx.deps.desired_tone}",
            k=1,
            filter=tone_filter
        )
        all_retrieved_data["tones"] = [{"content": d.page_content, "metadata": d.metadata} for d in tone_docs]
        print(f"Retrieved {len(all_retrieved_data['tones'])} tone documents")

        # Create style selection from retrieved template
        if all_retrieved_data["templates"]:
            metadata = all_retrieved_data["templates"][0].get("metadata", {})
            selected_style = StyleSelectionOutput(
                selected_template={
                    "style": metadata.get("style", "professional"),
                    "content": all_retrieved_data["templates"][0]["content"]
                },
                tone=metadata.get("tone", ctx.deps.desired_tone),
                style=metadata.get("style", "professional"),
                industry=metadata.get("industry", "general"),
                level=metadata.get("level", "mid"),
                retrieved_documents=all_retrieved_data["templates"]
            )
        else:
            selected_style = None
            print("WARNING: No templates found for specified tone")
    else:
        print(f"\n--- DEFAULT RETRIEVAL WITH STYLE AGENT ---")
        
        # 1. Get general templates
        template_filter = {"type": "template"}
        print(f"Searching templates with filter: {template_filter}")
        
        template_docs = retriever.similarity_search(
            query=ctx.deps.job_description or ctx.deps.job_title,
            k=3,
            filter=template_filter
        )
        all_retrieved_data["templates"] = [{"content": d.page_content, "metadata": d.metadata} for d in template_docs]
        print(f"Retrieved {len(all_retrieved_data['templates'])} template documents")

        # Default style structure
        default_style = {
            "selected_template": {
                "style": "professional",
                "content": all_retrieved_data["templates"][0]["content"] if all_retrieved_data["templates"] else ""
            },
            "tone": "professional",
            "style": "professional",
            "industry": "general",
            "level": "mid",
            "retrieved_documents": all_retrieved_data["templates"]
        }

        if all_retrieved_data["templates"]:
            metadata = all_retrieved_data["templates"][0].get("metadata", {})
            default_style.update({
                "tone": metadata.get("tone", "professional"),
                "style": metadata.get("style", "professional"),
                "industry": metadata.get("industry", "general"),
                "level": metadata.get("level", "mid"),
            })
            default_style["selected_template"]["style"] = metadata.get("style", "professional")

        # Run style agent for nuanced selection
        try:
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
                applicant_experience_level=getattr(ctx.deps, 'applicant_experience_level', None),
                desired_tone="auto",
                # Add the missing parsed data
                applicant_name=getattr(ctx.deps, 'applicant_name', ''),
                working_experience=getattr(ctx.deps, 'working_experience', ''),
                qualifications=getattr(ctx.deps, 'qualifications', ''),
                skillsets=getattr(ctx.deps, 'skillsets', ''),
                github_username=getattr(ctx.deps, 'github_username', ''),
                resume_data=getattr(ctx.deps, 'resume_data', None),
                github_data=getattr(ctx.deps, 'github_data', None)
            )
            style_result = await style_agent.run(deps=rag_input)
            structured = style_result.data if hasattr(style_result, "data") else style_result
            print(f"Style agent output: {structured}")

            if isinstance(structured, str):
                stripped = structured.strip()
                if stripped in ["professional", "fun-loving", "most-improved", "short-and-sweet", "unique", "career-change", "enthusiastic"]:
                    structured = {
                        "selected_template": {
                            "style": stripped,
                            "content": default_style["selected_template"]["content"]
                        },
                        "tone": stripped,
                        "style": stripped,
                        "industry": default_style["industry"],
                        "level": default_style["level"],
                        "retrieved_documents": all_retrieved_data["templates"]
                    }
                elif stripped == "":
                    print("Empty style result, using default.")
                    structured = default_style
                else:
                    try:
                        structured = json.loads(stripped)
                        print(f"Parsed JSON: {structured}")
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing error: {e}")
                        structured = default_style
            elif not isinstance(structured, dict):
                structured = default_style

            # Ensure required fields
            for key in ["style", "tone", "industry", "level"]:
                if key not in structured:
                    structured[key] = default_style[key]
            structured["retrieved_documents"] = all_retrieved_data["templates"]
            structured.setdefault("selected_template", default_style["selected_template"])
            
            selected_style = StyleSelectionOutput(**structured)
        except Exception as e:
            print(f"Style agent error: {e}")
            selected_style = StyleSelectionOutput(**default_style)

    # 3. Retrieve additional supporting data regardless of path taken
    print(f"\n--- RETRIEVING SUPPORTING DATA ---")
    
    # Get industry-specific phrases
    try:
        phrase_docs = retriever.similarity_search(
            query=f"{ctx.deps.job_description or ctx.deps.job_title} industry phrases",
            k=2,
            filter={"type": "phrase"}
        )
        all_retrieved_data["phrases"] = [{"content": d.page_content, "metadata": d.metadata} for d in phrase_docs]
        print(f"Retrieved {len(all_retrieved_data['phrases'])} phrase documents")
    except Exception as e:
        print(f"Error retrieving phrases: {e}")
        all_retrieved_data["phrases"] = []

    # Get relevant skills
    try:
        skill_docs = retriever.similarity_search(
            query=f"{ctx.deps.job_description or ctx.deps.job_title} skills",
            k=3,
            filter={"type": "skill"}
        )
        all_retrieved_data["skills"] = [{"content": d.page_content, "metadata": d.metadata} for d in skill_docs]
        print(f"Retrieved {len(all_retrieved_data['skills'])} skill documents")
    except Exception as e:
        print(f"Error retrieving skills: {e}")
        all_retrieved_data["skills"] = []

    # Get company values
    try:
        value_docs = retriever.similarity_search(
            query=f"{ctx.deps.company_culture_notes or ctx.deps.hiring_company} values",
            k=2,
            filter={"type": "value"}
        )
        all_retrieved_data["values"] = [{"content": d.page_content, "metadata": d.metadata} for d in value_docs]
        print(f"Retrieved {len(all_retrieved_data['values'])} value documents")
    except Exception as e:
        print(f"Error retrieving values: {e}")
        all_retrieved_data["values"] = []

    # Print detailed retrieval results
    print(f"\n--- RETRIEVAL SUMMARY ---")
    for data_type, items in all_retrieved_data.items():
        print(f"{data_type.upper()}: {len(items)} items")
        for i, item in enumerate(items):
            print(f"  {i+1}. Metadata: {item.get('metadata', {})}")
            content_preview = item.get('content', '')[:100] + '...' if len(item.get('content', '')) > 100 else item.get('content', '')
            print(f"     Content: {content_preview}")

    # --- COVER LETTER GENERATION LOGIC ---
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

    try:
        # Build enhanced generation prompt using all retrieved data
        print(f"\n--- BUILDING GENERATION PROMPT ---")
        
        github_data = getattr(ctx.deps, 'github_data', None)
        used_github_info = {}

        # --- PATCH START ---
        # Always use the same logic for both output and prompt
        github_data = ctx.deps.github_data or {}
        used_github_info = {}
        if isinstance(github_data, dict) and github_data:
            used_github_info = {
                "name": github_data.get("name"),
                "username": github_data.get("username"),
                "bio": github_data.get("bio"),
                "location": github_data.get("location"),
                "repo_count": github_data.get("public_repos_count"),
                "followers": github_data.get("followers"),
                "following": github_data.get("following"),
                "key_skills": github_data.get("skills_tags", []),
                "notable_repositories": [
                    {
                        "name": repo.get("name"),
                        "description": repo.get("description"),
                        "language": repo.get("language")
                    }
                    for repo in github_data.get("top_repositories", [])
                ]
            }
        else:
            used_github_info = None
        github_info_formatted = format_github_info(used_github_info)

        base_prompt = f"""Generate a personalized cover letter for {input_data.job_title} at {input_data.hiring_company}.

APPLICANT INFORMATION:
- Name: {input_data.applicant_name}
- Working Experience: {input_data.working_experience}
- Qualifications: {input_data.qualifications}
- Skillsets: {input_data.skillsets}
- Company Culture Notes: {input_data.company_culture_notes}

GITHUB PROFILE INFORMATION:
{github_info_formatted}

JOB DETAILS:
- Job Description: {input_data.job_description}
- Preferred Qualifications: {input_data.preferred_qualifications}

"""

        # Add selected style and template info
        if selected_style and selected_style.selected_template:
            base_prompt += f"""STYLE GUIDANCE:
- Selected Style: {selected_style.selected_template.get('style', 'professional')}
- Tone: {selected_style.tone}
- Industry: {selected_style.industry}
- Level: {selected_style.level}

"""
            if selected_style.selected_template.get('content'):
                base_prompt += f"TEMPLATE REFERENCE:\n{selected_style.selected_template['content']}\n\n"

        # Add retrieved supporting data
        if all_retrieved_data["tones"]:
            tone_info = "\n".join([item["content"] for item in all_retrieved_data["tones"]])
            base_prompt += f"TONE GUIDELINES:\n{tone_info}\n\n"

        if all_retrieved_data["phrases"]:
            phrase_info = "\n".join([item["content"] for item in all_retrieved_data["phrases"]])
            base_prompt += f"INDUSTRY PHRASES TO CONSIDER:\n{phrase_info}\n\n"

        if all_retrieved_data["skills"]:
            skill_info = "\n".join([item["content"] for item in all_retrieved_data["skills"]])
            base_prompt += f"RELEVANT SKILLS TO HIGHLIGHT:\n{skill_info}\n\n"

        if all_retrieved_data["values"]:
            value_info = "\n".join([item["content"] for item in all_retrieved_data["values"]])
            base_prompt += f"COMPANY VALUES TO ALIGN WITH:\n{value_info}\n\n"

        base_prompt += """INSTRUCTIONS:
- Write a compelling, professional cover letter that integrates the style guidance and retrieved information
- Use industry-specific phrases naturally within the content
- Highlight relevant skills and experiences that match the job requirements
- Align with company values where appropriate
- Maintain the specified tone throughout
- Keep the letter between 200-400 words
- Make it feel personal and authentic, not template-driven

Write only the complete cover letter text."""

        print(f"Generated prompt length: {len(base_prompt)} characters")
        print(f"Prompt preview: {base_prompt[:5000]}...")

        generation_agent = Agent(
            model=cover_letter_agent.model,
            deps_type=str,
            system_prompt=SYSTEM_PROMPT
        )
        llm_result = await generation_agent.run(base_prompt, deps=base_prompt)
        text = llm_result.data if hasattr(llm_result, "data") else str(llm_result)

        print(f"\n--- GENERATION COMPLETE ---")
        print(f"Generated cover letter length: {len(text)} characters")

        github_info_dict = {}
        if hasattr(input_data, 'github_username') and input_data.github_username:
            github_info_dict = {"username": input_data.github_username}

        return CoverLetterOutput(
            cover_letter=text,
            summary=f"Generated cover letter for {input_data.job_title} at {input_data.hiring_company} using {selected_style.tone if selected_style else 'professional'} tone with {sum(len(v) for v in all_retrieved_data.values())} retrieved references",
            used_highlights=input_data.working_experience.split("; ") if input_data.working_experience else [],
            used_github_info=github_info_dict
        )

    except Exception as e:
        print(f"Error in cover letter generation: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback generation
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
        github_info_dict = {}
        if hasattr(input_data, 'github_username') and input_data.github_username:
            github_info_dict = {"username": input_data.github_username}
        return CoverLetterOutput(
            cover_letter=fallback_text,
            summary="Fallback cover letter generated due to processing error",
            used_highlights=[],
            used_github_info=github_info_dict
        )

