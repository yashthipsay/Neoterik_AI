# ----------------------------------------------------------------------------
# Tool: retrieve style templates via RAG
# ----------------------------------------------------------------------------
# @cover_letter_agent.tool
# async def retrieve_styles(
#     ctx: RunContext[StyleSelectionInput]
# ) -> StyleSelectionOutput:
#     """
#     RAG tool to fetch top style templates based on job description and preferences.
#     """
#     print("\n=== Debug: retrieve_styles ===")
#     print(f"Desired tone from input: {ctx.deps.desired_tone}")

#     retriever = get_style_retriever()

#     style_filter = {}

#     print(f"Setting style filter based on tone...")

#     # If desired_tone is specified, use it directly without running the style agent
#     if ctx.deps.desired_tone and ctx.deps.desired_tone != "None":
#         style_filter = {
#             "tone": ctx.deps.desired_tone
#         }
#         print(f"Using tone-specific filter: {style_filter}")
#         docs = retriever.similarity_search(
#             query=ctx.deps.job_description or ctx.deps.job_title,
#             k=1,
#             filter=style_filter
#         )
        
#         retrieved_docs = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
#         print(f"Retrieved {len(retrieved_docs)} documents")

#         # Create output structure directly from the retrieved document
#         if retrieved_docs:
#             metadata = retrieved_docs[0].get("metadata", {})
#             return StyleSelectionOutput(
#                 selected_template={
#                     "style": metadata.get("style", "professional"),
#                     "content": retrieved_docs[0]["content"]
#                 },
#                 tone=metadata.get("tone", ctx.deps.desired_tone),
#                 style=metadata.get("style", "professional"),
#                 industry=metadata.get("industry", "general"),
#                 level=metadata.get("level", "mid"),
#                 retrieved_documents=retrieved_docs
#             )
#     else:
#         # If no desired_tone, use default template filter and run style agent
#         style_filter = {"type": "template"}
#         print(f"Using default filter: {style_filter}")
#         docs = retriever.similarity_search(
#             query=ctx.deps.job_description or ctx.deps.job_title,
#             k=1,
#             filter=style_filter
#         )
        
#         retrieved_docs = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
#         print(f"Retrieved {len(retrieved_docs)} documents")

#         # Default style structure
#         default_style = {
#             "selected_template": {
#                 "style": "professional",
#                 "content": retrieved_docs[0]["content"] if retrieved_docs else ""
#             },
#             "tone": "professional",
#             "style": "professional",
#             "industry": "general",
#             "level": "mid",
#             "retrieved_documents": retrieved_docs
#         }

#         if retrieved_docs:
#             metadata = retrieved_docs[0].get("metadata", {})
#             default_style.update({
#                 "tone": metadata.get("tone", "professional"),
#                 "style": metadata.get("style", "professional"),
#                 "industry": metadata.get("industry", "general"),
#                 "level": metadata.get("level", "mid"),
#             })
#             default_style["selected_template"]["style"] = metadata.get("style", "professional")

#         # Only run style agent if no desired_tone was specified
#         try:
#             style_agent = Agent(
#                 model=cover_letter_agent.model,
#                 deps_type=StyleSelectionInput,
#                 system_prompt=STYLE_SYSTEM_PROMPT,
#             )
#             rag_input = StyleSelectionInput(
#                 job_title=ctx.deps.job_title,
#                 hiring_company=ctx.deps.hiring_company,
#                 job_description=ctx.deps.job_description,
#                 preferred_qualifications=ctx.deps.preferred_qualifications,
#                 company_culture_notes=ctx.deps.company_culture_notes,
#                 applicant_experience_level=ctx.deps.applicant_experience_level,
#                 desired_tone=ctx.deps.desired_tone
#             )
#             style_result = await style_agent.run(deps=rag_input)
#             structured = style_result.data if hasattr(style_result, "data") else style_result

#             if isinstance(structured, str):
#                 stripped = structured.strip()
#                 if stripped in ["professional", "fun-loving", "most-improved", "short-and-sweet", "unique", "career-change", "enthusiastic"]:
#                     structured = {
#                         "selected_template": {
#                             "style": stripped,
#                             "content": default_style["selected_template"]["content"]
#                         },
#                         "tone": stripped,
#                         "style": stripped,
#                         "industry": default_style["industry"],
#                         "level": default_style["level"],
#                         "retrieved_documents": retrieved_docs
#                     }
#                 elif stripped == "":
#                     print("Received empty style result string, using default style.")
#                     structured = default_style
#                 else:
#                     try:
#                         print(f"RAW STRUCTURED OUTPUT: {structured}")
#                         structured = json.loads(stripped)
#                         print(f"Parsed JSON structure: {structured}")
#                     except json.JSONDecodeError as e:
#                         print(f"JSON parsing error in retrieve_styles: {e}")
#                         structured = default_style
#             elif not isinstance(structured, dict):
#                 structured = default_style

#         except Exception as e:
#             print(f"Style agent error: {e}")
#             structured = default_style

#         # Ensure all required fields are present
#         for key in ["style", "tone", "industry", "level"]:
#             if key not in structured:
#                 structured[key] = default_style[key]
        
#         structured["retrieved_documents"] = retrieved_docs
#         structured.setdefault("selected_template", default_style["selected_template"])
        
#         print(f"Final structured output: {structured}")
#         return StyleSelectionOutput(**structured)

#     # Fallback return if something goes wrong
#     return StyleSelectionOutput(
#         selected_template={"style": "professional", "content": ""},
#         tone="professional",
#         style="professional",
#         industry="general",
#         level="mid",
#         retrieved_documents=[]
#     )





@cover_letter_agent.tool
async def retrieve_styles(
    ctx: RunContext[StyleSelectionInput]
) -> CoverLetterOutput:  # Changed return type to CoverLetterOutput
    """
    RAG tool to fetch top style templates and generate cover letter based on job description and preferences.
    """
    print("\n=== Debug: retrieve_styles ===")
    print(f"Desired tone from input: {ctx.deps.desired_tone}")

    retriever = get_style_retriever()
    style_filter = {}

    print(f"Setting style filter based on tone...")

    # If desired_tone is specified, use it directly without running the style agent
    if ctx.deps.desired_tone and ctx.deps.desired_tone.lower() not in ["auto", "none", ""]:
        style_filter = {
            "tone": ctx.deps.desired_tone
        }
        print(f"Using tone-specific filter: {style_filter}")
        docs = retriever.similarity_search(
            query=ctx.deps.job_description or ctx.deps.job_title,
            k=1,
            filter=style_filter
        )
        
        retrieved_docs = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
        print(f"Retrieved {len(retrieved_docs)} documents")

        # Create style selection structure directly from the retrieved document
        if retrieved_docs:
            metadata = retrieved_docs[0].get("metadata", {})
            selected_style = StyleSelectionOutput(
                selected_template={
                    "style": metadata.get("style", "professional"),
                    "content": retrieved_docs[0]["content"]
                },
                tone=metadata.get("tone", ctx.deps.desired_tone),
                style=metadata.get("style", "professional"),
                industry=metadata.get("industry", "general"),
                level=metadata.get("level", "mid"),
                retrieved_documents=retrieved_docs
            )
        else:
            selected_style = None
    else:
        # If no desired_tone, use default template filter and run style agent
        style_filter = {"type": "template"}
        print(f"Using default filter: {style_filter}")
        docs = retriever.similarity_search(
            query=ctx.deps.job_description or ctx.deps.job_title,
            k=1,
            filter=style_filter
        )
        
        retrieved_docs = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
        print(f"Retrieved {len(retrieved_docs)} documents")

        # Default style structure
        default_style = {
            "selected_template": {
                "style": "professional",
                "content": retrieved_docs[0]["content"] if retrieved_docs else ""
            },
            "tone": "professional",
            "style": "professional",
            "industry": "general",
            "level": "mid",
            "retrieved_documents": retrieved_docs
        }

        if retrieved_docs:
            metadata = retrieved_docs[0].get("metadata", {})
            default_style.update({
                "tone": metadata.get("tone", "professional"),
                "style": metadata.get("style", "professional"),
                "industry": metadata.get("industry", "general"),
                "level": metadata.get("level", "mid"),
            })
            default_style["selected_template"]["style"] = metadata.get("style", "professional")

        # Only run style agent if no desired_tone was specified
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
                applicant_experience_level=ctx.deps.applicant_experience_level,
                desired_tone="auto"
            )
            style_result = await style_agent.run(deps=rag_input)
            structured = style_result.data if hasattr(style_result, "data") else style_result
            print(f"Structured output from style agent: {structured}")

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
                        "retrieved_documents": retrieved_docs
                    }
                elif stripped == "":
                    print("Received empty style result string, using default style.")
                    structured = default_style
                else:
                    try:
                        print(f"RAW STRUCTURED OUTPUT: {structured}")
                        structured = json.loads(stripped)
                        print(f"Parsed JSON structure: {structured}")
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing error in retrieve_styles: {e}")
                        structured = default_style
            elif not isinstance(structured, dict):
                structured = default_style

        except Exception as e:
            print(f"Style agent error: {e}")
            structured = default_style

        # Ensure all required fields are present
        for key in ["style", "tone", "industry", "level"]:
            if key not in structured:
                structured[key] = default_style[key]
        
        structured["retrieved_documents"] = retrieved_docs
        structured.setdefault("selected_template", default_style["selected_template"])
        
        print(f"Final structured output: {structured}")
        selected_style = StyleSelectionOutput(**structured)

    # --- COVER LETTER GENERATION LOGIC (moved from generate_with_style) ---
    
    # Convert StyleSelectionInput to CoverLetterInput for generation
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
            system_prompt= SYSTEM_PROMPT
        )
        
        llm_result = await generation_agent.run(prompt, deps=prompt)
        text = llm_result.data if hasattr(llm_result, "data") else str(llm_result)

        # Ensure used_github_info is always a dictionary
        github_info_dict = {}
        if hasattr(input_data, 'github_username') and input_data.github_username:
            github_info_dict = {"username": input_data.github_username}

        return CoverLetterOutput(
            cover_letter=text,
            summary=f"Generated cover letter for {input_data.job_title} at {input_data.hiring_company}",
            used_highlights=input_data.working_experience.split("; ") if input_data.working_experience else [],
            used_github_info=github_info_dict
        )

    except Exception as e:
        print(f"Error in cover letter generation: {e}")
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
        
        github_info_dict = {}
        if hasattr(input_data, 'github_username') and input_data.github_username:
            github_info_dict = {"username": input_data.github_username}
        
        return CoverLetterOutput(
            cover_letter=fallback_text,
            summary="Fallback cover letter generated due to processing error",
            used_highlights=[],
            used_github_info=github_info_dict
        )

# ----------------------------------------------------------------------------
# Main tool: generate cover letter using chosen style
# ----------------------------------------------------------------------------
# @cover_letter_agent.tool
# async def generate_with_style(
#     ctx: RunContext[CoverLetterInput]
# ) -> CoverLetterOutput:
#     """
#     Generates a cover letter after retrieving the best style using RAG.
#     """
#     input_data = ctx.deps

#     try:
#         # Get RAG retriever
#         retriever = get_style_retriever()
        
#         # If RAG is available, perform similarity search for templates
#         retrieved_texts = []
#         retrieved_docs = []
#         if retriever:
#             try:
#                 docs = retriever.similarity_search(
#                     query=input_data.job_description or input_data.job_title or "professional cover letter",
#                     k=3,
#                     filter={"type": "template"}
#                 )
#                 retrieved_texts = [d.page_content for d in docs]
#                 retrieved_docs = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
#             except Exception as e:
#                 print(f"RAG search error: {e}")
#                 retrieved_texts = []
#                 retrieved_docs = []

#         # Create style selection input
#         style_input = StyleSelectionInput(
#             job_title=input_data.job_title or "",
#             hiring_company=input_data.hiring_company or "",
#             job_description=input_data.job_description or "",
#             preferred_qualifications=input_data.preferred_qualifications or "",
#             company_culture_notes=input_data.company_culture_notes or "",
#             applicant_experience_level=getattr(input_data, 'applicant_experience_level', 'mid'),
#             desired_tone=getattr(input_data, 'desired_tone', 'professional'),
#             retrieved_documents=retrieved_docs,  # Use retrieved_docs instead of retrieved_texts
#         )

#         # If we have retrieved documents, use style agent for selection
#         selected_style = None
#         if retrieved_texts:
#             try:
#                 style_agent = Agent(
#                     model="gemini-2.5-flash",
#                     deps_type=StyleSelectionInput,
#                     system_prompt=STYLE_SYSTEM_PROMPT,
#                 )
                
#                 style_result = await style_agent.run(
#                     "Select the best template and style based on the job description and retrieved documents.", deps=style_input
#                 )
                
#                 structured = style_result.data if hasattr(style_result, "data") else str(style_result)
#                 if isinstance(structured, str):
#                     try:
#                         structured = json.loads(structured)
#                     except json.JSONDecodeError:
#                         # If JSON parsing fails, use default structure
#                         structured = {
#                             "selected_template": {"style": "professional", "content": ""},
#                             "tone": "professional",
#                             "style": "professional",  # Add required field
#                             "industry": "general",
#                             "level": "mid",
#                             "retrieved_documents": retrieved_docs  # Add required field
#                         }
                
#                 # Ensure all required fields are present
#                 if isinstance(structured, dict):
#                     structured.setdefault("style", "professional")
#                     structured.setdefault("retrieved_documents", retrieved_docs)
#                     structured.setdefault("selected_template", {"style": "professional", "content": ""})
#                     structured.setdefault("tone", "professional")
#                     structured.setdefault("industry", "general")
#                     structured.setdefault("level", "mid")
                
#                 selected_style = StyleSelectionOutput(**structured) if isinstance(structured, dict) else None
#             except Exception as e:
#                 print(f"Style selection error: {e}")
#                 selected_style = None

#         # Build the generation prompt
#         if selected_style and selected_style.selected_template:
#             prompt = (
#                 f"Using the '{selected_style.selected_template.get('style', 'professional')}' template and a '{selected_style.tone}' tone, "
#                 f"generate a personalized cover letter for {input_data.job_title} at {input_data.hiring_company}. "
#                 f"Here are the details:\n\n"
#                 f"Job Description: {input_data.job_description}\n"
#                 f"Applicant Name: {input_data.applicant_name}\n"
#                 f"Working Experience: {input_data.working_experience}\n"
#                 f"Qualifications: {input_data.qualifications}\n"
#                 f"Skillsets: {input_data.skillsets}\n"
#                 f"Company Culture Notes: {input_data.company_culture_notes}\n"
#             )
#             if selected_style.selected_template.get('content'):
#                 prompt += f"\nTemplate Content for reference:\n{selected_style.selected_template['content']}"
#         else:
#             # Fallback prompt without RAG
#             prompt = (
#                 f"Generate a professional cover letter for {input_data.job_title} at {input_data.hiring_company}. "
#                 f"Use the following information:\n\n"
#                 f"Job Description: {input_data.job_description}\n"
#                 f"Applicant Name: {input_data.applicant_name}\n"
#                 f"Working Experience: {input_data.working_experience}\n"
#                 f"Qualifications: {input_data.qualifications}\n"
#                 f"Skillsets: {input_data.skillsets}\n"
#                 f"Company Culture Notes: {input_data.company_culture_notes}\n"
#                 f"\nWrite a compelling, professional cover letter that highlights the candidate's relevant experience and enthusiasm for the role."
#             )

#         # Generate the letter using a simple text generation approach
#         generation_agent = Agent(
#             model=cover_letter_agent.model,
#             deps_type=str,
#             system_prompt="You are a professional cover letter writer. Generate clear, compelling cover letters based on the provided information."
#         )
#         # Add prompt from the context here
#         llm_result = await generation_agent.run(prompt, deps=prompt)
#         text = llm_result.data if hasattr(llm_result, "data") else str(llm_result)

#         # FIX: Ensure used_github_info is always a dictionary
#         github_info_dict = {}
#         if hasattr(input_data, 'github_username') and input_data.github_username:
#             github_info_dict = {"username": input_data.github_username}

#         return CoverLetterOutput(
#             cover_letter=text,
#             summary=f"Generated cover letter for {input_data.job_title} at {input_data.hiring_company}",
#             used_highlights=input_data.working_experience.split("; ") if input_data.working_experience else [],
#             used_github_info=github_info_dict  # Always pass a dictionary
#         )

#     except Exception as e:
#         print(f"Error in generate_with_style: {e}")
#         # Return a basic fallback cover letter
#         fallback_text = (
#             f"Dear Hiring Manager,\n\n"
#             f"I am writing to express my interest in the {input_data.job_title or 'position'} "
#             f"at {input_data.hiring_company or 'your company'}.\n\n"
#             f"With my background in {input_data.skillsets or 'relevant technologies'} "
#             f"and experience in {input_data.working_experience or 'the field'}, "
#             f"I believe I would be a valuable addition to your team.\n\n"
#             f"Thank you for considering my application.\n\n"
#             f"Sincerely,\n{input_data.applicant_name or 'Applicant'}"
#         )
        
#         # FIX: Ensure fallback also uses dictionary for used_github_info
#         github_info_dict = {}
#         if hasattr(input_data, 'github_username') and input_data.github_username:
#             github_info_dict = {"username": input_data.github_username}
        
#         return CoverLetterOutput(
#             cover_letter=fallback_text,
#             summary="Fallback cover letter generated due to processing error",
#             used_highlights=[],
#             used_github_info=github_info_dict  # Always pass a dictionary
#         )