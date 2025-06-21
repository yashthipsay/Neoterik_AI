
from pydantic import BaseModel, Dict, List
from pydantic_ai import Agent, RunContext
from .models import (
    StyleSelectionInput,
    StyleSelectionOutput,
    CoverLetterTemplate,
    ToneGuideline,
    IndustryPhrase,
    SkillSnippet,
    CompanyValue,
    GlobalRAGData,
)
from dotenv import load_dotenv
load_dotenv()
from langchain.docstore.document import Document
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

import json
import os

os.environ["HF_TOKEN"] = "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"

# System prompt for style selection
STYLE_SYSTEM_PROMPT = """
You are an expert assistant for Neoterik AI, selecting the optimal cover letter style based on job details and user preferences. Analyze the provided job information and retrieved documents to choose the best template, tone, style, industry, and level. Ensure the selection aligns with the job description, company culture, and applicant’s experience level.
"""

cover_letter_agent = Agent(
    model="groq:deepseek-r1-distill-llama-70b",  # Google Gemini 2.0 Flash model for fast, quality generation
    deps=StyleSelectionInput,           # Input type dependency for the agent
    system_prompt=STYLE_SYSTEM_PROMPT,          # System prompt defining the agent's role
)

def load_rag_data(filepath: str) -> list[Document]:
    with open(filepath, 'r') as f:
        raw = json.load(f)
        data = GlobalRAGData(**raw)

    docs = []

    # Cover Letter Templates
    for template in data.templates:
        content = (
            f"Tone: {template.tone}\n"
            f"Industry: {template.industry}\n"
            f"Level: {template.level}\n"
            f"Style: {template.style}\n"
            f"Content: {template.content}"
        )
        metadata = {
            "type": "template",
            "tone": template.tone,
            "industry": template.industry,
            "level": template.level,
            "style": template.style,
            **(template.metadata or {}),
        }
        docs.append(Document(page_content=content, metadata=metadata))

    # Tone Guidelines
    for tone in data.tones:
        content = f"Tone: {tone.tone}\nDescription: {tone.description}\nExample: {tone.example}"
        docs.append(Document(page_content=content, metadata={"type": "tone", "tone": tone.tone}))

    # Industry Phrases
    for phr in data.phrases:
        content = f"Industry: {phr.industry}\nPhrases: {'; '.join(phr.phrases)}"
        docs.append(Document(page_content=content, metadata={"type": "phrase", "industry": phr.industry}))

    # Skill Snippets
    for skill in data.skills:
        content = f"Skill: {skill.skill}\nDescription: {skill.description}"
        docs.append(Document(page_content=content, metadata={"type": "skill", "skill": skill.skill}))

    # Company Values
    for val in data.values:
        content = f"Value: {val.value}\nExample: {val.example_sentence}"
        docs.append(Document(page_content=content, metadata={"type": "value", "value": val.value}))

    return docs

# Load and flatten the data
# Use the correct file name for RAG data
RAG_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "cover_letter_templates.json")
documents = load_rag_data(RAG_DATA_PATH)

# Optional: split large documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# Embed and store in Chroma
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vectorstore = Chroma.from_documents(chunks, embedding=embed_model)
retriever = vectorstore.as_retriever()

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embed_model,
    persist_directory="rag_store",
    collection_name="neoterik_rag"
)


