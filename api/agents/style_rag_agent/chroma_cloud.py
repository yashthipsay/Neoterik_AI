import os
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.api.models.Collection import Collection
import chromadb
from chromadb.config import Settings
# === Load env ===
load_dotenv()

# === Config ===
CHROMA_API_KEY = os.getenv("ck-J8EfrmcW4TPquPQW4CVshQDWtqSBDtAz7KHgDRmXLtxy")
CHROMA_TENANT = os.getenv("f1e0c47a-7d1c-4e4c-9bb6-df2879d6a1d8")
CHROMA_DATABASE = os.getenv("neoterik-rag-test")
COLLECTION_NAME = "cover-letter-templates"
RAG_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "cover_letter_templates.json")

# === Initialize Chroma Cloud Client ===
def get_chroma_client():
    return chromadb.CloudClient(
        api_key='ck-J8EfrmcW4TPquPQW4CVshQDWtqSBDtAz7KHgDRmXLtxy',
        tenant='f1e0c47a-7d1c-4e4c-9bb6-df2879d6a1d8',
        database='neoterik-rag-test'
    )

# === Get or create collection ===
def get_chroma_collection(client) -> Collection:
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.get_or_create_collection(name=COLLECTION_NAME)

# === Load and flatten the data ===
def load_rag_data(filepath: str) -> list[Document]:
    with open(filepath, 'r') as f:
        raw = json.load(f)

    docs = []
    # --- Templates ---
    for template in raw.get("templates", []):
        if not template.get("tone"):
            continue  # skip empty or malformed
        content = (
            f"Tone: {template.get('tone','')}\n"
            f"Industry: {template.get('industry','')}\n"
            f"Level: {template.get('level','')}\n"
            f"Style: {template.get('style','')}\n"
            f"Keywords: {', '.join(template.get('keywords', []))}\n"
            f"Content: {template.get('content','')}"
        )
        # Flatten keywords for metadata (comma-separated string)
        keywords = template.get("keywords", [])
        if isinstance(keywords, list):
            keywords_str = ", ".join(keywords)
        else:
            keywords_str = str(keywords) if keywords else ""
        # Flatten metadata values to str/int/float/bool/None
        meta = {k: (", ".join(v) if isinstance(v, list) else v) for k, v in (template.get("metadata", {}) or {}).items()}
        metadata = {
            "type": "template",
            "tone": template.get("tone"),
            "industry": template.get("industry"),
            "level": template.get("level"),
            "style": template.get("style"),
            "keywords": keywords_str,
            **meta,
        }
        docs.append(Document(page_content=content, metadata=metadata))

    # --- Tones ---
    for tone in raw.get("tones", []):
        content = (
            f"Tone: {tone.get('tone','')}\n"
            f"Description: {tone.get('description','')}\n"
            f"Keywords: {', '.join(tone.get('keywords', [])) if 'keywords' in tone else ''}\n"
            f"Example: {tone.get('example','')}"
        )
        keywords = tone.get("keywords", [])
        if isinstance(keywords, list):
            keywords_str = ", ".join(keywords)
        else:
            keywords_str = str(keywords) if keywords else ""
        metadata = {
            "type": "tone",
            "tone": tone.get("tone"),
            "keywords": keywords_str,
        }
        docs.append(Document(page_content=content, metadata=metadata))

    # --- Phrases ---
    for phrase in raw.get("phrases", []):
        content = (
            f"Industry: {phrase.get('industry','')}\n"
            f"Phrases: {', '.join(phrase.get('phrases', []))}"
        )
        phrases = phrase.get("phrases", [])
        if isinstance(phrases, list):
            phrases_str = ", ".join(phrases)
        else:
            phrases_str = str(phrases) if phrases else ""
        metadata = {
            "type": "phrase",
            "industry": phrase.get("industry"),
            "phrases": phrases_str,
        }
        docs.append(Document(page_content=content, metadata=metadata))

    # --- Skills ---
    for skill in raw.get("skills", []):
        content = (
            f"Skill: {skill.get('skill','')}\n"
            f"Description: {skill.get('description','')}"
        )
        metadata = {
            "type": "skill",
            "skill": skill.get("skill"),
        }
        docs.append(Document(page_content=content, metadata=metadata))

    # --- Values ---
    for value in raw.get("values", []):
        content = (
            f"Value: {value.get('value','')}\n"
            f"Example: {value.get('example_sentence','')}"
        )
        metadata = {
            "type": "value",
            "value": value.get("value"),
        }
        docs.append(Document(page_content=content, metadata=metadata))

    return docs


# === Chunk documents ===
def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# === Upload to Chroma ===
def store_documents_in_chroma(chunks: list[Document]):
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    client = get_chroma_client()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embed_model,
        collection_name=COLLECTION_NAME,
        client=client
    )
    print(f"✅ Uploaded {len(chunks)} document chunks to Chroma Cloud.")

# === Main ===
if __name__ == "__main__":
    print("🔗 Connecting to Chroma Cloud...")
    client = get_chroma_client()
    collection = get_chroma_collection(client)
    print(f"✅ Connected to collection: {collection.name}")

    print("📄 Loading and chunking documents...")
    documents = load_rag_data(RAG_DATA_PATH)
    chunks = chunk_documents(documents)

    print("🚀 Uploading to Chroma...")
    store_documents_in_chroma(chunks)
