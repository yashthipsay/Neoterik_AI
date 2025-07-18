import os
from dotenv import load_dotenv
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# === Load environment variables ===
load_dotenv()

# === CONFIG ===
CHROMA_API_KEY = 'ck-J8EfrmcW4TPquPQW4CVshQDWtqSBDtAz7KHgDRmXLtxy'  # ⚠️ Avoid hardcoding credentials
CHROMA_TENANT = 'f1e0c47a-7d1c-4e4c-9bb6-df2879d6a1d8'
CHROMA_DATABASE = 'neoterik-rag-test'
COLLECTION_NAME = "cover-letter-templates"

# === Initialize Chroma Cloud Client ===
def get_chroma_client():
    return chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE
    )

# === Load embedding model ===
embed_model = HuggingFaceEmbeddings(model_name="lightonai/modernbert-embed-large")

# === Connect to Chroma Cloud vectorstore ===
client = get_chroma_client()
db = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embed_model,
    client=client
)

# === Define your query ===
query = (
    "Experienced backend developer seeking a DevOps engineering role. "
    "Generate a concise cover letter template emphasizing cloud infrastructure experience, "
    "CI/CD expertise, and team collaboration on scalable systems."
)

# === Run similarity search with metadata filter ===
results = db.similarity_search(query=query, k=3, filter={"type": "template"})

# === Output collection document count ===
print("DB documents count:", len(db.get()["documents"]))

# === Output results ===
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print("Metadata:", doc.metadata)
    print("Content Preview:", doc.page_content[:500], "...\n")