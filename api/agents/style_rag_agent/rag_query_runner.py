from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
import os
# === CONFIG ===
persist_directory = "rag_store"  # Update this to match your setup
collection_name = "neoterik_rag"  # Optional unless you use multiple collections

# === Load embedding model ===
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# === Load persisted Chroma vectorstore ===
db = Chroma(
    persist_directory=persist_directory,
    collection_name=collection_name,
    embedding_function=embed_model
)

os.environ["HF_TOKEN"] = "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"
# === Define your query ===
query = "Biology student transitioning to data analyst career. Create a light-weight cover letter template for a resume that highlights transferable skills and relevant coursework."

# === Run similarity search with metadata filter ===
results = db.similarity_search(
    query=query,
    k=3,  # Top N matches
    filter={"type": "template"}  # Optional metadata filter
)

print("DB documents count:", len(db.get()["documents"]))


# === Output results ===
for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print("Metadata:", doc.metadata)
    print("Content Preview:", doc.page_content[:250], "...\n")