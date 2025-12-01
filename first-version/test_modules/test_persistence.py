from rag_system import RAGSystem
import os
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Clean up previous state
if os.path.exists("documents.pkl"):
    os.remove("documents.pkl")
if os.path.exists("faiss_index"):
    shutil.rmtree("faiss_index")

print("--- Test 1: Ingestion and Save ---")
rag1 = RAGSystem()
# Create a dummy document manually to avoid API calls
from langchain.schema import Document
doc = Document(page_content="Test content", metadata={"source": "test.pdf", "page": 1})
rag1.all_documents.append(doc)
rag1.total_tokens = 100
rag1._save_state()
print("State saved.")

print("\n--- Test 2: Load on Restart ---")
rag2 = RAGSystem()
print(f"Loaded documents: {len(rag2.all_documents)}")
print(f"Loaded tokens: {rag2.total_tokens}")

if len(rag2.all_documents) == 1 and rag2.total_tokens == 100:
    print("SUCCESS: State persisted correctly.")
else:
    print("FAILURE: State not persisted.")

# Clean up
if os.path.exists("documents.pkl"):
    os.remove("documents.pkl")
