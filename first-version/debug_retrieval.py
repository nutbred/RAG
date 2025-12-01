"""
Debug script to test retrieval performance from the command line.
Usage: python debug_retrieval.py <query>
"""

import sys
from dotenv import load_dotenv

load_dotenv()

from rag_system import RAGSystem

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_retrieval.py <query>")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    print("Loading RAG system...")
    rag = RAGSystem("config.yaml")
    
    if not rag.all_documents:
        print("Error: No documents have been ingested yet.")
        print("Please run pipeline.py first and ingest some documents.")
        sys.exit(1)
    
    print(f"Total documents indexed: {len(rag.all_documents)}")
    print(f"Total estimated tokens: {rag.total_tokens}")
    
    rag.debug_retrieval(query)

if __name__ == "__main__":
    main()
