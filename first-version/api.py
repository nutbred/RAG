import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from rag_system import RAGSystem

app = FastAPI(title="RAG System API")
rag = RAGSystem("config.yaml")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    query: str
    file_filters: Optional[List[str]] = None

class QueryResponse(BaseModel):
    answer: str
    intent: str
    mode: str

@app.get("/")
def read_root():
    return {"status": "RAG System is running"}

@app.post("/ingest")
async def ingest_files(files: List[UploadFile] = File(...)):
    """
    Ingest uploaded PDF files.
    """
    saved_paths = []
    try:
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(file_path)
        
        rag.ingest(saved_paths)
        
        return {
            "message": f"Successfully ingested {len(saved_paths)} files.",
            "files": [os.path.basename(p) for p in saved_paths]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system.
    """
    try:
        # Get intent for metadata
        intent = rag.intent_classifier.predict(request.query)
        
        response_text = rag.query(request.query, file_filters=request.file_filters)
        
        response_text = rag.query(request.query, file_filters=request.file_filters)
        
        return {
            "answer": response_text,
            "intent": intent,
            "mode": "Hybrid/Full" # Placeholder, or we can parse it from logs if we really wanted to.
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list")
def list_files():
    """
    List all ingested files.
    """
    return {"files": rag.list_ingested_files()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
