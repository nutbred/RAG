# Hybrid RAG System with Intent Classification 

## Setup
1.  **Environment Variables**:
    Open `.env` and add your API keys:
    ```env
    GOOGLE_API_KEY="your_google_api_key"
    LLAMA_PARSE_API_KEY="your_llama_parse_api_key"
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Running the CLI Pipeline
To run the interactive command-line interface:
```
python pipeline.py
```
**Commands**:
- `ingest <path_to_pdf>`: Upload and process documents.
- `query <question>`: Ask a question based on ingested documents.
- `list`: Show all ingested files.
- `exit`: Quit the program.

## Running the API (testing)
```
python api.py
```
# First Version features:
- FAISS + BM25 search
- Intent Classification
- Parent Document Retrieval
