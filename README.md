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
```bash
python pipeline.py
```
**Commands**:
- `ingest <path_to_pdf>`: Upload and process documents.
- `query <question>`: Ask a question based on ingested documents.
- `list`: Show all ingested files.
- `exit`: Quit the program.

## Running the API
To start the FastAPI server:
```
python api.py
```
The server will run at `http://0.0.0.0:8000`.

**How `api.py` works**:
It wraps the RAG system in a FastAPI application with the following endpoints:
- `POST /ingest`: Accepts file uploads and processes them.
- `POST /query`: Accepts a JSON body `{"query": "..."}` and returns an answer.
- `GET /list`: Returns a list of ingested files.

## Running Tests
To run the API tests:
```bash
python test_modules/test_api.py
```
**How it works**:
The test file uses `FastAPI.testclient` to simulate requests to the running application without needing to start the server manually. It verifies that:
- The root endpoint (`/`) returns a success status.
- The `/list` endpoint returns a valid file list.
- The `/query` endpoint returns a valid response structure.
