import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "RAG System is running"}

def test_list_files():
    response = client.get("/list")
    assert response.status_code == 200
    assert "files" in response.json()
    print("List files response:", response.json())

def test_query_chat():
    # Test chat intent
    payload = {"query": "Hello"}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    print("Chat response:", data)
    assert "answer" in data
    # Intent might be Hardcoded_Chat or Chat depending on classifier
    # assert data["intent"] in ["Hardcoded_Chat", "Chat"]

if __name__ == "__main__":
    print("Running API tests...")
    test_read_root()
    test_list_files()
    test_query_chat()
    print("API tests passed!")
