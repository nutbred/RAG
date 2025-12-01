import os
import warnings
from typing import List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!")

import joblib
from sentence_transformers import SentenceTransformer

class IntentClassifier:
    def __init__(self):
        self.chit_chat_keywords = {kw.lower() for kw in ["hi", "hello", "hola", "hey", "hi there", "hello there", "hey there"]}
        self.ml_model_path = "intent_router.pkl"
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        if os.path.exists(self.ml_model_path):
            self.classifier = joblib.load(self.ml_model_path)
        else:
            print(f"Warning: {self.ml_model_path} not found. ML classification disabled.")
            self.classifier = None

    def predict(self, query: str) -> str:
        """
        Predicts the intent of the query.
        Returns: "Hardcoded_Chat", "ML_Chat", or "Retrieval"
        """
        cleaned_query = query.strip().lower()
        cleaned_no_punct = cleaned_query.rstrip("!?.")
        
        if cleaned_query in self.chit_chat_keywords or cleaned_no_punct in self.chit_chat_keywords:
            return "Hardcoded_Chat"

        if self.classifier:
            try:
                query_embedding = self.encoder.encode([query])
                prediction = self.classifier.predict(query_embedding)[0]
                prob = self.classifier.predict_proba(query_embedding)[0]
                if max(prob) < 0.7:
                    return "Retrieval"
                
                ml_intent = "Retrieval" if prediction == 1 else "Chat"
                
                if ml_intent == "Chat":
                    return "ML_Chat"
                else:
                    return "Retrieval"
            except Exception as e:
                print(f"Error in ML classification: {e}")
                return "Retrieval"
        
        return "Retrieval"

class LLMClient:
    def __init__(self, model_name: str, api_key_env: str):
        self.model_name = model_name
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            print(f"Warning: {api_key_env} not set.")
        
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            convert_system_message_to_human=True
        )

    def invoke(self, prompt: str) -> Any:
        return self.llm.invoke(prompt)

class VectorDBClient:
    def __init__(self, index_path: str, embedding_function: Any, embedding_size: int = 384):
        self.index_path = index_path
        self.embeddings = embedding_function
        self.embedding_size = embedding_size
        self.vectorstore = self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self.index_path):
            return FAISS.load_local(
                self.index_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            index = faiss.IndexFlatL2(self.embedding_size)
            return FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )

    def add_documents(self, documents: List[Any]):
        self.vectorstore.add_documents(documents)

    def save(self):
        self.vectorstore.save_local(self.index_path)

    def similarity_search_with_score(self, query: str, k: int = 5):
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def as_retriever(self):
        return self.vectorstore
