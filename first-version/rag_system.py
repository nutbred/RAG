import os
import yaml
import json
import pickle
import warnings
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

import parser  
from rag_modules import IntentClassifier, LLMClient, VectorDBClient

@dataclass
class Config:
    threshold_ratio: float
    chunk_size: int
    chunk_overlap: int
    llm_model_name: str
    llm_api_key_env: str
    embedding_model_name: str
    embedding_device: str
    retrieval_k: int
    parsing_api_key_env: str

    @classmethod
    def load(cls, path: str = "config.yaml"):
        with open(path, "r") as f:
            config_data = yaml.safe_load(f)
        return cls(
            threshold_ratio=config_data["system"]["threshold_ratio"],
            chunk_size=config_data["system"]["chunk_size"],
            chunk_overlap=config_data["system"]["chunk_overlap"],
            llm_model_name=config_data["llm"]["model_name"],
            llm_api_key_env=config_data["llm"]["api_key_env"],
            embedding_model_name=config_data["embedding"]["model_name"],
            embedding_device=config_data["embedding"]["device"],
            retrieval_k=config_data["retrieval"]["k"],
            parsing_api_key_env=config_data["parsing"]["api_key_env"]
        )

class RAGSystem:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = Config.load(config_path)
        self._setup_environment()
        
        self.intent_classifier = IntentClassifier()
        self.llm_client = LLMClient(self.config.llm_model_name, self.config.llm_api_key_env)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model_name,
            model_kwargs={'device': self.config.embedding_device}
        )
        
        self.index_path = "faiss_index"
        self.vector_db_client = VectorDBClient(
            index_path=self.index_path,
            embedding_function=self.embeddings
        )
        self.vectorstore = self.vector_db_client.vectorstore
        
        self.docstore = InMemoryStore() # For Parent Document Retrieval
        
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        self.parent_retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=self.child_splitter,
            # parent_splitter=None, # We will feed parent documents directly
        )
        
        self.bm25_retriever = None # Initialized after ingestion
        self.ensemble_retriever = None
        
        self.total_tokens = 0
        self.all_documents = [] 
        self.docs_path = "documents.pkl"
        
        self._load_state()
        if self.all_documents:
            self.bm25_retriever = BM25Retriever.from_documents(self.all_documents)
            self.bm25_retriever.k = self.config.retrieval_k
            
            # Initialize Ensemble if we have both
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.parent_retriever, self.bm25_retriever],
                weights=[0.5, 0.5]
            )
        else:
            self.bm25_retriever = None
            self.ensemble_retriever = None

    def _save_state(self):
        state = {
            "documents": self.all_documents,
            "total_tokens": self.total_tokens
        }
        with open(self.docs_path, "wb") as f:
            pickle.dump(state, f)
        print(f"State saved to {self.docs_path}")

    def _load_state(self):
        if os.path.exists(self.docs_path):
            try:
                with open(self.docs_path, "rb") as f:
                    state = pickle.load(f)
                self.all_documents = state.get("documents", [])
                self.total_tokens = state.get("total_tokens", 0)
                print(f"Loaded {len(self.all_documents)} documents from {self.docs_path}")
            except Exception as e:
                print(f"Error loading state: {e}")
                self.all_documents = []
                self.total_tokens = 0

    def list_ingested_files(self) -> List[str]:
        """Return a list of unique source files ingested"""
        sources = set()
        for doc in self.all_documents:
            sources.add(doc.metadata.get("source", "Unknown"))
        return sorted(list(sources))

    def _setup_environment(self):
        if not os.environ.get(self.config.parsing_api_key_env):
            print(f"Warning: {self.config.parsing_api_key_env} not set.")
        if not os.environ.get(self.config.llm_api_key_env):
            print(f"Warning: {self.config.llm_api_key_env} not set.")

    def ingest(self, file_paths: List[str]):
        print(f"Starting ingestion for {len(file_paths)} files...")
        
        results = parser.parse_multiple_paths_parallel(file_paths)
        
        new_documents = []
        
        for path, (data, tokens, all_text) in results.items():
            if isinstance(data, str) and data.startswith("Error"):
                print(f"Skipping {path}: {data}")
                continue
                
            self.total_tokens += tokens
            try:
                pages = data.pages
            except AttributeError:
                if isinstance(data, dict) and "pages" in data:
                    pages = data["pages"]
                else:
                    pages = [{"text": all_text, "page_number": 1}]
            
            for i, page in enumerate(pages):
                page_text = ""
                if hasattr(page, "text"):
                    page_text = parser.remove_footer(page.text)
                elif isinstance(page, dict):
                    page_text = parser.remove_footer(page.get("text", ""))
                
                if page_text.strip():
                    doc = Document(
                        page_content=page_text,
                        metadata={"source": path, "page": i + 1}
                    )
                    new_documents.append(doc)
        
        if not new_documents:
            print("No new documents to ingest.")
            return

        self.all_documents.extend(new_documents)
        
        self.parent_retriever.add_documents(new_documents)
        
        self.vector_db_client.save()
        
        self.bm25_retriever = BM25Retriever.from_documents(self.all_documents)
        self.bm25_retriever.k = self.config.retrieval_k
        
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.parent_retriever, self.bm25_retriever],
            weights=[0.5, 0.5] # Equal weight
        )
        
        self._save_state()
        
        print(f"Ingestion complete. Total estimated tokens: {self.total_tokens}")

    def query(self, user_query: str, file_filters: List[str] = None) -> str:
        # 1. Check Intent
        intent = self.intent_classifier.predict(user_query)
        print(f"Detected Intent: {intent}")
        
        if intent == "Hardcoded_Chat":
            return "Hi! How can you help you today? Need help with your documents?"
        
        if intent == "ML_Chat":
            return self.llm_client.invoke(user_query).content

        if file_filters:
            filtered_docs = self._get_filtered_docs(file_filters)
            current_tokens = int(sum(len(doc.page_content) / 4 for doc in filtered_docs))
        else:
            current_tokens = self.total_tokens
            filtered_docs = self.all_documents

        MAX_WINDOW = 1000000 
        threshold = MAX_WINDOW * self.config.threshold_ratio
        
        print(f"Current tokens (filtered): {current_tokens}, Threshold: {threshold}")
        
        if current_tokens < threshold:
            print("Mode: Full Context")
            return self._query_full_context(user_query, filtered_docs)
        else:
            print("Mode: RAG (Hybrid)")
            return self._query_rag(user_query, file_filters)

    def _get_filtered_docs(self, file_filters: List[str]) -> List[Document]:
        filtered = []
        for doc in self.all_documents:
            source = doc.metadata.get("source", "")
            if any(f in source for f in file_filters):
                filtered.append(doc)
        return filtered

    def _query_full_context(self, user_query: str, docs: List[Document]) -> str:
        context = ""
        for doc in docs:
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", "unknown")
            context += f"--- Source: {source}, Page: {page} ---\n{doc.page_content}\n\n"
        
        prompt = f"""
        You are a helpful assistant. Answer the user's question based on the following context.
        Always cite your sources using the format [Source: filename, Page: number].

        
        Context:
        {context}
        
        Question: {user_query}
        """
        
        response = self.llm_client.invoke(prompt)
        return response.content

    def _query_rag(self, user_query: str, file_filters: List[str] = None) -> str:
        if not self.ensemble_retriever:
            return "Error: No documents indexed."
            
        docs = self.ensemble_retriever.invoke(user_query)
        
        if file_filters:
            filtered_docs = []
            for doc in docs:
                source = doc.metadata.get("source", "")
                if any(f in source for f in file_filters):
                    filtered_docs.append(doc)
            docs = filtered_docs
            
        if not docs:
            return "No relevant documents found in the selected files."
        
        context = ""
        for doc in docs:
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", "unknown")
            context += f"--- Source: {source}, Page: {page} ---\n{doc.page_content}\n\n"
        
        prompt = f"""
        You are a helpful assistant. Answer the user's question based on the following retrieved context.
        Always cite your sources using the format [Source: filename, Page: number].
        
        Context:
        {context}
        
        Question: {user_query}
        """
        
        response = self.llm_client.invoke(prompt)
        return response.content

    def debug_retrieval(self, user_query: str):
        """Debug retrieval performance by showing vector and BM25 results"""
        print(f"\n{'='*60}")
        print(f"DEBUG RETRIEVAL: '{user_query}'")
        
        intent = self.intent_classifier.predict(user_query)
        print(f"Detected Intent: {intent}")
        
        if intent in ["Hardcoded_Chat", "ML_Chat"]:
            print("Intent is Chat. Skipping retrieval debug output.")
            print(f"{'='*60}")
            return

        print(f"{'='*60}")
        
        print("\n[Vector Search Results]")
        print("-" * 60)
        try:
            vector_results = self.vectorstore.similarity_search_with_score(user_query, k=self.config.retrieval_k)
            for i, (doc, score) in enumerate(vector_results):
                source = os.path.basename(doc.metadata.get("source", "unknown"))
                page = doc.metadata.get("page", "unknown")
                print(f"\n{i+1}. L2 Distance: {score:.4f}")
                print(f"   Source: {source} | Page: {page}")
                print(f"   Content: {doc.page_content[:150]}...")
        except Exception as e:
            print(f"Error during vector search: {e}")

        print("\n\n[BM25 Search Results]")
        print("-" * 60)
        if self.bm25_retriever:
            try:
                bm25_results = self.bm25_retriever.invoke(user_query)
                for i, doc in enumerate(bm25_results):
                    source = os.path.basename(doc.metadata.get("source", "unknown"))
                    page = doc.metadata.get("page", "unknown")
                    print(f"\n{i+1}. Source: {source} | Page: {page}")
                    print(f"   Content: {doc.page_content[:150]}...")
            except Exception as e:
                print(f"Error during BM25 search: {e}")
        else:
            print("BM25 Retriever not initialized.")
        
        print("\n\n[Ensemble (Combined) Results]")
        print("-" * 60)
        if self.ensemble_retriever:
            try:
                ensemble_results = self.ensemble_retriever.invoke(user_query)
                for i, doc in enumerate(ensemble_results):
                    source = os.path.basename(doc.metadata.get("source", "unknown"))
                    page = doc.metadata.get("page", "unknown")
                    print(f"\n{i+1}. Source: {source} | Page: {page}")
                    print(f"   Content: {doc.page_content[:150]}...")
            except Exception as e:
                print(f"Error during ensemble search: {e}")
        
        print(f"\n{'='*60}\n")
        
if __name__ == "__main__":
    pass
