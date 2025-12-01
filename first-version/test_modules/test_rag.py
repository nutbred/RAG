import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock missing modules BEFORE importing rag_system
sys.modules['langchain_huggingface'] = MagicMock()
sys.modules['langchain_chroma'] = MagicMock()
sys.modules['langchain_community.retrievers'] = MagicMock()
sys.modules['langchain.retrievers'] = MagicMock()
sys.modules['langchain.schema'] = MagicMock()
sys.modules['langchain.text_splitter'] = MagicMock()
sys.modules['langchain_google_genai'] = MagicMock()
sys.modules['langchain.storage'] = MagicMock()
sys.modules['chromadb'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['langchain_community.vectorstores'] = MagicMock()

# Mock rag_modules
sys.modules['rag_modules'] = MagicMock()

# Add current dir to path
sys.path.append(os.getcwd())

# Now import rag_system
from rag_system import RAGSystem, Config

class TestRAGSystem(unittest.TestCase):
    def setUp(self):
        # Mock Config
        self.config_patcher = patch('rag_system.Config.load')
        self.mock_config_load = self.config_patcher.start()
        self.mock_config = MagicMock()
        self.mock_config.threshold_ratio = 0.2
        self.mock_config.chunk_size = 100
        self.mock_config.chunk_overlap = 10
        self.mock_config.llm_model_name = "gemini-test"
        self.mock_config.llm_api_key_env = "GOOGLE_API_KEY" # Added
        self.mock_config.parsing_api_key_env = "LLAMA_PARSE_API_KEY" # Added
        self.mock_config.embedding_model_name = "all-MiniLM-L6-v2"
        self.mock_config.embedding_device = "cpu"
        self.mock_config.retrieval_k = 2
        self.mock_config_load.return_value = self.mock_config

        # Mock internal components that are instantiated in __init__
        self.embeddings_patcher = patch('rag_system.HuggingFaceEmbeddings')
        self.embeddings_patcher.start()
        
        # Mock Modules
        self.intent_classifier_patcher = patch('rag_system.IntentClassifier')
        self.mock_intent_classifier = self.intent_classifier_patcher.start()
        # Default to Retrieval intent
        self.mock_intent_classifier.return_value.predict.return_value = "Retrieval"
        
        self.llm_client_patcher = patch('rag_system.LLMClient')
        self.mock_llm_client = self.llm_client_patcher.start()
        
        self.vector_db_client_patcher = patch('rag_system.VectorDBClient')
        self.mock_vector_db_client = self.vector_db_client_patcher.start()
        
        # Mock os.path.exists
        self.os_path_patcher = patch('os.path.exists')
        self.mock_os_path = self.os_path_patcher.start()
        self.mock_os_path.return_value = False
        
        self.splitter_patcher = patch('rag_system.RecursiveCharacterTextSplitter')
        self.splitter_patcher.start()
        
        self.store_patcher = patch('rag_system.InMemoryStore')
        self.store_patcher.start()
        
        self.parent_retriever_patcher = patch('rag_system.ParentDocumentRetriever')
        self.parent_retriever_patcher.start()

    def tearDown(self):
        self.config_patcher.stop()
        self.embeddings_patcher.stop()
        self.intent_classifier_patcher.stop()
        self.llm_client_patcher.stop()
        self.vector_db_client_patcher.stop()
        self.os_path_patcher.stop()
        self.splitter_patcher.stop()
        self.store_patcher.stop()
        self.parent_retriever_patcher.stop()

    def test_threshold_logic_low_tokens(self):
        rag = RAGSystem()
        rag.total_tokens = 1000 # Low tokens
        rag._query_full_context = MagicMock(return_value="Full Context Response")
        rag._query_rag = MagicMock(return_value="RAG Response")
        
        response = rag.query("test")
        
        rag._query_full_context.assert_called_once()
        rag._query_rag.assert_not_called()
        self.assertEqual(response, "Full Context Response")

    def test_threshold_logic_high_tokens(self):
        rag = RAGSystem()
        rag.total_tokens = 500000 # High tokens (above 200k threshold)
        rag._query_full_context = MagicMock(return_value="Full Context Response")
        rag._query_rag = MagicMock(return_value="RAG Response")
        
        response = rag.query("test")
        
        rag._query_rag.assert_called_once()
        rag._query_full_context.assert_not_called()
        self.assertEqual(response, "RAG Response")

    def test_intent_hardcoded_chat(self):
        rag = RAGSystem()
        # Mock intent to Hardcoded_Chat
        self.mock_intent_classifier.return_value.predict.return_value = "Hardcoded_Chat"
        
        response = rag.query("Hello")
        
        self.assertEqual(response, "Hi! How can you help you today? Need help with your documents?")
        self.mock_llm_client.return_value.invoke.assert_not_called()

    def test_intent_ml_chat(self):
        rag = RAGSystem()
        # Mock intent to ML_Chat
        self.mock_intent_classifier.return_value.predict.return_value = "ML_Chat"
        self.mock_llm_client.return_value.invoke.return_value.content = "I am fine."
        
        response = rag.query("How are you?")
        
        self.assertEqual(response, "I am fine.")
        # Should call LLM directly
        self.mock_llm_client.return_value.invoke.assert_called_with("How are you?")

if __name__ == '__main__':
    unittest.main()
