import sys

print("Checking retrievers...")

try:
    from langchain.retrievers import ParentDocumentRetriever
    print("Found ParentDocumentRetriever in langchain.retrievers")
except ImportError as e:
    print(f"ParentDocumentRetriever not found in langchain.retrievers: {e}")

try:
    from langchain_community.retrievers import ParentDocumentRetriever
    print("Found ParentDocumentRetriever in langchain_community.retrievers")
except ImportError as e:
    print(f"ParentDocumentRetriever not found in langchain_community.retrievers: {e}")

try:
    from langchain.retrievers import EnsembleRetriever
    print("Found EnsembleRetriever in langchain.retrievers")
except ImportError as e:
    print(f"EnsembleRetriever not found in langchain.retrievers: {e}")

try:
    from langchain.retrievers.ensemble import EnsembleRetriever
    print("Found EnsembleRetriever in langchain.retrievers.ensemble")
except ImportError as e:
    print(f"EnsembleRetriever not found in langchain.retrievers.ensemble: {e}")

try:
    from langchain_community.retrievers import EnsembleRetriever
    print("Found EnsembleRetriever in langchain_community.retrievers")
except ImportError as e:
    print(f"EnsembleRetriever not found in langchain_community.retrievers: {e}")
