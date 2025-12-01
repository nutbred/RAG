import langchain
import sys

print(f"Langchain version: {langchain.__version__}")
print(f"Langchain file: {langchain.__file__}")
print(f"Langchain dir: {dir(langchain)}")

try:
    import langchain.retrievers
    print("langchain.retrievers imported successfully")
    print(f"langchain.retrievers dir: {dir(langchain.retrievers)}")
except ImportError as e:
    print(f"Failed to import langchain.retrievers: {e}")

try:
    import langchain_community.retrievers
    print("langchain_community.retrievers imported successfully")
    print(f"langchain_community.retrievers dir: {dir(langchain_community.retrievers)}")
except ImportError as e:
    print(f"Failed to import langchain_community.retrievers: {e}")
