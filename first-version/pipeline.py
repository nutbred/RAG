from rag_system import RAGSystem
import os
import shlex
from dotenv import load_dotenv
load_dotenv()

def main():
    rag = RAGSystem("config.yaml")
    
    print("--- RAG System Initialized ---")
    
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please set it before running.")
        return

    if not os.environ.get("LLAMA_PARSE_API_KEY"):
        print("Warning: LLAMA_PARSE_API_KEY not found. Parsing might fail.")

    is_ingesting = False
    
    while True:
        print("\nCommands:")
        print("  ingest <path1> [path2] [path3] ... - Add one or more documents")
        print("  list                    - List all ingested files")
        print("  query <text>            - Query all files")
        print("  query --files f1,f2 <text> - Query specific files (partial filename match)")
        print("  debug <text>            - Debug retrieval for a query")
        print("  exit                    - Exit")
        
        command = input("\n> ").strip()
        
        if command.startswith("ingest "):
            paths_str = command[7:].strip()
            try:
                raw_paths = shlex.split(paths_str, posix=False)
                paths = [p.strip('"').strip("'") for p in raw_paths]
            except ValueError as e:
                print(f"Error parsing paths: {e}")
                continue
            
            valid_paths = []
            for path in paths:
                if os.path.exists(path):
                    valid_paths.append(path)
                else:
                    print(f"Warning: File not found: {path}")
            
            if not valid_paths:
                print("No valid files to ingest.")
                continue
            
            is_ingesting = True
            print(f"\nIngesting {len(valid_paths)} file(s)...")
            print("Please wait, queries are blocked until ingestion completes.")
            
            try:
                rag.ingest(valid_paths)
                print(f"Successfully ingested {len(valid_paths)} file(s).")
            except Exception as e:
                print(f"Error during ingestion: {e}")
            finally:
                is_ingesting = False
                
        elif command == "list":
            files = rag.list_ingested_files()
            if files:
                print("\nIngested files:")
                for i, f in enumerate(files, 1):
                    print(f"  {i}. {os.path.basename(f)}")
            else:
                print("No files ingested yet.")
                
        elif command.startswith("debug "):
            if is_ingesting:
                print("Error: Ingestion in progress. Please wait until it completes.")
                continue
                
            query_text = command[6:].strip()
            rag.debug_retrieval(query_text)
                
        elif command.startswith("query "):
            if is_ingesting:
                print("Error: Ingestion in progress. Please wait until it completes.")
                continue
                
            if "--files " in command:
                parts = command.split("--files ", 1)
                after_flag = parts[1]
                file_part, query_part = after_flag.split(" ", 1) if " " in after_flag else (after_flag, "")
                file_filters = [f.strip() for f in file_part.split(",")]
                query_text = query_part.strip()
                
                if not query_text:
                    print("Error: No query provided after --files")
                    continue
                    
                print(f"\nQuerying files matching: {file_filters}")
                response = rag.query(query_text, file_filters=file_filters)
            else:
                query_text = command[6:].strip()
                response = rag.query(query_text)
            
            print(f"\n{'='*60}")
            print("RESPONSE:")
            print(f"{'='*60}")
            print(response)
            print(f"{'='*60}\n")
            
        elif command == "exit":
            break
        else:
            print("Unknown command.")

if __name__ == "__main__":
    main()
