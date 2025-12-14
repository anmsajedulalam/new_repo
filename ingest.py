import os
import glob
from langchain_community.document_loaders import (
    TextLoader, 
    UnstructuredMarkdownLoader, 
    PyPDFLoader, 
    CSVLoader, 
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import config

def load_documents():
    """Loads text, markdown, pdf, csv, and excel files from the data directory."""
    documents = []
    
    # Mapping extensions to loaders
    loader_mapping = {
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".pdf": PyPDFLoader,
        ".csv": CSVLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".xls": UnstructuredExcelLoader
    }

    for ext, loader_cls in loader_mapping.items():
        files = glob.glob(os.path.join(config.DATA_PATH, f"*{ext}"))
        for file_path in files:
            try:
                print(f"Loading {file_path}...")
                if ext == ".md":
                    # Fallback/specific handling for MD if needed, but keeping simple for now
                    try:
                        loader = loader_cls(file_path)
                        documents.extend(loader.load())
                    except:
                        # Fallback to text loader for MD
                        loader = TextLoader(file_path, encoding='utf-8')
                        documents.extend(loader.load())
                elif ext == ".csv":
                    # CSV often needs encoding spec or args, but default is usually safe for simple files
                    loader = loader_cls(file_path, encoding='utf-8')
                    documents.extend(loader.load())
                else:
                    loader = loader_cls(file_path)
                    documents.extend(loader.load())
                
                print(f"Successfully loaded {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
    return documents

def ingest_data():
    """Main ingestion function."""
    print("Loading documents...")
    docs = load_documents()
    if not docs:
        print("No documents found in data directory.")
        return

    print(f"Splitting {len(docs)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Created {len(all_splits)} chunks.")

    print("Initializing Azure Embeddings...")
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=config.AZURE_EMBEDDING_DEPLOYMENT,
        openai_api_version=config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
    )

    print("Creating FAISS index...")
    try:
        vector_store = FAISS.from_documents(documents=all_splits, embedding=embeddings)
        
        print(f"Saving index to {config.INDEX_PATH}...")
        vector_store.save_local(config.INDEX_PATH)
        print("Ingestion complete!")
    except Exception as e:
        print(f"Error creating/saving vector store: {e}")
        print("Please check your Azure Configuration (Deployment Name/Key) in config.py")

def add_single_file(file_path):
    """
    Ingests a single file into the existing FAISS index.
    """
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"

    documents = []
    # Reuse loading logic (simplified mapping)
    loader_mapping = {
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".pdf": PyPDFLoader,
        ".csv": CSVLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".xls": UnstructuredExcelLoader
    }
    
    ext = os.path.splitext(file_path)[1]
    if ext not in loader_mapping:
        return f"Unsupported file type: {ext}"
    
    try:
        loader_cls = loader_mapping[ext]
        if ext == ".md":
             try:
                 loader = loader_cls(file_path)
                 documents.extend(loader.load())
             except:
                 loader = TextLoader(file_path, encoding='utf-8')
                 documents.extend(loader.load())
        else:
            loader = loader_cls(file_path)
            documents.extend(loader.load())
            
        # Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        new_chunks = text_splitter.split_documents(documents)
        
        # Load existing index
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=config.AZURE_EMBEDDING_DEPLOYMENT,
            openai_api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
        )
        
        if os.path.exists(config.INDEX_PATH):
            vector_store = FAISS.load_local(config.INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            vector_store.add_documents(new_chunks)
        else:
            vector_store = FAISS.from_documents(new_chunks, embeddings)
            
        vector_store.save_local(config.INDEX_PATH)
        return f"Successfully added {file_path} ({len(new_chunks)} chunks)."
        
    except Exception as e:
        return f"Error adding file: {e}"

if __name__ == "__main__":
    ingest_data()
