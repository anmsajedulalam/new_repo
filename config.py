import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Creating a robust config that can be easily updated by the user
    
    # Azure OpenAI Settings
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    # Based on the user provided truncated URL 'gpt-4.1-min...' and 'text-embedd...', 
    # and standard Azure naming conventions.
    # User might need to update these if they fail.
    AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4.1-mini") # Guessing based on "gpt-4.1-mini"

    AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002") # Standard default
    
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview") # Required for structured output
    
    # Optional: Tavily API Key for Web Search Fallback
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    # Vector Store Settings
    INDEX_PATH = "faiss_index"
    DATA_PATH = "data"

config = Config()
