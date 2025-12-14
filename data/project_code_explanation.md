# System Code Explanation

This document explains the inner workings of the Agentic RAG system script-by-script.

## 1. Configuration (`config.py`)
This file acts as the central control room for the project.
*   **Purpose**: Manages API keys, endpoint URLs, and deployment names for Azure OpenAI.
*   **Key Logic**:
    *   It reads from environment variables (using `os.getenv`).
    *   It sets defaults (e.g., `gpt-4.1-mini`) if variables aren't found.
    *   It defines paths for data storage (`data/`) and the vector index (`faiss_index/`).

## 2. Data Ingestion (`ingest.py`)
This script processes your documents so the AI can "read" them later.
*   **Step 1: Loading**: It uses specific loaders for different file types:
    *   `TextLoader` for `.txt`
    *   `PyPDFLoader` for `.pdf`
    *   `CSVLoader` for `.csv`
    *   `UnstructuredExcelLoader` for `.xlsx`
*   **Step 2: Splitting**: It chops large documents into smaller pieces ("chunks") of 1000 characters. This is crucial because LLMs have a limit on how much text they can read at once.
*   **Step 3: Embedding**: It sends these chunks to Azure's Embedding model (`text-embedding-ada-002`), which converts text into lists of numbers (vectors).
*   **Step 4: Indexing**: It saves these vectors into a FAISS index locally. This allows us to search for "similar concepts" later mathematically.

*   **Dynamic Ingestion (`add_single_file`)**:
    *   This specific function allows adding a single file on-the-fly without keeping the main process waiting for a full re-index. it loads the existing index, adds the new file's chunks, and saves it back.

## 3. The Agent (`agent.py`)
This is the brain of the operation, built with **LangGraph**. It defines a workflow graph where data flows between "Nodes".

### The State
We define an `AgentState` that tracks:
*   `question`: The user's query.
*   `documents`: A list of retrieved text chunks.
*   `messages`: The full history of the conversation (for memory).

### The Nodes (Steps)
1.  **`retrieve`**:
    *   Takes the user's question.
    *   Searches the FAISS index.
    *   **Optimization**: We fetch only `k=2` documents to save costs.
2.  **`grade_documents` (The Critic)**:
    *   An LLM looks at each retrieved chunk and asks: "Is this relevant?"
    *   It produces a binary "yes/no" score using Structured Output.
    *   It filters out the "no" documents.
3.  **`transform_query` (Improvement)**:
    *   **Triggered if**: The Critic finds NO relevant documents.
    *   It rewrites the user's question to be better for a web search (e.g., "Weather Tokyo" -> "current weather forecast Tokyo Japan").
4.  **`web_search` (Fallback)**:
    *   Uses **Tavily** to search the live internet with the rewritten query.
    *   If no API key is present, it returns a polite "Search disabled" message.
5.  **`generate`**:
    *   Takes the valid documents (from local or web).
    *   Sends them to the LLM to write the final answer.
    *   **Memory**: It injects the last ~3 turns of conversation history so the agent understands context.

### The Flow
*   `Start` -> `retrieve` -> `grade`
*   **Decision Point**:
    *   If Relevant Docs found -> `generate` -> `End`
    *   If No Docs -> `transform_query` -> `web_search` -> `generate` -> `End`

## 4. The Interface (`main.py`)
This runs the chat loop in your terminal.
*   **Session Management**: It generates a unique `thread_id` (UUID) every time you run it. This tells the Agent's memory "This is a new conversation."
*   **Loop**:
    1.  Waits for user input.
    2.  Check for **Commands**:
        *   `/add <file>`: Calls `ingest.add_single_file` to ingest a new document instantly.
    3.  Sends the input to the Agent Graph.
    4.  Prints the final response.
    5.  Repeats until you type "exit".

## 5. The Dashboard (`app.py`)
This is the **Streamlit** web application, an alternative to `main.py`.
*   **Structure**:
    *   **Sidebar**: Contains configuration and a **File Uploader**. The uploader saves the file to disk and immediately calls `ingest.add_single_file`, enabling dynamic RAG updates.
    *   **Chat Window**: A persistent chat interface that mimics standard LLM UIs.
*   **State**: Uses `st.session_state` to store the conversation history and the unique `session_id`.

## 6. Deployment
*   **`Dockerfile`**: Defines a lightweight Python environment (based on `python:3.10-slim`), installs dependencies, and sets the default command to run Streamlit.
*   **`docker-compose.yml`**: Orchestrates the container. It mounts valid volumes (`data/`, `faiss_index/`) so that your documents and vector index persist even if you restart the container.

## Summary
1.  **Ingest** turns files into math.
2.  **Config** holds the keys.
3.  **Agent** thinks, checks itself, and searches the web if needed.
4.  **Main/App** lets you talk to it (Terminal or Web).
5.  **Docker** packages it all up for easy shipping.
