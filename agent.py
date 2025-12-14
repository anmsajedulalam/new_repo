from typing import Annotated, Sequence, TypedDict, Literal
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import END, StateGraph
from config import config

# --- Setup ---
def get_retriever():
    """Load the FAISS index and return a retriever."""
    if not os.path.exists(config.INDEX_PATH):
        raise FileNotFoundError(f"Index not found at {config.INDEX_PATH}. Run ingest.py first.")
    
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=config.AZURE_EMBEDDING_DEPLOYMENT,
        openai_api_version=config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
    )
    vector_store = FAISS.load_local(config.INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    # OPTIMIZATION: k=2 to save tokens (default is usually 4)
    return vector_store.as_retriever(search_kwargs={"k": 2})

# --- State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    documents: Sequence[str]
    question: str
    route: str # New field for routing decision

from langgraph.checkpoint.memory import MemorySaver

# --- Nodes ---

def router_node(state):
    """
    Route the user's query to either the vectorstore or general conversation.
    """
    print("---ROUTER---")
    question = state["question"]
    
    llm = AzureChatOpenAI(
        azure_deployment=config.AZURE_DEPLOYMENT_NAME,
        openai_api_version=config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        temperature=0
    )
    
    class RouteQuery(BaseModel):
        """Route a user query to the most relevant datasource."""
        datasource: Literal["vectorstore", "general"] = Field(
            ...,
            description="Given a user question choose to route it to web search or a vectorstore user question.",
        )

    structured_llm_router = llm.with_structured_output(RouteQuery)
    
    system = """You are an expert at routing a user question to a vectorstore or general conversation.
    The vectorstore contains documents about specific domain knowledge (Company Policies, Project Omega, specific tech).
    Use the vectorstore for questions on these topics.
    For greetings ("hi", "hello"), compliments, casual chat ("how are you"), or simple acknowledgments ("ok", "thanks"), use 'general' conversation.
    """
    
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    
    router = route_prompt | structured_llm_router
    try:
        source = router.invoke({"question": question})
        decision = source.datasource
    except Exception as e:
        # Fallback to general if parsing fails or content filter triggers
        print(f"Router failed (possible content filter): {e}")
        # If it's a content filter error, we should probably stop.
        if "content_filter" in str(e) or "ResponsibleAIPolicyViolation" in str(e):
            decision = "blocked"
        else:
            decision = "general"
        
    print(f"Routing to: {decision}")
    return {"route": decision}

def handle_blocked(state):
    """
    Handle content filter blocks gracefully.
    """
    print("---CONTENT BLOCKED---")
    return {"messages": [AIMessage(content="I'm sorry, but I cannot answer that question as it triggered our content safety policies. Please ask something else.")]}

def general_conversation(state):
    """
    Handle general conversation (chitchat) without RAG.
    """
    print("---GENERAL CONVERSATION---")
    messages = state["messages"]
    
    llm = AzureChatOpenAI(
        azure_deployment=config.AZURE_DEPLOYMENT_NAME,
        openai_api_version=config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        temperature=0.7 # Slight creativity for chitchat
    )
    
    # Simple prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a friendly and helpful AI assistant. Engage in polite conversation with the user. If they ask about specific topics you don't know, suggest they ask about the domain documents."),
            ("placeholder", "{messages}"),
        ]
    )
    
    chain = prompt | llm
    
    # Trim history
    messages_to_send = messages[-6:] if len(messages) > 6 else messages
    
    response = chain.invoke({"messages": messages_to_send})
    return {"messages": [response]}


def retrieve(state):
    """
    Retrieve documents based on the latest question.
    """
    print("---RETRIEVE---")
    question = state["question"]
    try:
        retriever = get_retriever()
        documents = retriever.invoke(question)
        print(f"Retrieved {len(documents)} documents.")
        return {"documents": documents}
    except Exception as e:
        print(f"Retrieval failed: {e}")
        return {"documents": []}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    print("---CHECK RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    # LLM with function call/structured output to grade
    llm = AzureChatOpenAI(
        azure_deployment=config.AZURE_DEPLOYMENT_NAME,
        openai_api_version=config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        temperature=0
    )

    class Grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    structured_llm_grader = llm.with_structured_output(Grade)

    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    
    retrieval_grader = grade_prompt | structured_llm_grader
    
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
            
    return {"documents": filtered_docs}

def generate(state):
    """
    Generate answer using the valid documents.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    messages = state["messages"]
    
    if not documents:
        return {"messages": [AIMessage(content="I could not find relevant information in the provided documents to answer your question.")]}
    
    llm = AzureChatOpenAI(
        azure_deployment=config.AZURE_DEPLOYMENT_NAME,
        openai_api_version=config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        temperature=0
    )
    
    # Prompt with History
    # We include previous messages to provide context
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."),
            ("placeholder", "{messages}"), # Injects history
            ("human", "Question: {question} \n\n Context: {context} \n\n Answer:"),
        ]
    )
    
    rag_chain = prompt | llm
    
    # Format docs
    context = "\n\n".join([d.page_content for d in documents])
    
    # OPTIMIZATION: Trim history to last 6 messages (~3 turns) to control context usage
    # We allow more context than strictly 2 turns to ensure continuity, but cap it.
    messages_to_send = messages[-6:] if len(messages) > 6 else messages
    
    response = rag_chain.invoke({"question": question, "context": context, "messages": messages_to_send})
    return {"messages": [response]}

def transform_query(state):
    """
    Transform the query to produce a better question.
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    
    llm = AzureChatOpenAI(
        azure_deployment=config.AZURE_DEPLOYMENT_NAME,
        openai_api_version=config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        temperature=0
    )

    # Simple re-writer
    system = """You are a question re-writer that converts an input question to a better version that is optimized \n 
     for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ]
    )
    
    chain = prompt | llm
    output = chain.invoke({"question": question})
    better_question = output.content
    print(f"Rewritten Query: {better_question}")
    return {"question": better_question}

def web_search(state):
    """
    Web search based on the re-phrased question.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    
    # Check if API key exists
    if not config.TAVILY_API_KEY:
        print("Tavily API Key missing. Skipping web search.")
        from langchain_core.documents import Document
        return {"documents": [Document(page_content="Web search is disabled because no TAVILY_API_KEY was found. Relevant info not found in local docs.")]}

    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        # Explicitly pass the key from config
        tool = TavilySearchResults(k=3, tavily_api_key=config.TAVILY_API_KEY)
        docs = tool.invoke({"query": question})
        # Tavily returns list of dicts: [{'content': '...', 'url': '...'}]
        # We need to format them as 'documents' (objects or strings) for the 'generate' node
        # The 'generate' node expects objects with .page_content attribute
        
        from langchain_core.documents import Document
        web_results = []
        for d in docs:
            web_results.append(Document(page_content=d["content"]))
            
        return {"documents": web_results}
        
    except Exception as e:
        print(f"Web search failed: {e}")
        return {"documents": []}

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.
    """
    print("---DECIDE TO GENERATE---")
    state["question"]
    filtered_documents = state["documents"]
    
    if not filtered_documents:
        # No relevant documents found, so we transform query and route to web search
        print("---DECISION: TRANSFORM QUERY & WEB SEARCH---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

def decide_route(state):
    """
    Route based on router_node decision.
    """
    print(f"---DECISION: {state['route']}---")
    return state["route"]

# --- Graph Definition ---
def build_graph():
    workflow = StateGraph(AgentState)

    # Define the nodes
    workflow.add_node("router", router_node) # Entry point
    workflow.add_node("general_conversation", general_conversation)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("handle_blocked", handle_blocked)

    # Build graph
    workflow.set_entry_point("router")
    
    # Logic: Router -> (Vectorstore or General or Blocked)
    workflow.add_conditional_edges(
        "router",
        decide_route,
        {
            "vectorstore": "retrieve",
            "general": "general_conversation",
            "blocked": "handle_blocked",
        },
    )
    
    workflow.add_edge("general_conversation", END)
    workflow.add_edge("handle_blocked", END)
    
    workflow.add_edge("retrieve", "grade_documents")
    
    # Conditional edge
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    
    # Add Checkpointer for Memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# For direct testing
import os
if __name__ == "__main__":
    if not os.path.exists(config.INDEX_PATH):
        print("Index not found. Please run ingest.py first.")
    else:
        app = build_graph()
        result = app.invoke({"question": "What is the capital of France?"})
        print(result["messages"][-1].content)
