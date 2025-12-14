import streamlit as st
import os
import time
import uuid
from langchain_core.messages import HumanMessage, AIMessage

# Page Config
st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Verify Config
from config import config
if not config.AZURE_OPENAI_API_KEY:
    st.error("Azure OpenAI API Key not found. Please check your environment.")
    st.stop()

# Load Agent Graph (Cached resource)
@st.cache_resource
def load_agent():
    from agent import build_graph
    return build_graph()

app = load_agent()

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
    st.divider()
    
    st.subheader("📁 Upload Document")
    uploaded_file = st.file_uploader("Upload a file (.txt, .md, .pdf, .csv, .xlsx)", type=["txt", "md", "pdf", "csv", "xlsx"])
    
    if uploaded_file is not None:
        if st.button("Ingest File"):
            with st.spinner("Ingesting file..."):
                # Save temp file
                save_path = os.path.join(config.DATA_PATH, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Ingest
                from ingest import add_single_file
                result = add_single_file(save_path)
                st.success(result)

    st.divider()
    st.markdown("### System Status")
    st.success("Agent Active")
    st.info(f"Model: {config.AZURE_DEPLOYMENT_NAME}")

# Main Chat Interface
st.title("🤖 Agentic RAG Assistant")
st.markdown("Ask questions about your domain documents. The agent can search locally and fallback to web search if needed.")

# Display History
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# User Input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to state
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Status container for "Thinking..." steps
        status_placeholder = st.empty()
        
        # Prepare Config
        thread_config = {"configurable": {"thread_id": st.session_state.session_id}}
        inputs = {
            "question": prompt,
            "messages": [HumanMessage(content=prompt)]
        }
        
        # Async loop helper for Streamlit
        import asyncio
        
        async def process_stream():
            full_ans = ""
            status_text = "Starting..."
            status_placeholder.status(status_text, expanded=True)
            
            # Using astream_events to catch tokens and node transitions
            # version="v2" is recommended for newer langchain
            try:
                msg_count = 0
                async for event in app.astream_events(inputs, config=thread_config, version="v2"):
                    kind = event["event"]
                    
                    # 1. Update Status based on Node usage
                    if kind == "on_chain_start":
                        name = event["name"]
                        if name == "retrieve":
                            status_placeholder.markdown("🔍 **Retrieving documents...**")
                        elif name == "grade_documents":
                            status_placeholder.markdown("⚖️ **Grading relevance...**")
                        elif name == "web_search":
                            status_placeholder.markdown("🌐 **Searching the web...**")
                        elif name == "transform_query":
                            status_placeholder.markdown("🤔 **Refining query...**")
                        elif name == "generate":
                            status_placeholder.markdown("✍️ **Generating answer...**")
                            
                    # 2. Stream Tokens
                    if kind == "on_chat_model_stream":
                        # We only want to stream if we are in the 'generate' node roughly.
                        # Sometimes grader also streams, but it's structured output so usually not 'on_chat_model_stream' with content?
                        # Actually grader is structured output, so it might emit events but with parsed args.
                        # We focus on content chunks.
                        content = event["data"]["chunk"].content
                        if content:
                            full_ans += content
                            message_placeholder.markdown(full_ans + "▌")
                            
                return full_ans
                
            except Exception as e:
                st.error(f"Error during streaming: {e}")
                return ""

        # Run async loop
        try:
            final_answer = asyncio.run(process_stream())
            
            # Final Cleanup
            message_placeholder.markdown(final_answer)
            status_placeholder.empty() # Clear status messages
            
            # Append to history
            if final_answer:
                st.session_state.messages.append(AIMessage(content=final_answer))
                
        except Exception as e:
            st.error(f"Fatal error: {e}")

# Footer
st.markdown("---")
st.caption("Agentic RAG System | Powered by LangGraph & Azure OpenAI")
