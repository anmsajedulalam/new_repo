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
        
        # Synchronous Node-level Streaming (Reliable fallback)
        try:
            status_text = "Starting..."
            status_placeholder.status(status_text, expanded=True)
            
            full_ans = ""
            
            # Use .stream() to get updates from each node
            for output in app.stream(inputs, config=thread_config):
                # output is like {'retrieve': {'documents': [...]}}
                for key, value in output.items():
                    if key == "router":
                        status_placeholder.markdown("🚦 **Routing query...**")
                    elif key == "general_conversation":
                        status_placeholder.markdown("💬 **Chatting...**")
                        if "messages" in value:
                            final_msg = value["messages"][-1]
                            full_ans = final_msg.content
                            message_placeholder.markdown(full_ans)
                    elif key == "handle_blocked":
                        status_placeholder.markdown("🚫 **Content Blocked**")
                        if "messages" in value:
                            final_msg = value["messages"][-1]
                            full_ans = final_msg.content
                            message_placeholder.markdown(full_ans)
                    elif key == "retrieve":
                        status_placeholder.markdown("🔍 **Retrieving documents...**")
                    elif key == "grade_documents":
                        status_placeholder.markdown("⚖️ **Grading relevance...**")
                    elif key == "web_search":
                        status_placeholder.markdown("🌐 **Searching the web...**")
                    elif key == "transform_query":
                        status_placeholder.markdown("🤔 **Refining query...**")
                    elif key == "generate":
                        status_placeholder.markdown("✍️ **Generating answer...**")
                        # The generate node returns the final message update
                        if "messages" in value:
                            final_msg = value["messages"][-1]
                            full_ans = final_msg.content
                            message_placeholder.markdown(full_ans)
                            
            # Final Cleanup
            if full_ans:
                st.session_state.messages.append(AIMessage(content=full_ans))
            status_placeholder.empty()
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()

# Footer
st.markdown("---")
st.caption("Agentic RAG System | Powered by LangGraph & Azure OpenAI")
