"""
HydroAssist - Streamlit Web Interface
Phase 1: Intelligent Hydrogeological Consultant Chatbot
"""

import streamlit as st
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.graph import HydroAssistChat
from src.core.config import load_config
from langchain_core.messages import HumanMessage, AIMessage

# Page configuration
st.set_page_config(
    page_title="HydroAssist - Hydrogeological Consultant",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1E88E5;
    font-weight: bold;
}
.subtitle {
    font-size: 1rem;
    color: #666;
    margin-bottom: 2rem;
}
.context-box {
    padding: 1rem;
    background-color: #f0f2f6;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: bold;
    color: #1E88E5;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_chat_system(_config_hash=None):
    """Load and cache the chat system."""
    try:
        config = load_config()
        chat = HydroAssistChat(config)
        return chat, None
    except Exception as e:
        return None, str(e)

def get_config_hash():
    """Get hash of config file to invalidate cache when config changes."""
    import hashlib
    config_path = Path("configs/default.yaml")
    if config_path.exists():
        return hashlib.md5(config_path.read_bytes()).hexdigest()
    return "default"


# Initialize session state
if "chat" not in st.session_state:
    config_hash = get_config_hash()
    chat, error = load_chat_system(_config_hash=config_hash)

    if error:
        st.error(f"Failed to initialize HydroAssist: {error}")
        st.stop()

    st.session_state.chat = chat
    st.session_state.initialized = True

# Main header
st.markdown('<p class="main-header">💧 HydroAssist</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Open-source intelligent hydrogeological consultant powered by Gemini AI</p>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("🔍 Current Context")

    # Get context from chat
    context = st.session_state.chat.get_context()

    # Display metrics
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Aquifer Type",
            context['aquifer'].capitalize(),
            delta=None
        )

    with col2:
        st.metric(
            "Messages",
            context['message_count'],
            delta=None
        )

    # Selected method
    st.markdown("**Selected Method:**")
    method_display = context['method'] if context['method'] else "None"
    st.info(method_display)

    # Intent
    st.markdown("**Current Intent:**")
    intent_display = context['intent'].capitalize()
    st.info(intent_display)

    st.divider()

    # Phase indicator
    st.subheader("📊 Available Features")
    st.success("✅ Phase 1: Consultation (Active)")
    st.info("🚧 Phase 2: Calculation (Coming Soon)")

    st.divider()

    # Controls
    st.subheader("⚙️ Controls")

    # Reset button
    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.chat.reset()
        st.rerun()

    # Export chat
    if st.button("💾 Export Chat", use_container_width=True):
        messages = st.session_state.chat.state["messages"]

        chat_text = "\n\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in messages
        ])

        st.download_button(
            label="Download Chat History",
            data=chat_text,
            file_name="hydroassist_chat.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.divider()

    # About
    with st.expander("ℹ️ About HydroAssist"):
        st.markdown("""
        **HydroAssist Phase 1** is an intelligent consultant for hydrogeological pumping test analysis.

        **Knowledge Base:**
        - USGS TWI Book 3-B1
        - Classical pumping test methods

        **Technology:**
        - LangGraph state management
        - Gemini 2.0 Flash LLM
        - ChromaDB vector store
        - RAG architecture

        **Phase 2 Roadmap:**
        - Curve-fitting analysis
        - Parameter estimation
        - Diagnostic plots
        """)

    st.caption("Built with LangGraph + Gemini 2.0")

# Main chat interface
st.subheader("💬 Chat")

# Get messages from chat state
messages = st.session_state.chat.state["messages"]

# Display chat history
for message in messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Display retrieved sources if available
retrieved_docs = st.session_state.chat.state.get("retrieved_docs", [])

if retrieved_docs:
    with st.expander(f"📚 Retrieved Sources ({len(retrieved_docs)})"):
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc.get("metadata", {})

            col1, col2 = st.columns([3, 1])

            with col1:
                source = metadata.get('source', 'Unknown')
                page = metadata.get('page')

                citation = f"**Source {i}:** {source}"
                if page:
                    citation += f", p.{page}"

                st.markdown(citation)
                st.caption(f"Method: {metadata.get('method', 'N/A')} | Aquifer: {metadata.get('aquifer', 'N/A')}")

            with col2:
                score = doc.get('score', 0)
                st.metric("Score", f"{score:.2f}")

            # Content preview
            content = doc['content']
            preview = content[:200] + "..." if len(content) > 200 else content
            st.text(preview)

            st.divider()

# Chat input
if prompt := st.chat_input("Ask about pumping test methods, aquifer analysis, or request calculations..."):
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Get response from chat system
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.chat.send_message(prompt)

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)

            # Force rerun to update sidebar context
            st.rerun()

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.error("Please try rephrasing your question or reset the conversation.")

# Welcome message for first-time users
if len(messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 **Welcome to HydroAssist!** I'm your hydrogeological consultant.

        I can help you with:
        - 📖 Understanding pumping test analysis methods (Theis, Cooper-Jacob, Hantush-Jacob, etc.)
        - 🎯 Selecting appropriate methods for different aquifer conditions
        - 📐 Explaining assumptions and limitations
        - 🔬 Discussing aquifer characterization theory

        **Coming in Phase 2:** Calculation and curve-fitting capabilities

        **Try asking:**
        - "What is the Theis method?"
        - "How do I analyze a confined aquifer?"
        - "When should I use Cooper-Jacob instead of Theis?"
        """)

# Footer
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("HydroAssist Phase 1 (BTP-1)")
with col2:
    st.caption("Powered by Gemini 2.0 Flash")
with col3:
    st.caption("Open Source Project")
