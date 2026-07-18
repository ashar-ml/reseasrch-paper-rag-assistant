import streamlit as st
import pandas as pd
from utils.api_client import RAGApiClient

# Page settings
st.set_page_config(
    page_title="Research Paper RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize API Client
if "api_client" not in st.session_state:
    st.session_state.api_client = RAGApiClient()

client = st.session_state.api_client

# Sleek Light Modern CSS Styling Injection (Clean Light Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Apply clean white theme & font family globally */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif !important;
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    
    /* Sleek Sidebar Styling with subtle clean off-white background */
    [data-testid="stSidebar"] {
        background-color: #fafafa !important;
        border-right: 1px solid rgba(0, 0, 0, 0.06) !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #475569 !important;
        font-size: 0.9rem;
    }
    
    /* Title Gradient styling */
    .title-text {
        background: linear-gradient(135deg, #6d28d9 0%, #4f46e5 50%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle-text {
        color: #475569;
        font-size: 1.02rem;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f3f4f6;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #475569;
        font-weight: 600;
        font-size: 0.92rem;
        transition: all 0.2s;
        border: none;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
    }

    /* Custom Section Headers */
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a !important;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
        background: linear-gradient(90deg, #0f172a, #475569);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Elegant Clean Cards on White Background */
    .metric-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        color: #0f172a !important;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(79, 70, 229, 0.3) !important;
        box-shadow: 0 8px 16px rgba(79, 70, 229, 0.08);
    }

    /* Document Item Row */
    .doc-item {
        display: flex;
        align-items: center;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }
    .doc-item:hover {
        background: #f1f5f9;
        border-color: rgba(79, 70, 229, 0.2);
    }
    .doc-icon {
        font-size: 1.25rem;
        margin-right: 10px;
        color: #4f46e5;
    }
    .doc-name {
        font-size: 0.82rem;
        font-weight: 500;
        color: #0f172a;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        flex: 1;
    }
    .doc-size {
        font-size: 0.7rem;
        color: #475569;
        margin-left: 8px;
    }

    /* Custom CSS to style Chat Messages */
    div[data-testid="stChatMessage"] {
        border-radius: 16px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 1.2rem !important;
        animation: fadeIn 0.4s ease;
        border: 1px solid #e2e8f0;
    }
    div[data-testid="stChatMessage"][data-test-role="user"] {
        background-color: #f5f3ff !important;
        border: 1px solid rgba(124, 58, 237, 0.15) !important;
    }
    div[data-testid="stChatMessage"][data-test-role="assistant"] {
        background-color: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.01) !important;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Buttons styling */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15) !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%) !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.2) !important;
        transform: translateY(-1px) !important;
    }

    /* Special Styling for Suggestion Prompt Buttons */
    div.element-container:has(button[key^="prompt_"]) button {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        color: #334155 !important;
        text-align: left !important;
        height: auto !important;
        min-height: 84px !important;
        padding: 14px 18px !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    div.element-container:has(button[key^="prompt_"]) button:hover {
        background: #f1f5f9 !important;
        border-color: rgba(79, 70, 229, 0.25) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.03) !important;
    }

    /* File uploader custom borders */
    [data-testid="stFileUploader"] {
        background: #fafafa !important;
        border: 1px dashed #cbd5e1 !important;
        border-radius: 10px !important;
    }

    /* Popover components styling */
    div[data-testid="stPopover"] button {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        color: #6d28d9 !important;
        border-radius: 20px !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        padding: 4px 12px !important;
        box-shadow: none !important;
    }
    div[data-testid="stPopover"] button:hover {
        background: rgba(109, 40, 217, 0.05) !important;
        border-color: rgba(109, 40, 217, 0.15) !important;
        transform: translateY(-1px) !important;
    }

    /* Inputs style */
    .stTextInput>div>div>input {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }

    /* Scrollbars customization */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.01);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(0, 0, 0, 0.08);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 0, 0, 0.18);
        border-radius: 3px;
    }

    /* Reduce page top padding to position header at the absolute top */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 1.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization for authentication and chat history
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history_loaded" not in st.session_state:
    st.session_state.history_loaded = False

# Auto-fetch history on load if authenticated and not yet loaded
if st.session_state.authenticated and not st.session_state.history_loaded:
    history_res = client.fetch_history(st.session_state.username)
    if history_res.get("status") == "success":
        st.session_state.messages = history_res.get("history", [])
    st.session_state.history_loaded = True

# Sidebar Content (Authenticated & Guest users)
with st.sidebar:
    st.markdown('<div style="font-size: 1.5rem; font-weight: bold; margin-bottom: 15px; color: #4f46e5;">RAG Assistant</div>', unsafe_allow_html=True)
    if st.session_state.authenticated:
        st.markdown(f"### Welcome, **{st.session_state.username}**!")
        col_side1, col_side2 = st.columns(2)
        with col_side1:
            if st.button("Logout", key="logout_btn", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.username = ""
                st.session_state.messages = []
                st.session_state.history_loaded = False
                st.rerun()
        with col_side2:
            if st.button("Clear Chat", key="clear_chat_btn", use_container_width=True):
                client.clear_history(st.session_state.username)
                st.session_state.messages = []
                st.success("History cleared!")
                st.rerun()
    else:
        st.markdown("### Welcome, **Guest**!")
        if st.button("Clear Chat", key="clear_guest_chat_btn", use_container_width=True):
            st.session_state.messages = []
            st.success("Chat cleared!")
            st.rerun()
        
    # Connection Health Check (evaluated early for document listing checks)
    health = client.check_health()

    st.divider()
    
    # Document Inventory Listing
    st.markdown("### Ingested Documents")
    if health.get("status") == "healthy":
        papers = client.list_papers()
        if papers:
            for paper in papers:
                filename = paper.get("filename", "Unknown")
                size_kb = paper.get("size_kb", 0)
                st.markdown(f"""
                <div class="doc-item">
                    <div style="flex: 1; min-width: 0;">
                        <div class="doc-name" title="{filename}">{filename}</div>
                        <div style="color: #64748b; font-size: 0.7rem; margin-top: 2px;">{size_kb:.1f} KB</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No papers uploaded yet. Ingest a paper below to build your knowledge base.")
    else:
        st.caption("Unable to fetch documents. Backend offline.")

    st.divider()
    
    # Document Upload Section
    st.markdown("### Ingest Research Papers")
    uploaded_files = st.file_uploader(
        "Upload paper PDF", 
        type=["pdf"], 
        accept_multiple_files=True,
        key="uploader"
    )
    
    if uploaded_files:
        for file in uploaded_files:
            if st.button(f"Process {file.name[:20]}...", key=f"btn_upload_{file.name}"):
                with st.spinner(f"Ingesting {file.name}..."):
                    res = client.upload_pdf(file.name, file.read())
                    if res.get("status") in ["received", "success"]:
                        st.success(f"Ingested: {file.name}")
                        st.rerun()
                    else:
                        st.error(f"Error: {res.get('message', 'Failed upload')}")
                        
    st.divider()
    st.markdown("### Assistant Health Status")
    if health.get("status") == "healthy":
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 10px; padding: 10px 14px; margin-bottom: 12px;">
            <div style="color: #10b981; font-weight: 600; font-size: 0.85rem; display: flex; align-items: center; gap: 6px;">
                Connected to Backend
            </div>
            <div style="color: #475569; font-size: 0.72rem; margin-top: 5px; line-height: 1.4;">
                Provider: <b>{health.get('provider')}</b><br>
                Model: <b>{health.get('embedding_model')}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 10px; padding: 10px 14px; margin-bottom: 12px;">
            <div style="color: #ef4444; font-weight: 600; font-size: 0.85rem; display: flex; align-items: center; gap: 6px;">
                Backend Offline
            </div>
            <div style="color: #475569; font-size: 0.72rem; margin-top: 5px;">
                FastAPI endpoint is unreachable. Please verify server status.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.caption("Research Paper RAG Assistant • Developed by Muhammad Ashar Ali")

# --- Top Navigation / Header Bar ---
header_col1, header_col2 = st.columns([8.2, 1.8])

with header_col1:
    st.markdown('<div class="title-text" style="font-size: 2rem;">Research Paper RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text" style="font-size: 0.95rem; margin-bottom: 0.5rem;">Academic QA powered by LangGraph, Hybrid Retrieval & Cross-Encoders</div>', unsafe_allow_html=True)

with header_col2:
    # Spacer removed to push controls to the very top
    if st.session_state.authenticated:
        # Show username and logout in top-right
        st.markdown(
            f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 6px 12px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.85rem; color: #475569; font-weight: 500;">{st.session_state.username}</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
        if st.button("Logout", key="top_logout_btn", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.session_state.messages = []
            st.session_state.history_loaded = False
            st.rerun()
    else:
        with st.popover("Login / Sign Up", use_container_width=True):
            st.markdown("<h4 style='margin-top:0; color:#0f172a; font-size:1.1rem; font-weight:600;'>Account Access</h4>", unsafe_allow_html=True)
            auth_mode = st.radio("Action:", ["Login", "Sign Up"], horizontal=True, key="popover_auth_mode", label_visibility="collapsed")
            username = st.text_input("Username", placeholder="Enter username", key="popover_user")
            password = st.text_input("Password", type="password", placeholder="Enter password", key="popover_pass")
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            
            if auth_mode == "Login":
                if st.button("Sign In", key="popover_signin_btn", use_container_width=True):
                    if not username or not password:
                        st.error("Please enter both credentials.")
                    else:
                        res = client.login(username, password)
                        if res.get("status") == "success":
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            history_res = client.fetch_history(username)
                            if history_res.get("status") == "success":
                                st.session_state.messages = history_res.get("history", [])
                            st.session_state.history_loaded = True
                            st.success("Successfully logged in!")
                            st.rerun()
                        else:
                            st.error(f"Login failed: {res.get('message')}")
            else:
                if st.button("Create Account", key="popover_signup_btn", use_container_width=True):
                    if not username or not password:
                        st.error("Please enter both credentials.")
                    elif len(username) < 3:
                        st.error("Username must be >= 3 characters.")
                    elif len(password) < 4:
                        st.error("Password must be >= 4 characters.")
                    else:
                        res = client.signup(username, password)
                        if res.get("status") == "success":
                            st.success("Account created! You can now log in.")
                        else:
                            st.error(f"Signup failed: {res.get('message')}")

# Main Content Layout using Tabbed Interface
tab_chat, tab_analytics = st.tabs(["Research Chat Workspace", "Performance Analytics & Benchmarks"])

# -----------------
# TAB 1: Chat Space
# -----------------
with tab_chat:
    st.markdown('<h3 style="margin-top: 0; margin-bottom: 0.2rem; color:#0f172a;">Research Chat Workspace</h3>', unsafe_allow_html=True)

    # Chat history container
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Custom Popover Citation Chips under assistant message
                citations = message.get("citations", [])
                if citations:
                    st.markdown("<div style='margin-top: 14px; font-size: 0.78rem; color: #475569; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;'>Cited Sources:</div>", unsafe_allow_html=True)
                    
                    # Columns to align source chips horizontally
                    cols = st.columns(min(len(citations), 4))
                    for idx, citation in enumerate(citations):
                        col_idx = idx % 4
                        with cols[col_idx]:
                            btn_label = f"[{idx+1}] {citation['paper_title'][:15]}... (P. {citation['page']})"
                            with st.popover(btn_label, use_container_width=True):
                                st.markdown(f"**Document:** `{citation['paper_title']}`")
                                st.markdown(f"**Page Number:** `{citation['page']}`")
                                st.markdown("**Retransmitted Snippet:**")
                                st.markdown(f"<div style='font-style: italic; color: #1e293b; background: rgba(241, 245, 249, 0.8); padding: 12px; border-radius: 8px; border-left: 3px solid #4f46e5; font-size: 0.88rem;'>\"{citation['text_snippet']}\"</div>", unsafe_allow_html=True)

    # Landing Hero Area for Empty Chat Space
    if not st.session_state.messages:
        display_name = st.session_state.username if st.session_state.authenticated else "Researcher"
        st.markdown(f"""
        <div style="text-align: center; margin-top: 0.5vh; margin-bottom: 2vh;">
            <h2 style="font-weight: 700; margin-top: 0px; color: #0f172a; font-size: 1.85rem;">
                Hello, {display_name}!
            </h2>
            <p style="color: #475569; font-size: 0.95rem; max-width: 500px; margin: 0.5rem auto 2.5rem auto;">
                Ask questions, inspect methodology, or compare results across your academic library.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 1.5rem;'>GET STARTED WITH A PROMPT SUGGESTION</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("**Summarize Methodology**\n\nExplain how the research papers parse their datasets & models.", key="prompt_1"):
                st.session_state.input_value = "Summarize the methodology, models, and datasets used in the uploaded research papers."
                st.rerun()
            if st.button("**Identify Limitations**\n\nWhat are the weaknesses or future paths described?", key="prompt_2"):
                st.session_state.input_value = "What limitations, weaknesses, or future work do the authors discuss in these papers?"
                st.rerun()
        with col2:
            if st.button("**Extract Key Findings**\n\nSummarize the core contributions and empirical results.", key="prompt_3"):
                st.session_state.input_value = "What are the main findings, results, and contributions of the uploaded papers?"
                st.rerun()
            if st.button("**Compare Architectures**\n\nHow do the systems differ in their model design?", key="prompt_4"):
                st.session_state.input_value = "Contrast and compare the different system architectures, methodologies, or algorithms introduced across the papers."
                st.rerun()

    # Chat prompt handling (Typed input or Suggestion Card click)
    clicked_prompt = None
    if "input_value" in st.session_state and st.session_state.input_value:
        clicked_prompt = st.session_state.input_value
        del st.session_state.input_value

    typed_prompt = st.chat_input("Ask a research question...")
    prompt = clicked_prompt if clicked_prompt else typed_prompt

    if prompt:
        # 1. Store and display user query
        st.session_state.messages.append({"role": "user", "content": prompt})
        if st.session_state.authenticated:
            client.save_history(st.session_state.username, "user", prompt)
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. Query assistant & display stream response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                with st.spinner("Analyzing papers and generating response..."):
                    res = client.query_assistant(prompt)
                    
                    if res.get("status") == "error":
                        response_placeholder.error(res.get("message"))
                    else:
                        answer = res["answer"]
                        citations = res.get("citations", [])
                        response_placeholder.markdown(answer)
                        
                        if citations:
                            st.markdown("<div style='margin-top: 14px; font-size: 0.78rem; color: #475569; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;'>Cited Sources:</div>", unsafe_allow_html=True)
                            cols = st.columns(min(len(citations), 4))
                            for idx, citation in enumerate(citations):
                                col_idx = idx % 4
                                with cols[col_idx]:
                                    btn_label = f"[{idx+1}] {citation['paper_title'][:15]}... (P. {citation['page']})"
                                    with st.popover(btn_label, use_container_width=True):
                                        st.markdown(f"**Document:** `{citation['paper_title']}`")
                                        st.markdown(f"**Page Number:** `{citation['page']}`")
                                        st.markdown("**Retransmitted Snippet:**")
                                        st.markdown(f"<div style='font-style: italic; color: #1e293b; background: rgba(241, 245, 249, 0.8); padding: 12px; border-radius: 8px; border-left: 3px solid #4f46e5; font-size: 0.88rem;'>\"{citation['text_snippet']}\"</div>", unsafe_allow_html=True)
                        
                        # Store in history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "citations": citations
                        })
                        if st.session_state.authenticated:
                            client.save_history(st.session_state.username, "assistant", answer, citations)


# -----------------
# TAB 2: Performance Analytics & Benchmarks
# -----------------
with tab_analytics:
    st.markdown('<h3 style="margin-top: 0; margin-bottom: 0.5rem; color:#0f172a;">Performance Analytics & Benchmarks</h3>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text" style="font-size: 0.95rem; margin-bottom: 1.5rem;">Compare models, run evaluation metrics using RAGAS, and inspect query latency metrics.</div>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)
    
    # Left Column: RAGAS Evaluation
    with col_left:
        st.markdown('<div class="section-header">RAGAS Pipeline Evaluator</div>', unsafe_allow_html=True)
        st.markdown("Run automated evaluation metrics suite on the RAG pipeline using RAGAS (faithfulness, recall, answer relevance).")
        
        if st.button("Run RAGAS Evaluation", key="btn_run_ragas"):
            with st.spinner("Running evaluation (faithfulness, recall, answer relevance)..."):
                eval_res = client.run_evaluation()
                status = eval_res.get("status", "error")
                if status in ["success", "fallback"]:
                    metrics = eval_res.get("scores", eval_res.get("results_placeholder", {}))
                    
                    if status == "success":
                        st.success("**Live RAGAS Metrics Computed**")
                    else:
                        st.warning("**Fallback Metrics Displayed**")
                        st.info("Define a valid `GROQ_API_KEY` in `.env` to compute live RAGAS scores.")
                    
                    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                    for metric, val in metrics.items():
                        st.markdown(f"""
                        <div style="margin-bottom: 12px;">
                            <div class="metric-card" style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 0.8rem; color: #475569; text-transform: uppercase; font-weight: 600;">{metric.replace('_', ' ')}</span>
                                <h3 style="margin: 0; color: #6d28d9; font-weight: bold; font-size: 1.4rem; white-space: nowrap;">{val:.2f}</h3>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.success("RAGAS evaluation complete!")
                else:
                    st.error("Evaluation failed.")

    # Right Column: Embedding Model Speed Benchmarking
    with col_right:
        st.markdown('<div class="section-header">Embedding Comparison</div>', unsafe_allow_html=True)
        st.markdown("Benchmark the similarity and speed of **MiniLM** (lightweight) vs **BGE-Large** (highly accurate).")
        
        if st.button("Benchmark Embeddings", key="btn_run_benchmark"):
            with st.spinner("Running similarity benchmarks..."):
                res = client.evaluate_embeddings()
                if res.get("status") == "success":
                    data = res.get("results", {})
                    
                    chart_data = []
                    for model_name, stats in data.items():
                        st.markdown(f"""
                        <div class="metric-card" style="text-align: left; padding: 1.2rem; margin-bottom: 15px;">
                            <h4 style="margin: 0 0 8px 0; color: #0f172a;">Model: <code>{model_name}</code></h4>
                            <div style="font-size: 0.85rem; color: #475569; line-height: 1.5;">
                                • Dimensions: <b>{stats.get('dimension')}</b><br>
                                • Avg Cosine Similarity: <b>{stats.get('average_similarity') if stats.get('average_similarity') is not None else 'N/A'}</b>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        chart_data.append({
                            "Model": model_name,
                            "Query Latency (ms)": stats.get('query_encoding_latency_ms'),
                            "Doc Latency (ms)": stats.get('doc_encoding_latency_ms')
                        })
                    
                    if chart_data:
                        st.markdown("#### ⚡ Encoding Latency Comparison (ms)")
                        df = pd.DataFrame(chart_data).set_index("Model")
                        st.bar_chart(df)
                        
                    st.success("Benchmarks complete!")
                else:
                    st.error("Model benchmarking failed.")
