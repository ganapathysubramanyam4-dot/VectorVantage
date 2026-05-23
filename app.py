import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ==============================================================================
# 1. PREMIUM ULTRALIGHT INTERFACE DESIGN SYSTEM
# ==============================================================================
st.set_page_config(
    page_title="VectorVantage AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End SaaS UI Stylesheet Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #fcfcfd !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p {
        color: #0f172a !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }
    
    .saas-header-box {
        padding: 20px 0px;
        margin-bottom: 30px;
        border-bottom: 1px solid #e4e4e7;
    }
    .saas-title {
        font-size: 30px;
        font-weight: 700;
        color: #09090b;
        letter-spacing: -0.75px;
        margin: 0;
    }
    .saas-subtitle {
        font-size: 14px;
        color: #71717a;
        margin-top: 4px;
        font-weight: 400;
    }
    
    .stExpander {
        border: 1px solid #e4e4e7 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03) !important;
    }

    .conversation-thread {
        width: 100%;
        margin-bottom: 24px;
        display: flex;
        flex-direction: column;
    }
    
    .avatar-row {
        display: flex;
        align-items: center;
        margin-bottom: 6px;
        gap: 8px;
    }
    .avatar-icon-user {
        width: 24px;
        height: 24px;
        background-color: #09090b;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 10px;
        font-weight: 600;
    }
    .avatar-icon-ai {
        width: 24px;
        height: 24px;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 10px;
        font-weight: 600;
    }
    .profile-name {
        font-size: 13px;
        font-weight: 600;
        color: #27272a;
    }

    .saas-bubble {
        padding: 14px 18px;
        border-radius: 12px;
        max-width: 85%;
        line-height: 1.6;
        font-size: 15px;
        color: #09090b;
    }
    .user-align {
        align-self: flex-end;
    }
    .user-saas-bubble {
        background-color: #f4f4f5;
        border: 1px solid #e4e4e7;
        border-top-right-radius: 2px;
    }
    .ai-align {
        align-self: flex-start;
    }
    .ai-saas-bubble {
        background-color: #ffffff;
        border: 1px solid #e4e4e7;
        border-top-left-radius: 2px;
        box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.02);
    }
    
    [data-testid="stSidebar"] .stButton>button {
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 10px 0px !important;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2) !important;
        transition: all 0.15s ease !important;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
        transform: translateY(-1px);
    }
    
    [data-testid="stSidebar"] .stButton>button[kind="primary"] {
        background: #ffffff !important;
        color: #dc2626 !important;
        border: 1px solid #fca5a5 !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton>button[kind="primary"]:hover {
        background: #fef2f2 !important;
        border-color: #ef4444 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. HELPER FUNCTIONS: PARSING, CHUNKING & SEMANTIC SEARCH
# ==============================================================================
def extract_text_from_pdfs(pdf_files):
    combined_text = ""
    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    combined_text += text + "\n"
        except Exception as e:
            st.sidebar.error(f"Error reading file {pdf.name}: {str(e)}")
    return combined_text

def chunk_text_with_overlap(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    if len(words) <= chunk_size:
        return [" ".join(words)] if words else []
        
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += (chunk_size - overlap)
    return chunks

def retrieve_relevant_chunks(query, chunks, top_k=3):
    if not chunks:
        return ""
    
    clean_query = re.sub(r'[^\w\s]', '', query.lower().strip())
    if not clean_query:
        return ""

    corpus = chunks + [query]
    vectorizer = TfidfVectorizer(stop_words='english')
    
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        top_indices = similarity_scores.argsort()[-top_k:][::-1]
        
        relevant_text = ""
        has_any_match = False
        
        for idx in top_indices:
            if similarity_scores[idx] > 0.0:
                relevant_text += f"\n--- PDF SOURCE CHUNK ---\n{chunks[idx]}\n"
                has_any_match = True
                
        if not has_any_match:
            return ""
        return relevant_text
    except Exception:
        return "\n".join(chunks[:top_k])

def rewrite_query_with_history(user_query, chat_history, model_name):
    """Uses LLM orchestration layer to re-contextualize lazy follow-up queries."""
    if not chat_history:
        return user_query
        
    formatted_history = ""
    for msg in chat_history[-4:]:
        role = "User" if msg["role"] == "user" else "AI"
        formatted_history += f"{role}: {msg['text']}\n"
        
    rewrite_prompt = (
        f"Given the following chat history conversation dialogue and a new sub-query from the user, "
        f"rewrite the user query to be completely self-contained and descriptive so it can be used for searching documents.\n\n"
        f"CHAT TRANSCRIPT:\n{formatted_history}\n"
        f"NEW USER QUERY: {user_query}\n\n"
        f"CRITICAL: Output ONLY the resolved rewritten query string. No extra fluff words or intro structures."
    )
    try:
        rewriter_model = genai.GenerativeModel(model_name=model_name)
        response = rewriter_model.generate_content(rewrite_prompt)
        if response.text.strip():
            return response.text.strip()
    except Exception:
        pass
    return user_query

# ==============================================================================
# 3. SIDEBAR PANEL CONTROL, KEY ROUTING & TOOLS
# ==============================================================================
st.sidebar.markdown("<h2 style='font-size: 18px; font-weight: 600; margin-bottom: 20px; color:#0f172a;'>System Control</h2>", unsafe_allow_html=True)

api_key = st.secrets.get("GEMINI_API_KEY", "")
custom_key = st.sidebar.text_input("Overwrite Gemini API Key", type="password", help="Leave blank to use main cloud cluster key.")
final_api_key = custom_key if custom_key else api_key

if not final_api_key:
    st.sidebar.warning("🔑 Missing API Key Configuration.")
else:
    genai.configure(api_key=final_api_key)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

model_option = st.sidebar.selectbox(
    "LLM Architecture Brain:",
    ["gemini-2.5-flash", "gemini-2.5-pro"]
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

uploaded_files = st.sidebar.file_uploader(
    "Knowledge Documents (PDF):",
    type=["pdf"],
    accept_multiple_files=True
)

if "processed_chunks" not in st.session_state:
    st.session_state.processed_chunks = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_summary" not in st.session_state:
    st.session_state.document_summary = ""

if uploaded_files:
    if st.sidebar.button("Process & Index Data", use_container_width=True):
        with st.sidebar.status("Vectorizing context frames...", expanded=False) as status:
            raw_text = extract_text_from_pdfs(uploaded_files)
            if raw_text.strip():
                st.session_state.processed_chunks = chunk_text_with_overlap(raw_text)
                status.update(label="Index Complete", state="complete")
                
                try:
                    summary_model = genai.GenerativeModel(model_name=model_option)
                    summary_prompt = f"Provide a brief overview and high-level 3-bullet abstract summary of this text layout:\n\n{raw_text[:6000]}"
                    response = summary_model.generate_content(summary_prompt)
                    st.session_state.document_summary = response.text
                except Exception:
                    st.session_state.document_summary = "Abstract summary pipeline offline."
            else:
                status.update(label="Parsing Failed", state="error")

if st.session_state.processed_chunks:
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    if st.sidebar.button("Clear Indexed Environment", type="primary", use_container_width=True):
        st.session_state.processed_chunks = []
        st.session_state.chat_history = []
        st.session_state.document_summary = ""
        st.rerun()

if st.session_state.chat_history:
    log_data = ""
    for msg in st.session_state.chat_history:
        log_data += f"{msg['role'].upper()}: {msg['text']}\n\n"
    st.sidebar.download_button(
        label="Export Chat Session Logs",
        data=log_data,
        file_name="chat_session.txt",
        mime="text/plain",
        use_container_width=True
    )

# ==============================================================================
# 4. MAIN USER INTERFACE DISPLAY & CONVERSATION PIPELINE
# ==============================================================================

st.markdown("""
<div class="saas-header-box">
    <h1 class="saas-title">VectorVantage AI</h1>
    <p class="saas-subtitle">Document Intelligence Workspace & Real-time Context-Aware Neural Engine</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.document_summary:
    with st.expander("📊 Document Structural Overview Abstract", expanded=False):
        st.markdown(st.session_state.document_summary)
    st.markdown("<br>", unsafe_allow_html=True)

for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="conversation-thread user-align">
            <div class="avatar-row" style="justify-content: flex-end;">
                <span class="profile-name">You</span>
                <div class="avatar-icon-user">U</div>
            </div>
            <div class="saas-bubble user-saas-bubble user-align">
                {message["text"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="conversation-thread ai-align">
            <div class="avatar-row">
                <div class="avatar-icon-ai">VV</div>
                <span class="profile-name">VectorVantage AI</span>
            </div>
            <div class="saas-bubble ai-saas-bubble ai-align">
                {message["text"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

user_input = st.chat_input("Ask an actionable query from your document database workspace...")

if user_input:
    if not final_api_key:
        st.error("Please configure a valid Gemini API Key to initialize query response sequences.")
    else:
        st.markdown(f"""
        <div class="conversation-thread user-align">
            <div class="avatar-row" style="justify-content: flex-end;">
                <span class="profile-name">You</span>
                <div class="avatar-icon-user">U</div>
            </div>
            <div class="saas-bubble user-saas-bubble user-align">
                {user_input}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "user", "text": user_input})
        
        with st.spinner("Processing context alignment..."):
            optimized_search_query = rewrite_query_with_history(
                user_input, 
                st.session_state.chat_history[:-1],
                model_option
            )
            
            extracted_context = retrieve_relevant_chunks(optimized_search_query, st.session_state.processed_chunks)
            
            system_instruction_guardrail = (
                "You are an absolute objective RAG Data Analyst. Your sole mission is to answer user queries using only the provided context snippets extracted from the uploaded document.\n"
                "STRICT RULES:\n"
                "1. If the provided context is empty or does not contain explicit, undeniable proof to answer the question, reply exactly with: 'I cannot find the answer within the uploaded documents.'\n"
                "2. Do not use external pre-trained knowledge outside the provided text fragments.\n"
                "3. Keep answers direct, accurate, and completely factual based on context data."
            )
            
            full_generation_prompt = f"CONTEXT RELEVANT PIECES:\n{extracted_context}\n\nUSER QUESTION: {user_input}"
            
            try:
                model = genai.GenerativeModel(
                    model_name=model_option,
                    system_instruction=system_instruction_guardrail
                )
                
                response = model.generate_content(
                    full_generation_prompt,
                    generation_config={"temperature": 0.0}
                )
                
                ai_response_text = response.text
                
                st.markdown(f"""
                <div class="conversation-thread ai-align">
                    <div class="avatar-row">
                        <div class="avatar-icon-ai">VV</div>
                        <span class="profile-name">VectorVantage AI</span>
                    </div>
                    <div class="saas-bubble ai-saas-bubble ai-align">
                        {ai_response_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.chat_history.append({"role": "model", "text": ai_response_text})
                
            except Exception as e:
                st.error(f"API Operation execution failure: {str(e)}")
