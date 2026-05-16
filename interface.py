import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🌟 1. STREAMLIT PAGE CONFIGURATION (Updated Name)
st.set_page_config(page_title="VectorVantage AI", page_icon="🤖", layout="wide")

# --- CUSTOM CSS DESIGN ---
st.markdown("""
    <style>
    .stChatInput {
        position: fixed;
        bottom: 20px;
        z-index: 999;
    }
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 7rem;
    }
    </style>
""", unsafe_allow_html=True)

# 2. INITIALIZE SESSION STATES
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []
if "pdf_summary" not in st.session_state:
    st.session_state.pdf_summary = ""
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []

# --- SIDEBAR FEATURE: API CONFIGURATION & TOOLS ---
with st.sidebar:
    st.title("⚙️ VectorVantage Tools")
    
    # EXTRA FEATURE: Custom API Key input box for Users
    user_api_key = st.text_input(
        "Enter your Gemini API Key (Optional)", 
        type="password", 
        help="If the free tier limit is reached, you can generate your own API key from Google AI Studio and use it here."
    )
    
    # SMART API CONFIGURATION LOGIC
    if user_api_key:
        genai.configure(api_key=user_api_key)
        st.success("Using your Custom API Key!")
    else:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        else:
            st.warning("Please enter a custom key above or configure GEMINI_API_KEY in Streamlit Secrets.")
            st.stop()
            
    st.write("---")
    
    selected_model = st.selectbox("Select Model", ["gemini-2.5-flash", "gemini-2.5-pro"])
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    if st.session_state.messages:
        chat_download_text = "--- VECTORVANTAGE CHAT LOG ---\n\n"
        for msg in st.session_state.messages:
            role_label = "User" if msg["role"] == "user" else "VectorVantage AI"
            chat_download_text += f"[{role_label}]: {msg['content']}\n\n"
            chat_download_text += "-"*40 + "\n\n"
        
        st.download_button(
            label="📥 Download Chat History",
            data=chat_download_text,
            file_name="vectorvantage_chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )
        
    st.write("---")
    
    st.subheader("📄 Smart PDF Analysis (RAG)")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# --- HELPER FUNCTIONS FOR RAG ---
def chunk_text(text, chunk_size=1000, overlap=200):
    """Splits text into chunks with overlap so context isn't lost at edges."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def retrieve_relevant_chunks(query, chunks, top_k=3):
    """Finds the most relevant chunks based on user query using TF-IDF."""
    if not chunks:
        return ""
    
    corpus = chunks + [query]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    top_indices = similarity_scores.argsort()[-top_k:][::-1]
    
    relevant_text = ""
    for idx in top_indices:
        if similarity_scores[idx] > 0.05:
            relevant_text += f"\n--- Context Source ---\n{chunks[idx]}\n"
            
    return relevant_text

# --- FILE PROCESSING ---
if uploaded_files:
    current_file_names = [f.name for f in uploaded_files]
    
    if current_file_names != st.session_state.uploaded_file_names:
        with st.spinner("Extracting & Chunking text from PDFs..."):
            combined_text = ""
            try:
                for uploaded_file in uploaded_files:
                    reader = PdfReader(uploaded_file)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            combined_text += text + "\n"
                
                st.session_state.pdf_chunks = chunk_text(combined_text)
                st.session_state.uploaded_file_names = current_file_names
                st.session_state.pdf_summary = ""  
                st.success(f"Processed & split into {len(st.session_state.pdf_chunks)} intelligent chunks!")
            except Exception as e:
                st.error(f"Error reading PDF files: {e}")

if st.session_state.pdf_chunks and st.sidebar.button("❌ Remove All PDF Context", use_container_width=True):
    st.session_state.pdf_chunks = []
    st.session_state.pdf_summary = ""
    st.session_state.uploaded_file_names = []
    st.success("All PDF contexts removed!")
    st.rerun()

# 🌟 3. MAIN CHAT INTERFACE (Updated Title)
st.title("🤖 VectorVantage: Intelligent RAG Chatbot")

# Display PDF Status & Automatic Summary if attached
if st.session_state.pdf_chunks:
    st.info(f"💡 **RAG Active:** {len(st.session_state.uploaded_file_names)} PDF(s) split into {len(st.session_state.pdf_chunks)} chunks in memory!")
    st.caption(f"Active Files: {', '.join(st.session_state.uploaded_file_names)}")
    
    with st.expander("✨ Click here to view Combined PDF Summary", expanded=False):
        if not st.session_state.pdf_summary:
            with st.spinner("Generating One-time Summary..."):
                try:
                    summary_model = genai.GenerativeModel(selected_model)
                    summary_context = "\n".join(st.session_state.pdf_chunks[:10]) 
                    summary_response = summary_model.generate_content(
                        f"Provide a concise combined summary and key highlights for the following document text:\n\n{summary_context}"
                    )
                    st.session_state.pdf_summary = summary_response.text
                except Exception as e:
                    if "429" in str(e) or "ResourceExhausted" in str(e):
                        st.session_state.pdf_summary = "⚠️ Rate limit reached. Provide your own API key in the sidebar to load the summary!"
                    else:
                        st.session_state.pdf_summary = f"Could not generate summary: {e}"
        st.markdown(st.session_state.pdf_summary)

# Display Past Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle New User Input
if user_query := st.chat_input("Ask me anything about the documents..."):
    
    with st.chat_message("user"):
        st.markdown(user_query)
    
    st.session_state.messages.append({"role": "user", "content": user_query})

    try:
        model = genai.GenerativeModel(selected_model)
        formatted_history = []
        
        relevant_context = ""
        if st.session_state.pdf_chunks:
            with st.spinner("Searching document database..."):
                relevant_context = retrieve_relevant_chunks(user_query, st.session_state.pdf_chunks, top_k=3)
        
        if relevant_context:
            system_prompt = f"You are a helpful assistant. Use the following dynamic context extracted from the PDFs to answer the user's question accurately:\n\n{relevant_context}"
            formatted_history.append({"role": "user", "parts": [system_prompt]})
            formatted_history.append({"role": "model", "parts": ["Understood. I will combine the relevant document chunks and chat memory to answer you."]})
        
        for msg in st.session_state.messages:
            role = "user" if msg["role"] == "user" else "model"
            formatted_history.append({"role": role, "parts": [msg["content"]]})
            
        with st.spinner("Thinking..."):
            response = model.generate_content(formatted_history)
        
        with st.chat_message("assistant"):
            st.markdown(response.text)
            
        st.session_state.messages.append({"role": "assistant", "content": response.text})
        st.rerun()
        
    except Exception as e:
        if "429" in str(e) or "ResourceExhausted" in str(e):
            st.error("⚠️ Google Gemini Free Tier Rate Limit Reached! Please wait a few minutes, or enter your own API Key in the sidebar to continue.")
        else:
            st.error(f"An error occurred: {e}")
