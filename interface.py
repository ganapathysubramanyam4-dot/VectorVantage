import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader

# Streamlit Page Configuration
st.set_page_config(page_title="Gemini AI Multitasking Chatbot", page_icon="🤖", layout="wide")

# --- CUSTOM CSS DESIGN ---
# Making the chat UI look modern, clean, and professional (ChatGPT-inspired layout)
st.markdown("""
    <style>
    /* Styling the Chat Input Box to be pinned nicely at the bottom */
    .stChatInput {
        position: fixed;
        bottom: 20px;
        z-index: 999;
    }
    /* Adding subtle styling to the sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    /* Main container spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 7rem;
    }
    </style>
""", unsafe_allowed_html=True)

# 1. Fetch API Key from Streamlit Secrets
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("Please add GEMINI_API_KEY to your Streamlit Secrets!")
    st.stop()

# 2. Initialize Session States (Memory & Context)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_context" not in st.session_state:
    st.session_state.pdf_context = ""
if "pdf_summary" not in st.session_state:
    st.session_state.pdf_summary = ""
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []

# --- SIDEBAR FEATURES ---
with st.sidebar:
    st.title("⚙️ Settings & Tools")
    
    # Model Selector
    selected_model = st.selectbox("Select Model", ["gemini-2.5-flash", "gemini-2.5-pro"])
    
    # Clear Chat Button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    # --- NEW FEATURE: DOWNLOAD CHAT HISTORY ---
    if st.session_state.messages:
        # Preparing the chat history text format
        chat_download_text = "--- GEMINI CHAT SESSION LOG ---\n\n"
        for msg in st.session_state.messages:
            role_label = "User" if msg["role"] == "user" else "Gemini AI"
            chat_download_text += f"[{role_label}]: {msg['content']}\n\n"
            chat_download_text += "-"*40 + "\n\n"
        
        # Streamlit Download Button
        st.download_button(
            label="📥 Download Chat History",
            data=chat_download_text,
            file_name="gemini_chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )
        
    st.write("---")
    
    # MULTIPLE PDF Upload Feature
    st.subheader("📄 PDF Document Analysis")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        current_file_names = [f.name for f in uploaded_files]
        
        if current_file_names != st.session_state.uploaded_file_names:
            with st.spinner("Extracting text from all uploaded PDFs..."):
                combined_text = ""
                try:
                    for uploaded_file in uploaded_files:
                        reader = PdfReader(uploaded_file)
                        for page in reader.pages:
                            text = page.extract_text()
                            if text:
                                combined_text += text + "\n"
                    
                    st.session_state.pdf_context = combined_text
                    st.session_state.uploaded_file_names = current_file_names
                    st.session_state.pdf_summary = ""  
                    st.success(f"Successfully processed {len(uploaded_files)} PDF(s)!")
                except Exception as e:
                    st.error(f"Error reading PDF files: {e}")
    
    # Clear PDF Button
    if st.session_state.pdf_context and st.button("❌ Remove All PDF Context", use_container_width=True):
        st.session_state.pdf_context = ""
        st.session_state.pdf_summary = ""
        st.session_state.uploaded_file_names = []
        st.success("All PDF contexts removed!")
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("🤖 My Advanced Gemini Chatbot")

# Display PDF Status & Automatic Summary if attached
if st.session_state.pdf_context:
    st.info(f"💡 **Active Context:** {len(st.session_state.uploaded_file_names)} PDF(s) active in memory!")
    st.caption(f"Active Files: {', '.join(st.session_state.uploaded_file_names)}")
    
    with st.expander("✨ Click here to view Combined PDF Summary", expanded=True):
        if not st.session_state.pdf_summary:
            with st.spinner("Generating Summary for all documents..."):
                try:
                    summary_model = genai.GenerativeModel(selected_model)
                    summary_response = summary_model.generate_content(
                        f"Provide a concise combined summary and key highlights for the following text extracted from multiple documents:\n\n{st.session_state.pdf_context}"
                    )
                    st.session_state.pdf_summary = summary_response.text
                except Exception as e:
                    st.session_state.pdf_summary = f"Could not generate summary: {e}"
        st.markdown(st.session_state.pdf_summary)

# Display Past Chat Messages from History
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
        
        if st.session_state.pdf_context:
            system_prompt = f"You are a helpful assistant. Use the following content extracted from multiple PDFs to answer the user's questions if relevant:\n\n{st.session_state.pdf_context}"
            formatted_history.append({"role": "user", "parts": [system_prompt]})
            formatted_history.append({"role": "model", "parts": ["Understood. I will use the provided content from all PDFs and chat history to assist you."]})
        
        for msg in st.session_state.messages:
            role = "user" if msg["role"] == "user" else "model"
            formatted_history.append({"role": role, "parts": [msg["content"]]})
            
        with st.spinner("Thinking..."):
            response = model.generate_content(formatted_history)
        
        with st.chat_message("assistant"):
            st.markdown(response.text)
            
        st.session_state.messages.append({"role": "assistant", "content": response.text})
        st.rerun() # Forces page refresh to show the download button as soon as chat updates
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
