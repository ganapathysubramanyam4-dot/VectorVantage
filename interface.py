import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader

# 1. Configure Gemini API Key from Streamlit Secrets
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("API Key not found. Please add GEMINI_API_KEY to your Streamlit Secrets.")

# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# 2. Streamlit UI Layout
st.set_page_config(page_title="VectorVantage RAG AI", layout="wide")
st.title("VectorVantage: PDF AI Assistant 📚🚀")
st.write("Upload your PDF documents and ask questions based on their content.")

# Sidebar for PDF Uploads
with st.sidebar:
    st.subheader("Your Documents")
    pdf_docs = st.file_uploader(
        "Upload your PDF files here", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if st.button("Process Files"):
        if pdf_docs:
            with st.spinner("Processing PDF files..."):
                # Store the extracted text into Streamlit session state
                st.session_state.raw_text = get_pdf_text(pdf_docs)
                st.success("PDF files processed successfully!")
        else:
            st.warning("Please upload at least one PDF file first.")

# 3. Main Chat Interface
user_question = st.text_input("Ask a question about your uploaded PDFs:")

if st.button("Ask AI"):
    if user_question:
        # Check if text exists in session state
        if "raw_text" in st.session_state and st.session_state.raw_text:
            with st.spinner("Searching for answers..."):
                try:
                    # Structured prompt to feed context to Gemini
                    prompt = f"""
                    You are a professional AI assistant. Answer the user's question accurately based only on the provided context.
                    If the answer cannot be found or inferred from the context, explicitly state: "The answer to your question is not available in the provided PDF documents."
                    
                    Context:
                    {st.session_state.raw_text}
                    
                    Question: {user_question}
                    Answer:
                    """
                    
                    # Initialize Gemini 1.5 Flash
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    response = model.generate_content(prompt)
                    
                    # Display the response
                    st.success("Response:")
                    st.write(response.text)
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.info("Please upload your PDF documents and click 'Process Files' in the sidebar first.")
    else:
        st.warning("Please enter a question before clicking Ask AI.")
