import streamlit as st
import os
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

# --- Environment Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. Page Configuration
st.set_page_config(page_title="VectorVantage AI", layout="wide")

# --- Optimized Embeddings Loading ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

embeddings = load_embeddings()
DB_FAISS_PATH = 'faiss_index'

# --- SideBar: Document Management & ML Status ---
with st.sidebar:
    st.title("🚀 VectorVantage")
    st.markdown("---")
    
    st.header("⚙️ ML System Status")
    st.info("""
    **Search:** Hybrid (BM25 + FAISS)  
    **Memory:** Context Re-writing  
    **Model:** Mistral-7B  
    """)
    
    st.divider()
    st.header("📂 Document Management")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Building Intelligent Index..."):
                all_docs = []
                if not os.path.exists("temp_pdf"):
                    os.makedirs("temp_pdf")
                
                for uploaded_file in uploaded_files:
                    file_path = os.path.join("temp_pdf", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    loader = PyPDFLoader(file_path)
                    data = loader.load()
                    all_docs.extend(data)
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
                splits = text_splitter.split_documents(all_docs)
                
                db = FAISS.from_documents(splits, embeddings)
                db.save_local(DB_FAISS_PATH)
                st.session_state.splits = splits
                st.success("Indexing Successful!")
        else:
            st.error("Please upload a PDF.")

    st.divider()
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.chat_history = ""
        st.rerun()

# --- Main UI with Tabs ---
st.title("📑 VectorVantage AI Assistant")
tab1, tab2 = st.tabs(["💬 Smart Chat", "📊 Document Insights"])

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

with tab1:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask VectorVantage..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        if os.path.exists(DB_FAISS_PATH) and "splits" in st.session_state:
            # அசிஸ்டென்ட் லோட் ஆகுறதுக்கு முன்னாடி llm-ஐ CPU-க்கு போர்ஸ் பண்றோம்
            llm = Ollama(model="mistral", base_url="http://127.0.0.1:11434")
            
            # STEP 1: CONTEXTUAL QUERY RE-WRITING
            search_query = user_query
            if st.session_state.chat_history:
                try:
                    rewrite_prompt = f"Based on this history: {st.session_state.chat_history}. Re-write this question as a search query: {user_query}. Just give the question, nothing else."
                    search_query = llm.invoke(rewrite_prompt).strip()
                except:
                    search_query = user_query

            with st.chat_message("assistant", avatar="🧠"):
                st.markdown("### VectorVantage")
                
                db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
                vector_docs = db.similarity_search(search_query, k=2)
                
                bm25 = BM25Retriever.from_documents(st.session_state.splits)
                bm25.k = 2
                keyword_docs = bm25.invoke(search_query)
                
                combined_docs = vector_docs + keyword_docs
                context = "\n".join([d.page_content for d in combined_docs])
                source_info = [f"{os.path.basename(d.metadata.get('source', 'Unknown'))} (Page: {d.metadata.get('page',0)+1})" for d in combined_docs]

                response_placeholder = st.empty()
                full_response = ""
                
                final_prompt = f"Context: {context}\nHistory: {st.session_state.chat_history}\nQuestion: {user_query}\nAnswer:"
                
                try:
                    for chunk in llm.stream(final_prompt):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                except Exception as e:
                    st.error(f"Ollama Connection Error. Please ensure 'ollama serve' is running in CMD.")

                with st.expander("📚 Reference Sources"):
                    for info in list(set(source_info)):
                        st.write(f"📍 {info}")
                
                st.session_state.chat_history += f"\nUser: {user_query}\nAI: {full_response}\n"
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.warning("Please process documents first!")

with tab2:
    st.header("🔍 Automated Document Analysis")
    if "splits" in st.session_state:
        st.write(f"**Total Text Chunks Analyzed:** {len(st.session_state.splits)}")
        if st.button("Generate Key Insights"):
            with st.spinner("Analyzing..."):
                try:
                    insight_llm = Ollama(model="mistral", base_url="http://127.0.0.1:11434")
                    sample = st.session_state.splits[0].page_content[:800] # மெமரி எர்ரர் வராம இருக்க அளவு குறைச்சுருக்கேன்
                    insight = insight_llm.invoke(f"Summarize this in 5 short points: {sample}")
                    st.info(insight)
                except:
                    st.error("Failed to generate insights. Check if Ollama is running.")
    else:
        st.info("Upload and process a document to see insights here.")