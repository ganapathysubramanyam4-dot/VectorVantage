import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# 1. Function to read all PDFs from a folder
def load_all_pdfs(folder_path):
    all_text = ""
    # Checking every file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            print(f"Reading: {filename}")
            file_path = os.path.join(folder_path, filename)
            reader = PdfReader(file_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text += text
    return all_text

# Define the folder name where your PDFs are stored
folder_name = "my_documents" 

# Create the folder if it doesn't exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"'{folder_name}' folder created. Please put your PDF files inside it.")

# Load text from all PDFs
full_text = load_all_pdfs(folder_name)

if not full_text:
    print("No PDF files found! Please add PDFs to the 'my_documents' folder and run again.")
else:
    print(f"\nTotal characters extracted: {len(full_text)}")

    # 2. Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(full_text)
    print(f"Total chunks created: {len(chunks)}") 

    # 3. Embeddings & Creating FAISS Vector Database
    print("Initializing embeddings... please wait.")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_texts(chunks, embeddings)
    print("Vector database created successfully.")
    
    # 4. Initializing Ollama (Mistral)
    llm = Ollama(model="mistral")

    # 5. Interaction Loop
    while True:
        query = input("\nASK YOUR QUESTION (Type 'exit' to stop): ")
        
        if query.lower() == 'exit':
            print("Thank you! Goodbye.")
            break
            
        # Search the local database for relevant context
        docs = vector_db.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        # Creating the prompt for Ollama
        prompt = f"Using the following context:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        print("\nOLLAMA IS THINKING...")
        response = llm.invoke(prompt)

        print("\nANSWER FROM MISTRAL:")
        print(response)
        print("\n" + "-"*50)