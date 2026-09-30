import streamlit as st
import google.generativeai as genai
import chromadb
from pypdf import PdfReader
import os
from dotenv import load_dotenv
import uuid


def embed_texts(texts):
    """Generate embeddings using the installed Google SDK."""
    if isinstance(texts, str):
        texts = [texts]

    response = genai.embed_content(
        model="models/text-embedding-004",
        content=texts,
        task_type="retrieval_document"
    )

    if isinstance(response, dict) and "embedding" in response:
        return response["embedding"]
    if isinstance(response, list):
        return response
    return response

# --- Configuration & Initialization ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Please set your GOOGLE_API_KEY in the .env file.")
    st.stop()

genai.configure(api_key=api_key)

# Initialize ChromaDB
@st.cache_resource
def get_vector_db():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name="docmind_collection")

collection = get_vector_db()

# --- Utility Functions ---

def extract_text_from_pdf(file):
    """Extracts text from a PDF file along with page numbers."""
    reader = PdfReader(file)
    pages_content = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            pages_content.append({"text": text, "page": i + 1})
    return pages_content

def chunk_text(pages_content, filename, chunk_size=1000, overlap=200):
    """Chunks text into segments while maintaining metadata."""
    chunks = []
    for page in pages_content:
        text = page["text"]
        page_num = page["page"]
        
        # Simple sliding window chunking
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "metadata": {
                    "source": filename,
                    "page": page_num
                }
            })
            start += (chunk_size - overlap)
    return chunks

def process_documents(uploaded_files):
    """Extracts, chunks, and adds documents to the vector store."""
    for uploaded_file in uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            pages = extract_text_from_pdf(uploaded_file)
            chunks = chunk_text(pages, uploaded_file.name)

            ids = [c["id"] for c in chunks]
            documents = [c["text"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]
            embeddings = embed_texts(documents)

            collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    st.success("All documents processed and indexed!")

def get_answer(query):
    """Retrieves context and generates a grounded answer using Gemini."""
    query_embedding = embed_texts(query)[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=5, include=["documents", "metadatas"])

    context_list = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_list.append(f"[Source: {meta['source']}, Page: {meta['page']}]\n{doc}")

    context = "\n\n---\n\n".join(context_list)
    
    # Generate Answer
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = f"""
    You are DocMind, an expert academic assistant. Your goal is to answer the user's question based ONLY on the provided context.
    
    RULES:
    1. If the answer is not in the context, explicitly state that you don't have enough information.
    2. ALWAYS cite your source using the [Source: filename, Page: X] format provided in the context.
    3. Keep the answer clear, professional, and helpful.
    4. Use Markdown for formatting (bolding, lists, etc.).

    CONTEXT:
    {context}

    USER QUESTION:
    {query}
    
    ANSWER:
    """
    
    response = model.generate_content(prompt)
    return response.text

# --- Streamlit UI ---

st.set_page_config(page_title="DocMind", page_icon="🧠", layout="wide")

st.title("🧠 DocMind")
st.markdown("### Context-Grounded Academic Assistant")

# Sidebar for Uploads
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_files = st.file_uploader("Upload lecture PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Index Documents"):
        if uploaded_files:
            process_documents(uploaded_files)
        else:
            st.warning("Please upload at least one PDF.")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = get_answer(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error generating answer: {e}")
