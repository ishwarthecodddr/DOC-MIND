import hashlib
import os
import uuid

import chromadb
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader


# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="DocMind",
    page_icon="🧠",
    layout="wide",
)

load_dotenv()

# Local development -> .env
# Streamlit Cloud -> Secrets
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error(
        "GOOGLE_API_KEY is missing. "
        "Add it to your .env file locally or Streamlit Cloud Secrets."
    )
    st.stop()


# ============================================================
# Gemini Client
# ============================================================

client = genai.Client(api_key=api_key)

EMBEDDING_MODEL = "gemini-embedding-2"
GENERATION_MODEL = "gemini-3.5-flash-lite"


# ============================================================
# ChromaDB
# ============================================================

@st.cache_resource
def get_chroma_client():
    """
    One in-memory Chroma client for the running Streamlit process.

    Individual users get separate collections below, so their
    uploaded documents do not get mixed together.
    """
    return chromadb.Client()


chroma_client = get_chroma_client()


# ============================================================
# Session-specific collection
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

collection_name = f"docmind_{st.session_state.session_id}"

collection = chroma_client.get_or_create_collection(
    name=collection_name
)


# ============================================================
# Gemini Embeddings
# ============================================================
def embed_texts(texts):
    """
    Generate one embedding for each text.

    Gemini's embedding response can behave differently depending
    on how multiple contents are passed. Generating each embedding
    individually guarantees a 1:1 mapping between documents and
    embeddings.
    """

    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        return []

    embeddings = []

    for text in texts:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )

        if not response.embeddings:
            raise ValueError(
                "Gemini returned no embedding for a document chunk."
            )

        embeddings.append(
            response.embeddings[0].values
        )

    return embeddings
# ============================================================
# PDF Processing
# ============================================================

def extract_text_from_pdf(file):
    """
    Extract text from a PDF while preserving page numbers.
    """

    reader = PdfReader(file)

    pages_content = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        text = text.strip()

        if text:
            pages_content.append(
                {
                    "text": text,
                    "page": page_number,
                }
            )

    return pages_content


# ============================================================
# Chunking
# ============================================================

def chunk_text(
    pages_content,
    filename,
    chunk_size=1200,
    overlap=200,
):
    """
    Split page text into overlapping chunks while preserving
    filename and page metadata.
    """

    chunks = []

    step = chunk_size - overlap

    for page in pages_content:
        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):
            end = start + chunk_size

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(
                    {
                        "text": chunk,
                        "metadata": {
                            "source": filename,
                            "page": page_number,
                        },
                    }
                )

            start += step

    return chunks


# ============================================================
# Document ID
# ============================================================

def get_document_id(uploaded_file):
    """
    Create a deterministic ID from the uploaded PDF.

    This prevents the same PDF from being indexed repeatedly
    during the same session.
    """

    file_bytes = uploaded_file.getvalue()

    return hashlib.sha256(file_bytes).hexdigest()


# ============================================================
# Document Indexing
# ============================================================

def process_documents(uploaded_files):
    """
    Extract, chunk, embed and store uploaded PDFs.
    """

    total_chunks = 0

    for uploaded_file in uploaded_files:

        with st.spinner(f"Processing {uploaded_file.name}..."):

            document_id = get_document_id(uploaded_file)

            pages = extract_text_from_pdf(uploaded_file)

            if not pages:
                st.warning(
                    f"No extractable text found in {uploaded_file.name}."
                )
                continue

            chunks = chunk_text(
                pages,
                uploaded_file.name,
            )

            if not chunks:
                st.warning(
                    f"No text chunks could be created from "
                    f"{uploaded_file.name}."
                )
                continue

            # Remove previous version of this document
            # if it was already indexed.
            try:
                collection.delete(
                    where={
                        "document_id": document_id
                    }
                )
            except Exception:
                pass

            documents = [
                chunk["text"]
                for chunk in chunks
            ]

            metadatas = [
                {
                    **chunk["metadata"],
                    "document_id": document_id,
                }
                for chunk in chunks
            ]

            # Deterministic IDs instead of random UUIDs.
            ids = [
                f"{document_id}_{index}"
                for index in range(len(chunks))
            ]

            embeddings = embed_texts(documents)

            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

            total_chunks += len(chunks)

    if total_chunks:
        st.success(
            f"Indexed {total_chunks} chunks successfully."
        )


# ============================================================
# RAG Answer Generation
# ============================================================

def get_answer(query):
    """
    Retrieve relevant document chunks and generate a grounded
    answer using Gemini.
    """

    if collection.count() == 0:
        return (
            "Please upload and index at least one PDF before "
            "asking a question."
        )

    query_embedding = embed_texts(query)[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(5, collection.count()),
        include=[
            "documents",
            "metadatas",
        ],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return (
            "I couldn't find relevant information in the "
            "indexed documents."
        )

    context_parts = []

    for document, metadata in zip(documents, metadatas):

        source = metadata.get(
            "source",
            "Unknown source",
        )

        page = metadata.get(
            "page",
            "Unknown",
        )

        context_parts.append(
            f"[Source: {source}, Page: {page}]\n"
            f"{document}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are DocMind, an academic document assistant.

Your job is to answer the user's question using ONLY
the provided document context.

RULES:

1. Do not use outside knowledge.
2. If the answer cannot be found in the context,
   clearly say that the indexed documents do not contain
   enough information to answer the question.
3. Every factual claim should be supported by a citation.
4. Use exactly this citation format:
   [Source: filename, Page: X]
5. Do not invent sources or page numbers.
6. Be concise but sufficiently explanatory.
7. Use Markdown when helpful.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{query}

ANSWER:
"""

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    return response.text


# ============================================================
# UI
# ============================================================

st.title("🧠 DocMind")

st.markdown(
    "### Context-Grounded Academic Assistant"
)

st.caption(
    "Upload academic PDFs, index them, and ask questions "
    "with source-aware answers."
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("📚 Document Ingestion")

    uploaded_files = st.file_uploader(
        "Upload lecture PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button(
        "Index Documents",
        use_container_width=True,
    ):

        if uploaded_files:

            try:
                process_documents(uploaded_files)

            except Exception as error:
                st.error(
                    f"Indexing failed: {error}"
                )

        else:

            st.warning(
                "Please upload at least one PDF."
            )

    st.divider()

    st.caption(
        f"Indexed chunks: {collection.count()}"
    )

    st.caption(
        "Documents are isolated per session and are "
        "cleared when the app restarts."
    )


# ============================================================
# Chat History
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# Chat Input
# ============================================================

if prompt := st.chat_input(
    "Ask a question about your documents..."
):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Searching your documents..."):

            try:

                answer = get_answer(prompt)

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as error:

                error_message = (
                    f"Something went wrong: {error}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )