# DocMind 🧠

A context-grounded academic assistant that lets you upload
PDF documents and ask questions about them using Retrieval-
Augmented Generation (RAG).

## ✨ Features

- 📄 Upload academic PDFs
- 🔍 Semantic search using vector embeddings
- 🧠 Gemini-powered question answering
- 📚 Source-aware answers with page references
- 🔒 Session-isolated document collections
- 💬 Streamlit chat interface
- ⚡ Fast local vector retrieval with ChromaDB

## 🏗️ Architecture

```text
                    ┌─────────────────┐
                    │    Streamlit    │
                    │       UI        │
                    └────────┬────────┘
                             │
                       Upload PDF
                             │
                             ▼
                    ┌─────────────────┐
                    │  PDF Extraction │
                    │     pypdf       │
                    └────────┬────────┘
                             │
                          Chunks
                             │
                             ▼
                    ┌─────────────────┐
                    │ Gemini Embedding│
                    │       2        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    ChromaDB     │
                    │ Vector Search   │
                    └────────┬────────┘
                             │
                       Top Results
                             │
                             ▼
                    ┌─────────────────┐
                    │      Gemini     │
                    │  Answer Model   │
                    └────────┬────────┘
                             │
                             ▼
                    Grounded Answer
                    + Source Citations