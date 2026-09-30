# DocMind 🧠

A Retrieval-Augmented Generation (RAG) system for context-grounded question answering over academic documents.

## Features
- **PDF Ingestion:** Upload and index lecture notes, slides, and textbooks.
- **Local Vector Store:** Uses ChromaDB for persistent, local semantic search.
- **Grounded Answers:** Uses Gemini 1.5 Flash to synthesize answers strictly from your documents.
- **Citations:** Every answer includes source filename and page number references.
- **Minimalist UI:** Built with Streamlit for a clean, chat-focused experience.

## Setup Instructions

### 1. Clone/Download the Project
Ensure you are in the project directory: `C:\Users\KIIT0001\Documents\SEM7\genAI`

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
1. Create a `.env` file in the root directory (or rename `.env.example`).
2. Add your Google Gemini API Key:
   ```text
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

### 4. Run the Application
```bash
streamlit run app.py
```

## How to Use
1. **Upload:** Use the sidebar to upload one or more academic PDFs.
2. **Index:** Click "Index Documents" to process and store them in the local vector database.
3. **Chat:** Ask questions in the chat box. DocMind will retrieve the relevant sections and answer with citations.
