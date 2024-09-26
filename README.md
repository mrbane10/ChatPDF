# Chat with PDF

This Streamlit application allows users to upload a PDF document and chat with it. It extracts the content from the PDF, splits it into chunks, generates vector embeddings, and uses a language model to answer questions based on the document context.

## Features
- Upload a PDF document.
- Generate vector embeddings of the document for retrieval.
- Chat with the document using a large language model (LLM).
- Supports clearing conversation history.

## How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
2. **Set up your environment variables:**
   ```bash
   groq_key=YOUR_GROQ_API_KEY
3. **Run the app**
   ```bash
   streamlit run main.py
