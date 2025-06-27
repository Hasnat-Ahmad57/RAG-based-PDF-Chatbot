📄 RAG-based PDF Chatbot (LangChain + Gemini Pro)

Built an intelligent chatbot interface capable of answering questions from one or multiple uploaded PDF documents using Retrieval-Augmented Generation (RAG). Designed for efficient document search, context extraction, and interactive query handling.

🛠️ Tools & Tech:

Python, Streamlit, LangChain, FAISS, PyPDF2

Google Gemini 1.5 (via langchain_google_genai)

Embedding Model: Google embedding-001

PDF Parsing, Semantic Search, Conversational Chain

🔍 Features:

Upload and query any number of PDF files simultaneously.

Uses RecursiveCharacterTextSplitter for optimized chunking and FAISS for fast similarity search.

Custom prompt template ensures reliable answers and avoids hallucinations.

Built-in memory of chat history, complete with timestamps and PDF context.

Beautiful Streamlit-based chat UI with avatars, dark styling, and CSV export of conversation logs.

Reset, Rerun, and Conversation Clearing functionalities for a seamless user experience.

🚀 How to Run Locally:

Make sure you have Python & Streamlit installed:
pip install streamlit langchain PyPDF2 faiss-cpu pandas

Save the code in a file named app.py

Launch the app:
streamlit run app.py

Get your Gemini API key from:
https://ai.google.dev/

Paste your API key into the sidebar, upload one or more PDFs, and start asking questions.

🗂️ Use Cases:

Academic research document assistants

Legal document exploration

Business report question-answering

Personal knowledge base navigator
