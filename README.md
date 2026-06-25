# Conversational RAG For PDF + OCR

A Conversational Retrieval Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content. 
The system automatically handles both text based PDFs and scanned PDFs using OCR.

## Project Goal

The goal of this project is to build an intelligent document question answering system capable of handling both text based and scanned PDF documents.

## Features

- Upload one or more PDF files
- Automatic detection of text based PDFs
- OCR support for scanned PDFs using PaddleOCR
- Conversational question answering with chat history
- Vector search using ChromaDB
- Embeddings using Hugging Face
- LLM powered responses using Open Source LLM API
- Automatic PDF reprocessing when new files are uploaded
- User interface

## Tech Stack

- Python
- LangChain
- Streamlit
- ChromaDB
- Hugging Face Embeddings
- Open Source LLM
- PaddleOCR
- PaddlePaddle
- PyMuPDF

## How It Works

### Text Based PDF

1. PDF is uploaded.
2. Text is extracted directly using PyPDFLoader.
3. Text is split into chunks.
4. Embeddings are generated.
5. Chunks are stored in ChromaDB.
6. User questions are answered through Retrieval Augmented Generation.

### Scanned PDF

1. PDF is uploaded.
2. Direct text extraction is attempted.
3. If no text is found, PaddleOCR is used.
4. Extracted OCR text is converted into chunks.
5. Embeddings are generated.
6. Chunks are stored in ChromaDB.
7. User questions are answered through Retrieval Augmented Generation.

End of project.
