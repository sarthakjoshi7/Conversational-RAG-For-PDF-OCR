# Conversational RAG For PDF + OCR

A Conversational Retrieval Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content. 
The system automatically handles both text based PDFs and scanned PDFs using OCR technique.

## Project Goal

The goal of this project is to build an intelligent document question answering system capable of handling both text based and scanned PDF documents.
Traditional RAG applications work well with digital PDFs but struggle with scanned documents because the text is embedded inside images. 
This project addresses that limitation by integrating OCR with a Conversational RAG pipeline. The system automatically detects whether a PDF contains extractable text and falls back to OCR when required.

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

- PDF is uploaded.
- Text is extracted directly using PyPDFLoader.
- Text is split into chunks.
- Embeddings are generated.
- Chunks are stored in ChromaDB.
- User questions are answered through Retrieval Augmented Generation.

### Scanned PDF

- PDF is uploaded.
- Direct text extraction is attempted.
- If no text is found, PaddleOCR is used.
- Extracted OCR text is converted into chunks.
- Embeddings are generated.
- Chunks are stored in ChromaDB.
- User questions are answered through Retrieval Augmented Generation.

End of project.
