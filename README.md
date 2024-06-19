# PDF Query Assistant

# Overview
This Streamlit app allows users to upload a PDF file and ask questions about its content. 

It uses the Langchain library for text extraction, chunking, and vector storage, and a Groq language model for answering questions.

# Installation

Install dependencies with

pip install streamlit langchain langchain_community spacy faiss-cpu PyPDF2 langchain_groq groq mysql-connector-python

# Usage

Run the app

streamlit run app.py

Upload a PDF, enter your question, and click "Find Answer" to get responses based on the PDF content.

# Configuration

Groq Language Model: Configure via api_key in ChatGroq. Obtain the API key by signing up for Groq.

Spacy Embedding Model: Set model_name in SpacyEmbeddings.
