import json
import os
import sys
import boto3
import streamlit as st

# Docstring for the entire script
"""
This script uses AWS Bedrock and various LangChain components to create a PDF-based chatbot
that can answer questions from documents. The solution integrates embeddings, vector stores,
and language models (Claude and Llama2) to provide contextually aware answers based on 
the uploaded PDF files.

The following components are included in the script:

1. **Data Ingestion**: The `data_ingestion()` function loads and splits PDF documents 
   into smaller chunks for processing.
   
2. **Vector Embedding and Storage**: The `get_vector_store()` function converts the 
   documents into vector embeddings using Amazon Titan Embeddings and stores them in 
   a FAISS vector store.

3. **Language Models**: Two LLM models are used for answering questions:
   - Claude model (`ai21.j2-mid-v1`) is used for one type of response.
   - Llama2 model (`meta.llama2-70b-chat-v1`) is used for another type of response.

4. **Question Answering**: The `get_response_llm()` function uses the vector store 
   and a model to generate answers based on a given query.

5. **Streamlit Interface**: The script provides a web interface where users can:
   - Ask questions related to the PDF documents.
   - Update or create the vector store by uploading new PDFs and generating embeddings.
   - Get answers from either the Claude or Llama2 model.

## Dependencies:
- `boto3`: AWS SDK for Python to interact with AWS Bedrock services.
- `streamlit`: Web framework to provide the interactive UI for querying PDFs.
- `langchain`: Framework to handle language models, document loading, text splitting, 
  embeddings, and vector stores.
- `faiss`: FAISS vector store for efficient similarity search.

## Workflow:
1. Upload PDFs via the `data` folder.
2. Create or update the vector store.
3. Ask questions using the interactive text input in the Streamlit app.
4. Choose between Claude or Llama2 models to get responses based on the vector store.

## Usage:
1. Install the required dependencies using `pip install`.
2. Run the script with `streamlit run <script_name>.py`.
3. Interact with the application in your browser.
"""

# Code begins here
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding and Vector Store
from langchain.vectorstores import FAISS

# LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

## Data ingestion
def data_ingestion():
    """
    Loads and splits PDF documents from the 'data' directory into smaller chunks for processing.
    
    Returns:
        docs (list): A list of split documents ready for embedding and vectorization.
    """
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # Character split works better for the dataset
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

## Vector Embedding and Vector Store
def get_vector_store(docs):
    """
    Converts documents into vector embeddings using the Titan Embedding model and stores them 
    in a FAISS vector store.
    
    Args:
        docs (list): List of document chunks to be embedded.
    """
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    """
    Creates and returns an instance of the Claude language model from AI21 Labs.
    
    Returns:
        llm (Bedrock): A Bedrock LLM instance for the Claude model.
    """
    llm = Bedrock(model_id="ai21.j2-mid-v1", client=bedrock, model_kwargs={'maxTokens': 512})
    return llm

def get_llama2
