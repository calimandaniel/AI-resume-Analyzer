import streamlit as st
from io import BytesIO
import pandas as pd
import tempfile
import os
import re
from typing import List, Dict
import numpy as np

import asyncio
from asyncio import new_event_loop, set_event_loop, run

# Updated LangChain imports
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
# Existing imports

# Load environment variables
from dotenv import load_dotenv
current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.dirname(current_dir)
env_path = os.path.join(src_root, '.env')

if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"Loaded .env from: {env_path}")
else:
    load_dotenv()
    print("Using default .env loading")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.error("Google Generative AI not available. Install with: pip install google-generativeai")
    
class ResumeUploader:
    def __init__(self):
        self.uploaded_files = []
        self.processed_documents = []
        self.chunks = []
        self.embeddings = []
        self.embedding_method = "langchain_gemini"
        
        # LangChain components
        self.langchain_embeddings = None
        self.text_splitter = None
        self.vectorstore = None
        
        # Initialize LangChain components
        self._initialize_langchain_components()
    
    def _initialize_langchain_components(self):
        """Initialize LangChain components"""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                # Create a coroutine function for async operations
                async def initialize_async_components():
                    # Initialize embeddings
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=api_key
                    )
                    
                    # Initialize vector store if database URL exists
                    vectorstore = None
                    db_url = os.getenv("DATABASE_URL")
                    if db_url:
                        vectorstore = PGVector(
                            connection=db_url,
                            embeddings=embeddings,
                            collection_name="resume_langchain"
                        )
                    
                    return embeddings, vectorstore
                
                # Create and run event loop
                loop = new_event_loop()
                set_event_loop(loop)
                embeddings, vectorstore = run(initialize_async_components())
                
                # Assign to instance variables
                self.langchain_embeddings = embeddings
                self.vectorstore = vectorstore
                  
                # Initialize text splitter (no async needed)
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=400,  # tokens approximately
                    chunk_overlap=50,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
            
        except Exception as e:
            st.warning(f"LangChain initialization failed: {str(e)}")
            self.langchain_embeddings = None
            
    def load_documents_with_langchain(self, uploaded_files) -> List[Document]:
        """Load documents using LangChain loaders"""
        documents = []
        
        for uploaded_file in uploaded_files:
            try:
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                # Load document based on file type
                if uploaded_file.type == "application/pdf":
                    loader = PyPDFLoader(tmp_file_path)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    loader = Docx2txtLoader(tmp_file_path)
                elif uploaded_file.type == "text/plain":
                    loader = TextLoader(tmp_file_path, encoding='utf-8')
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.type}")
                    continue
                
                # Load documents
                docs = loader.load()
                
                # Add metadata
                for doc in docs:
                    doc.metadata.update({
                        'source': uploaded_file.name,
                        'file_type': uploaded_file.type,
                        'original_filename': uploaded_file.name
                    })
                    documents.append(doc)
                
                # Clean up
                os.unlink(tmp_file_path)
                st.success(f"Loaded with LangChain: {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {str(e)}")
        
        return documents

    def process_documents_with_langchain(self, documents: List[Document], chunk_size: int = 400, overlap: int = 50) -> List[Document]:
        """Process documents using LangChain text splitter"""
        try:
            # Update text splitter parameters
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size * 4,  # Convert tokens to characters 
                chunk_overlap=overlap * 4,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Split documents
            chunks = self.text_splitter.split_documents(documents)
            
            # Add additional metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': i,
                    'chunk_size': len(chunk.page_content),
                    'chunk_tokens': len(chunk.page_content.split()),
                    'processing_method': 'langchain'
                })
            
            st.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
            return chunks
            
        except Exception as e:
            st.error(f"Error processing documents with LangChain: {str(e)}")
            return []

    def save_to_langchain_vectorstore(self, chunks: List[Document]) -> bool:
        """Save chunks to LangChain vector store"""
        try:
            if not self.vectorstore:
                st.error("Vector store not initialized")
                return False
            
            with st.spinner(f"Saving {len(chunks)} chunks to LangChain vector store..."):
                # Add documents to vector store
                self.vectorstore.add_documents(chunks)
            
            st.success(f"✅ Saved {len(chunks)} chunks to LangChain vector store")
            return True
            
        except Exception as e:
            st.error(f"❌ Error saving to LangChain vector store: {str(e)}")
            return False
        
    def upload_resume(self):
        st.header("Upload Resumes")
        st.write("Upload multiple resume files (PDF, DOCX, or TXT format)")
        
        col1, col2 = st.columns([1, 1])
        # Chunking parameters
        st.subheader("Chunking Configuration")
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider("Chunk Size (tokens)", 100, 500, 400, 25)
        with col2:
            overlap_size = st.slider("Overlap Size (tokens)", 0, 100, 50, 10)
        
        uploaded_files = st.file_uploader(
            "Choose resume files", 
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"Successfully uploaded {len(uploaded_files)} file(s)!")
            
            # Clear previous uploads
            self.uploaded_files = []
            self.processed_documents = []
            self.chunks = []
            self.embeddings = []
            
            for uploaded_file in uploaded_files:
                st.write(f"📄 {uploaded_file.name}")
                
                self.uploaded_files.append({
                    'name': uploaded_file.name,
                    'content': uploaded_file,
                    'type': uploaded_file.type
                })
            
            # Single LangChain processing button
            if st.button("Process with LangChain"):
                self._process_with_langchain(uploaded_files, chunk_size, overlap_size)

    def _process_with_langchain(self, uploaded_files, chunk_size: int, overlap_size: int):
        """Process documents using LangChain"""
        
        if not self.langchain_embeddings:
            st.error("LangChain embeddings not initialized. Please check API key.")
            return
        
        progress_bar = st.progress(0)
        
        try:
            # Step 1: Load documents with LangChain
            st.info("Loading documents with LangChain...")
            documents = self.load_documents_with_langchain(uploaded_files)
            progress_bar.progress(0.3)
            
            if not documents:
                st.error("No documents were loaded successfully")
                return
            
            # Step 2: Process and chunk documents
            st.info("Chunking documents with LangChain...")
            chunks = self.process_documents_with_langchain(documents, chunk_size, overlap_size)
            progress_bar.progress(0.6)
            
            if not chunks:
                st.error("No chunks were created")
                return
            
            # Step 3: Save to vector store
            st.info("Saving to LangChain vector store...")
            success = self.save_to_langchain_vectorstore(chunks)
            progress_bar.progress(0.9)
            
            if success:
                progress_bar.progress(1.0)
                
                # Display results
                self._display_langchain_results(documents, chunks)
                
            else:
                st.error("Failed to save to vector store")
        
        except Exception as e:
            st.error(f"Error in LangChain processing: {str(e)}")


    def _display_langchain_results(self, documents: List[Document], chunks: List[Document]):
        """Display results from LangChain processing"""
        st.subheader("🎉 LangChain Processing Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Documents Loaded", len(documents))
        with col2:
            st.metric("Total Chunks", len(chunks))
        with col3:
            total_chars = sum(len(chunk.page_content) for chunk in chunks)
            st.metric("Total Characters", f"{total_chars:,}")
        with col4:
            avg_chunk_size = total_chars / len(chunks) if chunks else 0
            st.metric("Avg Chunk Size", f"{avg_chunk_size:.0f}")
        
        # Document summary
        doc_summary = {}
        for chunk in chunks:
            source = chunk.metadata.get('source', 'Unknown')
            if source not in doc_summary:
                doc_summary[source] = {'chunks': 0, 'total_chars': 0}
            doc_summary[source]['chunks'] += 1
            doc_summary[source]['total_chars'] += len(chunk.page_content)
        
        st.subheader("Document Summary")
        summary_data = []
        for source, data in doc_summary.items():
            summary_data.append({
                'Filename': source,
                'Chunks': data['chunks'],
                'Total Characters': f"{data['total_chars']:,}",
                'Avg Chunk Size': f"{data['total_chars'] / data['chunks']:.0f}",
                'Status': 'Processed'
            })
        
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)
        
        # Sample chunks viewer
        with st.expander("🔍 View Sample Chunks"):
            if chunks:
                selected_source = st.selectbox(
                    "Select document:",
                    list(doc_summary.keys())
                )
                
                source_chunks = [c for c in chunks if c.metadata.get('source') == selected_source]
                
                for i, chunk in enumerate(source_chunks[:3]):  # Show first 3 chunks
                    st.subheader(f"Chunk {i+1}")
                    st.text_area(
                        f"Content (Chunk {chunk.metadata.get('chunk_id', i)}):",
                        chunk.page_content,
                        height=150,
                        key=f"langchain_chunk_{i}"
                    )
                    st.json(chunk.metadata)
        
        st.success(f"🎉 Successfully processed {len(documents)} documents into {len(chunks)} chunks using LangChain!")