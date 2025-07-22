import tempfile
import os
from typing import List, Optional
from pathlib import Path
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import settings

class DocumentService:
    """Service for document loading and processing"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size * 4,  # Convert tokens to characters
            chunk_overlap=settings.chunk_overlap * 4,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_documents(self, uploaded_files) -> List[Document]:
        """Load documents from uploaded files"""
        documents = []
        
        for uploaded_file in uploaded_files:
            try:
                document = self._load_single_document(uploaded_file)
                if document:
                    documents.extend(document)
            except Exception as e:
                st.error(f"Failed to load {uploaded_file.name}: {str(e)}")
        
        return documents
    
    def _load_single_document(self, uploaded_file) -> Optional[List[Document]]:
        """Load a single document file"""
        uploaded_file.seek(0)
        
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=f".{uploaded_file.name.split('.')[-1]}"
        ) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            loader = self._get_loader(uploaded_file.type, tmp_file_path)
            docs = loader.load()
            
            # Add metadata
            for doc in docs:
                doc.metadata.update({
                    'source': uploaded_file.name,
                    'file_type': uploaded_file.type,
                    'original_filename': uploaded_file.name
                })
            
            return docs
            
        finally:
            os.unlink(tmp_file_path)
    
    def _get_loader(self, file_type: str, file_path: str):
        """Get appropriate document loader based on file type"""
        loaders = {
            "application/pdf": PyPDFLoader,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": Docx2txtLoader,
            "text/plain": lambda path: TextLoader(path, encoding='utf-8')
        }
        
        if file_type not in loaders:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return loaders[file_type](file_path)
    
    def chunk_documents(self, documents: List[Document], 
                    chunk_size: Optional[int] = None,
                    overlap: Optional[int] = None) -> List[Document]:
        """Split documents into chunks"""
        
        # Create custom splitter
        if chunk_size or overlap:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=(chunk_size or settings.chunk_size) * 4,
                chunk_overlap=(overlap or settings.chunk_overlap) * 4,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        else:
            splitter = self.text_splitter
        
        chunks = splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': i,
                'chunk_size': len(chunk.page_content),
                'chunk_tokens': len(chunk.page_content.split())
            })
        
        return chunks