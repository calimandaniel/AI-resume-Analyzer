import asyncio
from asyncio import new_event_loop, set_event_loop, run
from typing import List, Dict, Optional
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
import streamlit as st

from config import settings

class VectorService:
    """Service for vector store operations"""
    
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self._initialize()
    
    def _initialize(self):
        """Initialize embeddings and vector store"""
        try:
            if not settings.google_api_key:
                st.error("GOOGLE_API_KEY not found")
                return
            
            if not settings.database_url:
                st.error("DATABASE_URL not found")
                return
            
            # Create a coroutine function for async operations
            async def initialize_async_components():
                # Initialize embeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=settings.embedding_model,
                    google_api_key=settings.google_api_key
                )
                
                # Initialize vector store
                vector_store = PGVector(
                    connection=settings.database_url,
                    embeddings=embeddings,
                    collection_name="resume_langchain"
                )
                
                return embeddings, vector_store
            
            # Create and run event loop
            loop = new_event_loop()
            set_event_loop(loop)
            embeddings, vector_store = run(initialize_async_components())
            
            # Assign to instance variables
            self.embeddings = embeddings
            self.vector_store = vector_store
        except Exception as e:
            st.error(f"Failed to initialize vector store: {str(e)}")
            
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to vector store"""
        try:
            self.vector_store.add_documents(documents)
            return True
        except Exception as e:
            st.error(f"Failed to add documents: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 5, 
                         filters: Optional[Dict] = None) -> List[Document]:
        """Search for similar documents"""
        try:
            if filters:
                return self.vector_store.similarity_search(query, k=k, filter=filters)
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            st.error(f"Failed to search vector store: {str(e)}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Search for similar documents with similarity scores"""
        try:
            return self.vector_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            st.error(f"Failed to search with scores: {str(e)}")
            return []
    
    def is_ready(self) -> bool:
        """Check if vector store is ready"""
        return self.embeddings is not None and self.vector_store is not None
    
    def has_data(self) -> bool:
        """Check if vector store contains data"""
        try:
            test_results = self.vector_store.similarity_search("test", k=1)
            return len(test_results) > 0
        except Exception:
            return False
        
    def get_all_candidate_chunks(self, filename: str, k: int = 20) -> List[Document]:
        """Get all chunks for a specific candidate"""
        try:
            return self.vector_store.similarity_search(
                query="resume content",
                k=k,
                filter={"source": filename}
            )
        except Exception as e:
            st.error(f"Failed to get candidate chunks: {str(e)}")
            return []   