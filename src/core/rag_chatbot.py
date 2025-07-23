import os
import streamlit as st
from typing import List, Dict, Optional

from services import VectorService, LLMService
from utils import CandidateUtils, PromptBuilder

class RAGChatbot:
    """RAG-based chatbot for resume analysis"""
    
    def __init__(self):
        self.vector_service = VectorService()
        self.llm_service = LLMService()
        self.conversation_history = []

    def is_ready(self) -> bool:
        """Check if all services are ready"""
        return self.vector_service.is_ready() and self.llm_service.is_ready()

    def retrieve_relevant_chunks(self, query: str, job_description: str = "", 
                            top_k: int = 5) -> List[Dict]:
        """Retrieve relevant resume chunks with complete candidate context"""
        try:
            # Build search query
            combined_query = PromptBuilder.build_search_query(query, job_description)
            
            # Get initial relevant chunks
            similar_docs = self.vector_service.similarity_search_with_score(
                query=combined_query,
                k=top_k * 3
            )
            
            # Group by candidate
            candidates_chunks = CandidateUtils.group_chunks_by_candidate(similar_docs)
            
            # Enhance with complete candidate profiles
            enhanced_chunks = []
            for filename, chunks in candidates_chunks.items():
                enhanced_chunk = self._build_complete_candidate_profile(filename, chunks)
                enhanced_chunks.append(enhanced_chunk)
            
            # Sort by relevance and limit
            enhanced_chunks.sort(key=lambda x: x['similarity_score'])
            return enhanced_chunks[:top_k]
            
        except Exception as e:
            st.error(f"Error retrieving chunks: {str(e)}")
            return []
        
    def _build_complete_candidate_profile(self, filename: str, chunks: List[Dict]) -> Dict:
        """Build complete candidate profile with all resume sections"""
        # Get all chunks for this candidate using the service
        all_candidate_chunks = self.vector_service.get_all_candidate_chunks(filename, k=20)
        
        best_score = min(chunk['similarity_score'] for chunk in chunks)
        candidate_complete_text = CandidateUtils.build_complete_candidate_text(all_candidate_chunks)
        
        return {
            'text': candidate_complete_text,
            'chunks': [chunk.page_content for chunk in all_candidate_chunks],
            'metadata': chunks[0]['metadata'],
            'similarity_score': best_score,
            'filename': filename,
            'chunk_count': len(all_candidate_chunks),
            'relevant_sections': [chunk['text'] for chunk in chunks[:3]]
        }
        
    def _build_context(self, chunks: List[Dict], job_description: str = "") -> str:
        """Build context with complete candidate profiles"""
        return PromptBuilder.build_context(chunks, job_description)

    def _build_history_context(self, history: List[Dict]) -> str:
        """Build context from conversation history"""
        return PromptBuilder.build_history_context(history)

    def _create_comprehensive_prompt(self, query: str, context: str, history: str, job_description: str = "") -> str:
        """Create comprehensive analysis prompt"""
        return PromptBuilder.create_comprehensive_prompt(query, context, history, job_description)

    def generate_response(self, user_query: str, job_description: str = "", 
                        conversation_history: List[Dict] = None) -> str:
        """Generate comprehensive response with complete candidate profiles"""
        try:
            if not self.is_ready():
                return "RAG system not properly initialized."
            
            # Get complete profiles of most relevant candidates
            complete_profiles = self.retrieve_relevant_chunks(
                user_query, job_description, top_k=3
            )
            
            if not complete_profiles:
                return "No relevant candidate information found."
            
            # Build context and history using utils
            context = self._build_context(complete_profiles, job_description)
            history_context = self._build_history_context(conversation_history or [])
            
            # Create prompt
            prompt = self._create_comprehensive_prompt(user_query, context, history_context, job_description)
            
            # Generate response using LLM service
            response = self.llm_service.generate_content(prompt)
            
            # Add metadata
            candidate_names = [profile['filename'] for profile in complete_profiles]
            return f"{response}\n\n---\nAnalysis based on: {', '.join(candidate_names)}"
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return "Error processing request."