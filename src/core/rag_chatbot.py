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

    def get_complete_candidate_profile(self, filename: str) -> Dict:
        """Get complete candidate profile with all resume sections"""
        try:
            if not self.is_ready():
                return {"error": "RAG system not properly initialized"}
            
            # Get ALL chunks for this candidate
            all_chunks = self.vector_service.get_all_candidate_chunks(filename, k=50)
            
            if not all_chunks:
                return {"error": f"No data found for {filename}"}
            
            # Combine all chunks in proper order
            chunks_ordered = sorted(all_chunks, key=lambda x: x.metadata.get('chunk_id', 0))
            complete_resume = "\n".join([chunk.page_content for chunk in chunks_ordered])
            
            return {
                "filename": filename,
                "complete_text": complete_resume,
                "total_chunks": len(all_chunks),
                "sections": [{"id": chunk.metadata.get('chunk_id', i), 
                            "content": chunk.page_content} for i, chunk in enumerate(chunks_ordered)]
            }
            
        except Exception as e:
            return {"error": f"Error getting complete profile: {str(e)}"}
        
    def get_candidate_detailed_info(self, filename: str, specific_query: str = "") -> str:
        """Get detailed information about a specific candidate with their complete profile"""
        try:
            # Get complete profile
            complete_profile = self.get_complete_candidate_profile(filename)
            
            if 'error' in complete_profile:
                return f"Could not retrieve information for {filename}: {complete_profile['error']}"
            
            # Use LLM service to analyze
            return self.llm_service.analyze_candidate_profile(
                filename, 
                complete_profile['complete_text'], 
                specific_query
            )
            
        except Exception as e:
            return f"Error analyzing {filename}: {str(e)}"
        
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
        
    def analyze_job_match(self, job_description: str, top_k: int = 15) -> Dict:
        """Analyze all candidates against a job description"""
        try:
            if not self.is_ready():
                return {"error": "RAG system not properly initialized"}
            
            # Search for candidates matching the job description
            matching_chunks = self.retrieve_relevant_chunks(
                f"Find candidates matching: {job_description}",
                job_description,
                top_k=top_k
            )
            
            if not matching_chunks:
                return {"error": "No matching candidates found"}
            
            # Use CandidateUtils to process candidates
            candidates = {}
            for chunk in matching_chunks:
                filename = chunk['filename']
                if filename not in candidates:
                    candidates[filename] = {
                        'chunks': [],
                        'total_score': 0,
                        'avg_score': 0,
                        'best_match': None
                    }
                
                relevance = 1 - chunk['similarity_score']
                candidates[filename]['chunks'].append(chunk)
                candidates[filename]['total_score'] += relevance
                
                if not candidates[filename]['best_match'] or relevance > (1 - candidates[filename]['best_match']['similarity_score']):
                    candidates[filename]['best_match'] = chunk
            
            # Calculate scores and sort
            candidates = CandidateUtils.calculate_candidate_scores(candidates)
            sorted_candidates = CandidateUtils.sort_candidates_by_score(candidates)
            
            return {
                "candidates": sorted_candidates,
                "job_description": job_description,
                "total_candidates": len(candidates)
            }
            
        except Exception as e:
            return {"error": f"Error analyzing job match: {str(e)}"}
        
        
    def get_candidate_summary(self, filename: str) -> Dict:
        """Get a summary of a specific candidate"""
        try:
            if not self.is_ready():
                return {"error": "RAG system not properly initialized"}
            
            # Get candidate chunks using vector service
            candidate_chunks = self.vector_service.similarity_search(
                query=f"Resume content for {filename}",
                k=10,
                filters={"source": filename}
            )
            
            if not candidate_chunks:
                return {"error": f"No data found for {filename}"}
            
            # Combine all text from chunks
            full_text = "\n".join([chunk.page_content for chunk in candidate_chunks])
            
            # Generate summary using LLM service
            summary = self.llm_service.generate_candidate_summary(filename, full_text)
            
            return {
                "filename": filename,
                "summary": summary,
                "total_chunks": len(candidate_chunks),
                "total_tokens": sum([len(chunk.page_content.split()) for chunk in candidate_chunks])
            }
            
        except Exception as e:
            return {"error": f"Error generating candidate summary: {str(e)}"}