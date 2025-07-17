import os
import streamlit as st
from typing import List, Dict, Optional
import numpy as np
import asyncio
from asyncio import new_event_loop, set_event_loop, run

try:
    import google.generativeai as genai
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_postgres import PGVector
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from dotenv import load_dotenv

# Load environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.dirname(current_dir)
env_path = os.path.join(src_root, '.env')

if os.path.exists(env_path):
    load_dotenv(env_path)

class RAGChatbot:
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.llm_model = None
        self.conversation_history = []
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize embeddings, vector store, and LLM"""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            db_url = os.getenv("DATABASE_URL")
            
            if not api_key:
                st.error("GOOGLE_API_KEY not found in environment variables")
                return
            
            if not db_url:
                st.error("DATABASE_URL not found in environment variables")
                return
            
            # Initialize Gemini API (sync)
            genai.configure(api_key=api_key)
            
            # Create a coroutine function for async operations
            async def initialize_async_components():
                # Initialize embeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=api_key
                )
                
                # Initialize vector store if database URL exists
                vector_store = None
                if db_url:
                    vector_store = PGVector(
                        connection=db_url,
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
            
            # Initialize LLM (sync)
            self.llm_model = genai.GenerativeModel('gemini-1.5-flash')
            
        except Exception as e:
            st.error(f"Error initializing RAG components: {str(e)}")
            self.embeddings = None
            self.vector_store = None
            self.llm_model = None
    
    def is_ready(self) -> bool:
        """Check if all components are ready"""
        return all([self.embeddings, self.vector_store, self.llm_model])
    
    def retrieve_relevant_chunks(self, query: str, job_description: str = "", 
                               top_k: int = 5) -> List[Dict]:
        """Retrieve relevant resume chunks and group by candidate for complete context"""
        try:
            if not self.vector_store:
                return []
            
            # Get initial relevant chunks
            if job_description:
                combined_query = f"Job Requirements: {job_description}\n\nQuestion: {query}"
            else:
                combined_query = query
            
            # Get more chunks initially to ensure we capture complete candidates
            similar_docs = self.vector_store.similarity_search_with_score(
                query=combined_query,
                k=top_k * 3  # Get 3x more to ensure complete candidate coverage
            )
            
            # Group chunks by candidate (filename)
            candidates_chunks = {}
            for doc, score in similar_docs:
                filename = doc.metadata.get('source', 'Unknown')
                if filename not in candidates_chunks:
                    candidates_chunks[filename] = []
                
                chunk_dict = {
                    'text': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': score,
                    'filename': filename,
                    'chunk_id': doc.metadata.get('chunk_id', 0)
                }
                candidates_chunks[filename].append(chunk_dict)
            
            # For each relevant candidate, get ALL their chunks for complete context
            enhanced_chunks = []
            for filename, chunks in candidates_chunks.items():
                # Get ALL chunks for this candidate to provide complete context
                all_candidate_chunks = self.vector_store.similarity_search(
                    query="resume content",  # Generic query to get all chunks
                    k=20,  # Get up to 20 chunks per candidate
                    filter={"source": filename}
                )
                
                # Calculate best relevance score for this candidate
                best_score = min(chunk['similarity_score'] for chunk in chunks)
                
                # Add all chunks for this candidate with complete context
                candidate_complete_text = "\n".join([chunk.page_content for chunk in all_candidate_chunks])
                
                enhanced_chunk = {
                    'text': candidate_complete_text,  # COMPLETE CV content
                    'chunks': [chunk.page_content for chunk in all_candidate_chunks],  # Individual chunks
                    'metadata': chunks[0]['metadata'],  # Use first chunk's metadata
                    'similarity_score': best_score,  # Best relevance score
                    'filename': filename,
                    'chunk_count': len(all_candidate_chunks),
                    'relevant_sections': [chunk['text'] for chunk in chunks[:3]]  # Most relevant parts
                }
                enhanced_chunks.append(enhanced_chunk)
            
            # Sort by relevance and limit to requested number of candidates
            enhanced_chunks.sort(key=lambda x: x['similarity_score'])
            return enhanced_chunks[:top_k]
            
        except Exception as e:
            st.error(f"Error retrieving chunks: {str(e)}")
            return []
    
    
    def _build_context(self, chunks: List[Dict], job_description: str = "") -> str:
        """Build context with COMPLETE candidate profiles"""
        context_parts = []
        
        if job_description:
            context_parts.append(f"JOB DESCRIPTION:\n{job_description}\n")
        
        context_parts.append("COMPLETE CANDIDATE PROFILES:")
        
        for i, chunk in enumerate(chunks, 1):
            filename = chunk.get('filename', f'Candidate {i}')
            chunk_count = chunk.get('chunk_count', 1)
            
            context_parts.append(f"\n{'='*50}")
            context_parts.append(f"CANDIDATE {i}: {filename}")
            context_parts.append(f"Document Sections: {chunk_count}")
            context_parts.append(f"Relevance Score: {1 - chunk['similarity_score']:.3f}")
            context_parts.append(f"{'='*50}")
            
            # Add the COMPLETE CV content
            context_parts.append("COMPLETE RESUME CONTENT:")
            context_parts.append(chunk['text'])
            
            # Highlight most relevant sections
            if chunk.get('relevant_sections'):
                context_parts.append("\nMOST RELEVANT SECTIONS FOR THIS QUERY:")
                for j, section in enumerate(chunk['relevant_sections'], 1):
                    context_parts.append(f"\nRelevant Section {j}:")
                    context_parts.append(section)
            
            context_parts.append(f"\n{'='*50}\n")
        
        return "\n".join(context_parts)
    
    def _create_comprehensive_prompt(self, query: str, context: str, history: str, job_description: str = "") -> str:
        """Create prompt for comprehensive candidate analysis"""
        
        prompt = f"""You are an expert HR analyst with access to COMPLETE candidate profiles. You have the full curriculum vitae for each relevant candidate, not just fragments.

                {history}

                {context}

                USER QUESTION: {query}

                COMPREHENSIVE ANALYSIS INSTRUCTIONS:
                1. You have COMPLETE CV information for each candidate - use ALL available details
                2. When discussing a candidate, reference their full background, experience, education, and skills
                3. For follow-up questions about a specific candidate, provide detailed answers using their complete profile
                4. Compare candidates thoroughly using their complete information
                5. Cite specific sections: [Candidate X: filename - specific detail]
                6. If asked about someone's experience, education, or skills - provide comprehensive details
                7. For ranking questions, use complete profiles to make thorough comparisons
                8. Always mention which complete profiles you're analyzing
                9. Be specific about years of experience, technologies, projects, education, etc.
                10. If someone asks "tell me more about candidate X", provide a comprehensive overview

                Provide a detailed, comprehensive response using the complete candidate information:"""

        return prompt
    
    def get_complete_candidate_profile(self, filename: str) -> Dict:
        """Get complete candidate profile with all resume sections"""
        try:
            if not self.is_ready():
                return {"error": "RAG system not properly initialized"}
            
            # Get ALL chunks for this candidate
            all_chunks = self.vector_store.similarity_search(
                query="resume content",
                k=50,  # Get all chunks
                filter={"source": filename}
            )
            
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
            
            # Create detailed analysis prompt
            analysis_prompt = f"""
            Provide comprehensive information about this candidate based on their complete resume:
            
            CANDIDATE: {filename}
            SPECIFIC QUESTION: {specific_query if specific_query else "General overview"}
            
            COMPLETE RESUME:
            {complete_profile['complete_text']}
            
            Provide detailed information including:
            - Professional background and experience
            - Technical skills and competencies  
            - Education and certifications
            - Key achievements and projects
            - Career progression
            - Any other relevant details
            
            If a specific question was asked, focus on that while providing context from their complete profile.
            """
            
            response = self.llm_model.generate_content(analysis_prompt)
            return response.text
            
        except Exception as e:
            return f"Error analyzing {filename}: {str(e)}"
        
    def generate_response(self, user_query: str, job_description: str = "", 
                         conversation_history: List[Dict] = None) -> str:
        """Generate comprehensive response with complete candidate profiles"""
        try:
            if not self.is_ready():
                return "RAG system not properly initialized. Please check your configuration."
            
            # Get complete profiles of most relevant candidates
            complete_profiles = self.retrieve_relevant_chunks(
                user_query, job_description, top_k=3  # Get top 3 candidates with complete profiles
            )
            
            if not complete_profiles:
                return "I couldn't find any relevant candidate information. Please make sure resumes have been uploaded and processed."
            
            # Build comprehensive context with complete CVs
            context = self._build_context(complete_profiles, job_description)
            
            # Build conversation history
            history_context = self._build_history_context(conversation_history or [])
            
            # Create enhanced prompt for comprehensive analysis
            prompt = self._create_comprehensive_prompt(user_query, context, history_context, job_description)
            
            # Generate response using complete candidate information
            response = self.llm_model.generate_content(prompt)
            
            # Add metadata about which candidates were analyzed
            candidate_names = [profile['filename'] for profile in complete_profiles]
            response_with_context = f"{response.text}\n\n---\n*Analysis based on complete profiles of: {', '.join(candidate_names)}*"
            
            return response_with_context
                
        except Exception as e:
            st.error(f"Error generating comprehensive response: {str(e)}")
            return "I encountered an error while processing your request. Please try again."
    
    def _build_history_context(self, history: List[Dict]) -> str:
        """Build context from conversation history"""
        if not history:
            return ""
        
        history_parts = ["CONVERSATION HISTORY:"]
        for msg in history[-3:]:  # Last 3 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            history_parts.append(f"{role}: {msg['content']}")
        
        return "\n".join(history_parts)
    
    def _create_prompt(self, query: str, context: str, history: str, job_description: str = "") -> str:
        """Create the prompt for the LLM"""
        
        prompt = f"""You are an expert resume analyzer and job matching assistant. Your task is to analyze resumes against job requirements and provide insightful, conversational responses.

{history}

{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Analyze the relevant resume excerpts in the context of the job description (if provided)
2. Provide a comprehensive, conversational response that addresses the user's question
3. When discussing candidates, cite specific information from the resume excerpts using the format [Resume X: filename]
4. If comparing candidates, highlight strengths and potential fits for the role
5. Be specific about skills, experience, and qualifications mentioned in the resumes
6. If the query is about candidate ranking, provide clear reasoning based on the job requirements
7. Maintain a professional but conversational tone
8. If information is insufficient, suggest what additional details would be helpful
9. Always reference which resume sections you're drawing information from
10. Focus on finding the candidates with the most potential for the given role

Please provide a detailed response based on the resume information provided:"""

        return prompt
    
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
            
            # Group by filename for candidate analysis
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
                
                # Track best matching section
                if not candidates[filename]['best_match'] or relevance > (1 - candidates[filename]['best_match']['similarity_score']):
                    candidates[filename]['best_match'] = chunk
            
            # Calculate average scores
            for filename in candidates:
                chunks_count = len(candidates[filename]['chunks'])
                candidates[filename]['avg_score'] = candidates[filename]['total_score'] / chunks_count
            
            # Sort candidates by average score
            sorted_candidates = sorted(
                candidates.items(), 
                key=lambda x: x[1]['avg_score'], 
                reverse=True
            )
            
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
            
            # Search for all content related to this candidate
            candidate_chunks = self.vector_store.similarity_search(
                query=f"Resume content for {filename}",
                k=10,
                filter={"source": filename}
            )
            
            if not candidate_chunks:
                return {"error": f"No data found for {filename}"}
            
            # Combine all text from chunks
            full_text = "\n".join([chunk.page_content for chunk in candidate_chunks])
            
            # Generate summary using LLM
            summary_prompt = f"""Please provide a concise professional summary of this candidate based on their resume:

{full_text}

Focus on:
- Key skills and technologies
- Experience level and roles
- Education background
- Notable achievements
- Overall professional profile

Provide a 2-3 paragraph summary:"""

            summary_response = self.llm_model.generate_content(summary_prompt)
            summary = summary_response.text
            
            return {
                "filename": filename,
                "summary": summary,
                "total_chunks": len(candidate_chunks),
                "total_tokens": sum([len(chunk.page_content.split()) for chunk in candidate_chunks])
            }
            
        except Exception as e:
            return {"error": f"Error generating candidate summary: {str(e)}"}