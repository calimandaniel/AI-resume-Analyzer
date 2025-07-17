import google.generativeai as genai
from typing import Optional
import streamlit as st

from config import settings

class LLMService:
    """Service for Large Language Model operations"""
    
    def __init__(self):
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the LLM"""
        try:
            if not settings.google_api_key:
                st.error("GOOGLE_API_KEY not found")
                return
            
            genai.configure(api_key=settings.google_api_key)
            self.model = genai.GenerativeModel(settings.llm_model)
            
        except Exception as e:
            st.error(f"Failed to initialize LLM: {str(e)}")
    
    def generate_content(self, prompt: str) -> str:
        """Generate content using the LLM"""
        if not self.model:
            return "LLM not initialized"
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Failed to generate content: {str(e)}")
            return "Error generating response"
    
    def is_ready(self) -> bool:
        """Check if LLM is ready"""
        return self.model is not None
    
    def analyze_candidate_profile(self, filename: str, complete_text: str, specific_query: str = "") -> str:
        """Analyze a complete candidate profile"""
        try:
            analysis_prompt = f"""
            Provide comprehensive information about this candidate based on their complete resume:
            
            CANDIDATE: {filename}
            SPECIFIC QUESTION: {specific_query if specific_query else "General overview"}
            
            COMPLETE RESUME:
            {complete_text}
            
            Provide detailed information including:
            - Professional background and experience
            - Technical skills and competencies  
            - Education and certifications
            - Key achievements and projects
            - Career progression
            - Any other relevant details
            
            If a specific question was asked, focus on that while providing context from their complete profile.
            """
            
            response = self.model.generate_content(analysis_prompt)
            return response.text
        except Exception as e:
            st.error(f"Failed to analyze profile: {str(e)}")
            return f"Error analyzing {filename}: {str(e)}"

    def generate_candidate_summary(self, filename: str, full_text: str) -> str:
        """Generate a summary for a candidate"""
        try:
            summary_prompt = f"""Please provide a concise professional summary of this candidate based on their resume:

                                {full_text}

                                Focus on:
                                - Key skills and technologies
                                - Experience level and roles
                                - Education background
                                - Notable achievements
                                - Overall professional profile

                                Provide a 2-3 paragraph summary:"""

            summary_response = self.model.generate_content(summary_prompt)
            return summary_response.text
        except Exception as e:
            st.error(f"Failed to generate summary: {str(e)}")
            return f"Error generating summary for {filename}"