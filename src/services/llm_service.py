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