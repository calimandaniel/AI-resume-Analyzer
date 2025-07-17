from typing import List, Dict

class PromptBuilder:
    """Utility for building prompts"""
    
    @staticmethod
    def build_search_query(query: str, job_description: str = "") -> str:
        """Build search query combining user query and job description"""
        if job_description:
            return f"Job Requirements: {job_description}\n\nQuestion: {query}"
        return query
    
    @staticmethod
    def build_context(chunks: List[Dict], job_description: str = "") -> str:
        """Build context with complete candidate profiles"""
        context_parts = []
        
        if job_description:
            context_parts.append(f"JOB DESCRIPTION:\n{job_description}\n")
        
        context_parts.append("COMPLETE CANDIDATE PROFILES:")
        
        for i, chunk in enumerate(chunks, 1):
            filename = chunk.get('filename', f'Candidate {i}')
            chunk_count = chunk.get('chunk_count', 1)
            
            context_parts.extend([
                f"\n{'='*50}",
                f"CANDIDATE {i}: {filename}",
                f"Document Sections: {chunk_count}",
                f"Relevance Score: {1 - chunk['similarity_score']:.3f}",
                f"{'='*50}",
                "COMPLETE RESUME CONTENT:",
                chunk['text']
            ])
            
            if chunk.get('relevant_sections'):
                context_parts.append("\nMOST RELEVANT SECTIONS FOR THIS QUERY:")
                for j, section in enumerate(chunk['relevant_sections'], 1):
                    context_parts.extend([f"\nRelevant Section {j}:", section])
            
            context_parts.append(f"\n{'='*50}\n")
        
        return "\n".join(context_parts)
    
    @staticmethod
    def build_history_context(history: List[Dict]) -> str:
        """Build context from conversation history"""
        if not history:
            return ""
        
        history_parts = ["CONVERSATION HISTORY:"]
        for msg in history[-3:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_parts.append(f"{role}: {msg['content']}")
        
        return "\n".join(history_parts)
    
    @staticmethod
    def create_comprehensive_prompt(query: str, context: str, history: str, job_description: str = "") -> str:
        """Create comprehensive analysis prompt"""
        return f"""You are an expert HR analyst with access to COMPLETE candidate profiles.

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