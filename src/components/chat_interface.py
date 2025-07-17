import streamlit as st
import os
from typing import List, Dict
from core.rag_chatbot import RAGChatbot

class ChatInterface:
    def __init__(self):
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'job_description' not in st.session_state:
            st.session_state.job_description = ""
        
        self.rag_chatbot = RAGChatbot()
    
    def _check_database_has_data(self) -> bool:
        """Check if the database actually has resume data"""
        try:
            if not self.rag_chatbot.is_ready():
                return False
            
            # Try to search for any documents in the database
            test_results = self.rag_chatbot.vector_store.similarity_search("test", k=1)
            return len(test_results) > 0
            
        except Exception as e:
            st.error(f"Error checking database: {str(e)}")
            return False
    
    def run_chat(self):
        st.header("Resume Analyzer Chatbot")
        st.write("Ask questions about uploaded resumes and job matching using AI-powered analysis")
        
        # Check if RAG system is ready first
        if not self.rag_chatbot.is_ready():
            st.error("RAG system not properly initialized. Please check your environment variables.")
            return
        
        # Check if database has actual data
        if not self._check_database_has_data():
            st.warning("No resume data found in database. Please upload and process resumes first!")
            
            # Show database status
            with st.expander("Database Status"):
                try:
                    # Try to get database info
                    db_url = os.getenv("DATABASE_URL")
                    if db_url:
                        st.info(f"Database: Connected")
                        st.info(f"Collection: resume_langchain")
                        
                        # Try to count documents
                        all_docs = self.rag_chatbot.vector_store.similarity_search("", k=100)
                        st.info(f"Documents found: {len(all_docs)}")
                        
                        if len(all_docs) == 0:
                            st.warning("Database is connected but contains no resume data. Please upload resumes in the 'Upload Resumes' section.")
                    else:
                        st.error("No DATABASE_URL configured")
                        
                except Exception as e:
                    st.error(f"Database check error: {str(e)}")
            
            if st.button("🔄 Refresh and Check Again"):
                st.experimental_rerun()
            return
        
        # Job description input (optional)
        st.subheader("Job Description (Optional)")
        job_description = st.text_area(
            "Enter the job description for better candidate matching:",
            value=st.session_state.get('job_description', ''),
            placeholder="Example: We are looking for a Senior Python Developer with 5+ years of experience in web development, Django, REST APIs, and machine learning. The candidate should have experience with PostgreSQL, Docker, and cloud platforms like AWS.",
            height=120,
            key="job_desc_input",
            help="This helps the AI understand what you're looking for, but you can ask questions without it too."
        )
        
        # Update session state
        if job_description != st.session_state.job_description:
            st.session_state.job_description = job_description
        
        # Quick database info
        with st.expander("Database Info"):
            try:
                # Get some basic stats
                sample_docs = self.rag_chatbot.vector_store.similarity_search("", k=20)
                unique_sources = list(set([doc.metadata.get('source', 'Unknown') for doc in sample_docs]))
                
                st.write(f"**Candidates in database:** {len(unique_sources)}")
                if unique_sources:
                    st.write("**Candidate files:**")
                    for source in unique_sources[:10]:  # Show first 10
                        st.write(f"• {source}")
                    if len(unique_sources) > 10:
                        st.write(f"... and {len(unique_sources) - 10} more")
                        
            except Exception as e:
                st.write(f"Could not retrieve database stats: {str(e)}")
        
        # Chat interface
        st.subheader("Ask Anything About Your Candidates")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input - completely open ended
        if prompt := st.chat_input("Ask me anything about the candidates"):
            self._handle_user_input(prompt, job_description)
    
    def _handle_user_input(self, prompt: str, job_description: str):
        """Handle user input and generate AI response"""
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Show user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing candidates and generating response..."):
                response = self.rag_chatbot.generate_response(
                    user_query=prompt,
                    job_description=job_description,
                    conversation_history=st.session_state.messages[:-1]  # Exclude current message
                )
                
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})