import streamlit as st
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from components.resume_uploader import ResumeUploader
from components.chat_interface import ChatInterface

def main():
    st.set_page_config(
        page_title="Resume Analyzer Chatbot",
        layout="wide"
    )
    
    st.title("Resume Analyzer Chatbot")
    st.sidebar.title("Navigation")
    
    app_mode = st.sidebar.selectbox(
        "Choose the app mode", 
        ["Upload Resumes", "Chatbot"]
    )

    if app_mode == "Upload Resumes":
        uploader = ResumeUploader()
        uploader.upload_resume()
    
    elif app_mode == "Chatbot":
        chatbot = ChatInterface()
        chatbot.run_chat()

if __name__ == "__main__":
    main()