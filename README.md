# Resume Analyzer Chatbot

This project is a Resume Analyzer Chatbot built using Streamlit. It allows users to upload resumes, processes the content, and interacts with a chatbot that provides insights and answers based on the uploaded resumes.

## Features

- **Resume Uploading**: Users can upload their resumes in various formats (PDF, Word, TXT).
- **Text Extraction**: The application extracts text from uploaded resumes for processing.
- **Vector Database Integration**: The processed resumes are stored in a vector database for efficient retrieval.
- **Retrieval-Augmented Generation Chatbot**: The chatbot utilizes the processed data to provide relevant responses and insights.

## Project Structure

```
resume-analyzer-chatbot
├── src
│   ├── app.py                     # Main entry point for the Streamlit application
│   ├── components                  # UI components for the application
│   │   ├── chat_interface.py       # Chatbot interface implementation
│   │   ├── resume_uploader.py      # Resume uploading functionality
│   ├── core                        # Core functionality for processing resumes
│       └── rag_chatbot.py          # RAG chatbot implementation
├── requirements.txt                # Project dependencies
├── config.yaml                     # Configuration settings
├── .streamlit                      # Streamlit configuration
│   └── config.toml                # Streamlit-specific settings
├── .gitignore                      # Files to ignore in version control
└── README.md                       # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/calimandaniel/AI-resume-Analyzer.git
   cd AI-reusme-Analyzer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```
   streamlit run src/app.py
   ```

## Usage

- Upload your resume using the provided interface.
- Interact with the chatbot to get insights and answers based on your resume.