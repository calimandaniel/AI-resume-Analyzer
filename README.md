# AI Resume Analyzer

An intelligent Resume Analyzer system built with Streamlit that uses AI to analyze resumes, match candidates to job descriptions, and provide comprehensive insights through a conversational interface.

## Features

- **Multi-format Resume Processing**: Upload and process resumes in PDF, DOCX, and TXT formats
- **Intelligent Document Chunking**: Advanced text segmentation with configurable chunk sizes and overlap
- **Vector-based Search**: PostgreSQL vector database for efficient semantic search and retrieval
- **AI-Powered Analysis**: Google Gemini integration for comprehensive candidate analysis
- **Job Matching**: Match candidates against job descriptions with relevance scoring
- **Conversational Interface**: Natural language chat interface for querying candidate data
- **Complete Candidate Profiles**: Access full resume content with contextual analysis
- **Modular Architecture**: Clean, maintainable codebase with separation of concerns

## Architecture

The application follows a modular service-oriented architecture:

```
src/
├── app.py                          # Streamlit application entry point
├── config/
│   ├── settings.py                 # Centralized configuration management
│   └── __init__.py
├── services/                       # Business logic layer
│   ├── document_service.py         # Document processing and chunking
│   ├── vector_service.py           # Vector database operations
│   ├── llm_service.py             # Large Language Model integration
│   └── __init__.py
├── core/
│   ├── rag_chatbot.py             # RAG (Retrieval-Augmented Generation) engine
│   └── __init__.py
├── components/                     # UI components
│   ├── resume_uploader.py         # Resume upload and processing interface
│   ├── chat_interface.py          # Conversational chat interface
│   └── __init__.py
├── utils/                         # Utility functions
│   ├── candidate_utils.py         # Candidate data processing utilities
│   ├── prompt_builder.py          # AI prompt construction utilities
│   └── __init__.py
└── .env                           # Environment variables (not tracked)
```

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.9+
- **AI/ML**: Google Gemini (LLM), Google Embeddings
- **Vector Database**: PostgreSQL with pgvector
- **Document Processing**: LangChain, PyPDF2, python-docx
- **Configuration**: YAML, python-dotenv

## Prerequisites

- Python 3.9 or higher
- PostgreSQL database with pgvector extension
- Google AI API key

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/calimandaniel/AI-resume-Analyzer.git
   cd AI-resume-Analyzer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the `src/` directory:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   DATABASE_URL=postgresql://username:password@localhost:5432/your_database
   ```

4. **Configure application settings**:
   Modify `config.yaml` to customize processing parameters:
   ```yaml
   model:
     embedding_model: "models/embedding-001"
     llm_model: "gemini-1.5-flash"
   
   processing:
     chunk_size: 400
     chunk_overlap: 50
     max_candidates: 5
   ```

5. **Run the application**:
   ```bash
   streamlit run src/app.py
   ```

## Usage

### 1. Upload Resumes
- Navigate to "Upload Resumes" in the sidebar
- Configure chunking parameters (chunk size, overlap)
- Upload multiple resume files (PDF, DOCX, TXT)
- Process documents to extract and store in vector database

### 2. Interactive Analysis
- Switch to "Chatbot" mode
- Optionally add a job description for better matching
- Ask natural language questions about candidates:
  - "Who has Python experience?"
  - "Rank candidates for a senior developer role"
  - "Tell me about John Doe's background"
  - "Compare the top 3 candidates"

### 3. Advanced Queries
The system supports complex queries like:
- Candidate ranking and comparison
- Skill-based filtering
- Experience level analysis
- Education background review
- Project and achievement analysis

## Key Features

### Modular Services
- **DocumentService**: Handles file upload, text extraction, and chunking
- **VectorService**: Manages vector database operations and similarity search
- **LLMService**: Integrates with Google Gemini for content generation
- **RAGChatbot**: Orchestrates retrieval and generation for conversational AI

### Intelligent Processing
- Configurable text chunking with overlap
- Metadata preservation for source tracking
- Complete candidate profile reconstruction
- Context-aware prompt generation

### Scalable Architecture
- Service-oriented design for easy testing and maintenance
- Centralized configuration management
- Clean separation between UI, business logic, and data layers
- Extensible for additional AI models and data sources

## Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Your Google AI API key
- `DATABASE_URL`: PostgreSQL connection string with pgvector

### Application Settings
Customize behavior through `config.yaml`:
- Model selection (embedding and LLM models)
- Processing parameters (chunk size, overlap)
- Search and retrieval settings
- UI configuration options

## API Integration

The system integrates with:
- **Google Generative AI**: For embeddings and text generation
- **PostgreSQL + pgvector**: For vector storage and similarity search
- **LangChain**: For document processing and text splitting
