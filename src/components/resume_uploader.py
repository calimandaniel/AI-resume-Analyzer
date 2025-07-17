import streamlit as st
import pandas as pd
from typing import List, Dict

from services import DocumentService, VectorService
from config import settings

class ResumeUploader:
    def __init__(self):
        self.document_service = DocumentService()
        self.vector_service = VectorService()
        self.uploaded_files = []
        
    def upload_resume(self):
        st.header("Upload Resumes")
        st.write("Upload multiple resume files (PDF, DOCX, or TXT format)")
        
        # Chunking parameters
        st.subheader("Chunking Configuration")
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider("Chunk Size (tokens)", 100, 500, settings.chunk_size, 25)
        with col2:
            overlap_size = st.slider("Overlap Size (tokens)", 0, 100, settings.chunk_overlap, 10)
        
        uploaded_files = st.file_uploader(
            "Choose resume files", 
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"Successfully uploaded {len(uploaded_files)} file(s)!")
            
            # Display uploaded files
            for uploaded_file in uploaded_files:
                st.write(f"{uploaded_file.name}")
            
            # Process with services
            if st.button("Process Documents"):
                self._process_documents(uploaded_files, chunk_size, overlap_size)

    def _process_documents(self, uploaded_files, chunk_size: int, overlap_size: int):
        """Process documents using the modular services"""
        
        if not self._check_services_ready():
            return
        
        progress_bar = st.progress(0)
        
        try:
            # Step 1: Load documents using DocumentService
            st.info("Loading documents...")
            documents = self.document_service.load_documents(uploaded_files)
            progress_bar.progress(0.3)
            
            if not documents:
                st.error("No documents were loaded successfully")
                return
            
            # Step 2: Chunk documents using DocumentService
            st.info("Processing and chunking documents...")
            chunks = self.document_service.chunk_documents(
                documents, 
                chunk_size=chunk_size, 
                overlap=overlap_size
            )
            progress_bar.progress(0.6)
            
            if not chunks:
                st.error("No chunks were created")
                return
            
            # Step 3: Save to vector store using VectorService
            st.info("Saving to vector store...")
            success = self.vector_service.add_documents(chunks)
            progress_bar.progress(0.9)
            
            if success:
                progress_bar.progress(1.0)
                self._display_results(documents, chunks)
            else:
                st.error("Failed to save to vector store")
        
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

    def _check_services_ready(self) -> bool:
        """Check if all required services are ready"""
        if not self.vector_service.is_ready():
            st.error("Vector service not ready. Please check your environment variables.")
            
            # Show what's missing
            with st.expander("Configuration Status"):
                if not settings.google_api_key:
                    st.error("GOOGLE_API_KEY not found")
                else:
                    st.success("GOOGLE_API_KEY configured")
                
                if not settings.database_url:
                    st.error("DATABASE_URL not found")
                else:
                    st.success("DATABASE_URL configured")
                    
            return False
        
        return True

    def _display_results(self, documents: List, chunks: List):
        """Display processing results"""
        st.subheader("Processing Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Documents Loaded", len(documents))
        with col2:
            st.metric("Total Chunks", len(chunks))
        with col3:
            total_chars = sum(len(chunk.page_content) for chunk in chunks)
            st.metric("Total Characters", f"{total_chars:,}")
        with col4:
            avg_chunk_size = total_chars / len(chunks) if chunks else 0
            st.metric("Avg Chunk Size", f"{avg_chunk_size:.0f}")
        
        # Document summary
        doc_summary = self._build_document_summary(chunks)
        
        st.subheader("Document Summary")
        summary_data = []
        for source, data in doc_summary.items():
            summary_data.append({
                'Filename': source,
                'Chunks': data['chunks'],
                'Total Characters': f"{data['total_chars']:,}",
                'Avg Chunk Size': f"{data['total_chars'] / data['chunks']:.0f}",
                'Status': 'Processed'
            })
        
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)
        
        # Sample chunks viewer
        self._display_sample_chunks(chunks, doc_summary)
        
        st.success(f"Successfully processed {len(documents)} documents into {len(chunks)} chunks!")

    def _build_document_summary(self, chunks: List) -> Dict[str, Dict]:
        """Build summary statistics by document"""
        doc_summary = {}
        for chunk in chunks:
            source = chunk.metadata.get('source', 'Unknown')
            if source not in doc_summary:
                doc_summary[source] = {'chunks': 0, 'total_chars': 0}
            doc_summary[source]['chunks'] += 1
            doc_summary[source]['total_chars'] += len(chunk.page_content)
        return doc_summary

    def _display_sample_chunks(self, chunks: List, doc_summary: Dict):
        """Display sample chunks for inspection"""
        with st.expander("View Sample Chunks"):
            if chunks:
                selected_source = st.selectbox(
                    "Select document:",
                    list(doc_summary.keys())
                )
                
                source_chunks = [c for c in chunks if c.metadata.get('source') == selected_source]
                
                for i, chunk in enumerate(source_chunks[:3]):  # Show first 3 chunks
                    st.subheader(f"Chunk {i+1}")
                    st.text_area(
                        f"Content (Chunk {chunk.metadata.get('chunk_id', i)}):",
                        chunk.page_content,
                        height=150,
                        key=f"chunk_{selected_source}_{i}"
                    )
                    st.json(chunk.metadata)