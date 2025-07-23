from typing import List, Dict

class CandidateUtils:
    """Utility functions for candidate processing"""
    
    @staticmethod
    def group_chunks_by_candidate(similar_docs: List[tuple]) -> Dict[str, List[Dict]]:
        """Group chunks by candidate filename"""
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
        
        return candidates_chunks
    
    @staticmethod
    def build_complete_candidate_text(chunks: List) -> str:
        """Build complete text from candidate chunks"""
        chunks_ordered = sorted(chunks, key=lambda x: x.metadata.get('chunk_id', 0))
        return "\n".join([chunk.page_content for chunk in chunks_ordered])
    