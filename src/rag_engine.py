"""
QuickNotes-AI RAG Engine
FAISS + SentenceTransformers for semantic search over meeting notes.
100% Local - No Data Leaves Your Device
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pickle

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


@dataclass
class Document:
    """A document chunk for embedding."""
    id: str
    content: str
    source: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    """A search result with score."""
    document: Document
    score: float


class RAGEngine:
    """
    Retrieval-Augmented Generation engine using local models.
    Uses SentenceTransformers for embeddings and FAISS for vector search.
    """
    
    # Default embedding model (small and fast)
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    
    # Text chunking settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    def __init__(
        self,
        model_name: str = None,
        index_path: str = "data/vector_store"
    ):
        """
        Initialize RAG engine.
        
        Args:
            model_name: SentenceTransformer model name.
            index_path: Directory to store FAISS index.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.index_path = index_path
        
        self._model = None
        self._index = None
        self._documents: List[Document] = []
        self._dimension = None
        
        os.makedirs(index_path, exist_ok=True)
        
        # Try to load existing index
        self._load_index()
    
    @property
    def is_available(self) -> bool:
        """Check if all dependencies are available."""
        return SENTENCE_TRANSFORMERS_AVAILABLE and FAISS_AVAILABLE
    
    @property
    def document_count(self) -> int:
        """Number of indexed documents."""
        return len(self._documents)
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise RuntimeError(
                    "SentenceTransformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
            
            print(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            print(f"Model loaded! Embedding dimension: {self._dimension}")
    
    def _load_index(self):
        """Load existing FAISS index from disk."""
        index_file = os.path.join(self.index_path, "faiss.index")
        docs_file = os.path.join(self.index_path, "documents.pkl")
        
        if os.path.exists(index_file) and os.path.exists(docs_file):
            try:
                self._index = faiss.read_index(index_file)
                with open(docs_file, "rb") as f:
                    self._documents = pickle.load(f)
                print(f"Loaded {len(self._documents)} documents from index")
            except Exception as e:
                print(f"Failed to load index: {e}")
                self._index = None
                self._documents = []
    
    def _save_index(self):
        """Save FAISS index to disk."""
        if self._index is None:
            return
        
        index_file = os.path.join(self.index_path, "faiss.index")
        docs_file = os.path.join(self.index_path, "documents.pkl")
        
        try:
            faiss.write_index(self._index, index_file)
            with open(docs_file, "wb") as f:
                pickle.dump(self._documents, f)
            print(f"Saved {len(self._documents)} documents to index")
        except Exception as e:
            print(f"Failed to save index: {e}")
    
    def _create_index(self):
        """Create a new FAISS index."""
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not installed. Install with: pip install faiss-cpu")
        
        self._load_model()
        self._index = faiss.IndexFlatIP(self._dimension)  # Inner product for cosine similarity
    
    def add_text(
        self,
        text: str,
        source: str = "unknown",
        metadata: Dict[str, Any] = None
    ) -> int:
        """
        Add text to the index.
        
        Args:
            text: Text content to index.
            source: Source identifier (filename, meeting title, etc.)
            metadata: Optional metadata dict.
            
        Returns:
            Number of chunks added.
        """
        if not text.strip():
            return 0
        
        self._load_model()
        
        if self._index is None:
            self._create_index()
        
        # Chunk the text
        chunks = self._chunk_text(text)
        
        if not chunks:
            return 0
        
        # Create document objects
        new_docs = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                id=f"{source}_{len(self._documents) + i}",
                content=chunk,
                source=source,
                metadata=metadata or {}
            )
            new_docs.append(doc)
        
        # Generate embeddings
        embeddings = self._model.encode([d.content for d in new_docs])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to index
        self._index.add(embeddings.astype('float32'))
        self._documents.extend(new_docs)
        
        # Save to disk
        self._save_index()
        
        return len(new_docs)
    
    def add_texts_batch(
        self,
        texts: List[Tuple[str, str]],
        progress_callback=None
    ) -> int:
        """
        Add multiple texts in batch (saves only once at end).
        
        Args:
            texts: List of (text, source) tuples.
            progress_callback: Optional callback(current, total) for progress.
            
        Returns:
            Total number of chunks added.
        """
        if not texts:
            return 0
        
        self._load_model()
        
        if self._index is None:
            self._create_index()
        
        total_chunks = 0
        all_new_docs = []
        
        for i, (text, source) in enumerate(texts):
            if not text.strip():
                continue
            
            # Chunk the text
            chunks = self._chunk_text(text)
            
            for j, chunk in enumerate(chunks):
                doc = Document(
                    id=f"{source}_{len(self._documents) + len(all_new_docs) + j}",
                    content=chunk,
                    source=source,
                    metadata={}
                )
                all_new_docs.append(doc)
            
            total_chunks += len(chunks)
            
            if progress_callback:
                progress_callback(i + 1, len(texts))
        
        if all_new_docs:
            # Generate all embeddings at once (much faster)
            embeddings = self._model.encode([d.content for d in all_new_docs])
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Add to index
            self._index.add(embeddings.astype('float32'))
            self._documents.extend(all_new_docs)
            
            # Save once at the end
            self._save_index()
        
        return total_chunks
    
    def add_file(self, filepath: str) -> int:
        """
        Add a file to the index.
        Supports .txt and .pdf files.
        
        Args:
            filepath: Path to the file.
            
        Returns:
            Number of chunks added.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == ".txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif ext == ".pdf":
            text = self._extract_pdf_text(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        return self.add_text(
            text,
            source=filename,
            metadata={"filepath": filepath, "type": ext}
        )
    
    def _extract_pdf_text(self, filepath: str) -> str:
        """Extract text from PDF file."""
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF not installed. Install with: pip install PyMuPDF")
        
        text_parts = []
        
        with fitz.open(filepath) as doc:
            for page in doc:
                text_parts.append(page.get_text())
        
        return "\n\n".join(text_parts)
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.CHUNK_SIZE:
            return [text.strip()] if text.strip() else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.CHUNK_SIZE
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for period, question mark, or newline
                for sep in [". ", "? ", "! ", "\n\n", "\n"]:
                    idx = text.rfind(sep, start, end)
                    if idx > start:
                        end = idx + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.CHUNK_OVERLAP
        
        return chunks
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3
    ) -> List[SearchResult]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query.
            top_k: Maximum number of results.
            min_score: Minimum similarity score (0-1).
            
        Returns:
            List of SearchResult objects.
        """
        if self._index is None or len(self._documents) == 0:
            return []
        
        self._load_model()
        
        # Encode query
        query_embedding = self._model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search
        k = min(top_k, len(self._documents))
        scores, indices = self._index.search(query_embedding.astype('float32'), k)
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < min_score:
                continue
            
            results.append(SearchResult(
                document=self._documents[idx],
                score=float(score)
            ))
        
        return results
    
    def get_context(
        self,
        query: str,
        top_k: int = 3,
        max_tokens: int = 1500
    ) -> str:
        """
        Get context string for RAG query.
        
        Args:
            query: Search query.
            top_k: Number of documents to retrieve.
            max_tokens: Approximate max context length (chars / 4).
            
        Returns:
            Context string to pass to LLM.
        """
        results = self.search(query, top_k=top_k)
        
        if not results:
            return ""
        
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # Approximate chars
        
        for result in results:
            if total_chars >= max_chars:
                break
            
            source = result.document.source
            # Use metadata for better context if available
            title = result.document.metadata.get('title', source)
            date = result.document.metadata.get('date', '')
            
            header = f"{title} ({date})" if date else title
            content = result.document.content
            
            part = f"[Source: {header}]\n{content}"
            context_parts.append(part)
            total_chars += len(part)
        
        return "\n\n---\n\n".join(context_parts)
    
    def clear_index(self):
        """Clear all documents from the index."""
        self._index = None
        self._documents = []
        
        # Remove saved files
        index_file = os.path.join(self.index_path, "faiss.index")
        docs_file = os.path.join(self.index_path, "documents.pkl")
        
        for f in [index_file, docs_file]:
            if os.path.exists(f):
                os.remove(f)
        
        print("Index cleared")
    
    def get_indexed_sources(self) -> List[str]:
        """Get list of unique sources in the index."""
        return list(set(d.source for d in self._documents))
    
    def remove_source(self, source: str) -> int:
        """
        Remove all documents from a source.
        Note: This rebuilds the entire index.
        
        Args:
            source: Source to remove.
            
        Returns:
            Number of documents removed.
        """
        original_count = len(self._documents)
        remaining_docs = [d for d in self._documents if d.source != source]
        removed_count = original_count - len(remaining_docs)
        
        if removed_count == 0:
            return 0
        
        # Rebuild index
        self._documents = []
        self._index = None
        
        if remaining_docs:
            self._create_index()
            
            embeddings = self._model.encode([d.content for d in remaining_docs])
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            self._index.add(embeddings.astype('float32'))
            self._documents = remaining_docs
        
        self._save_index()
        
        return removed_count


# Singleton instance
_engine_instance: Optional[RAGEngine] = None


def get_rag_engine(
    model_name: str = None,
    index_path: str = "data/vector_store"
) -> RAGEngine:
    """Get or create RAG engine instance."""
    global _engine_instance
    
    if _engine_instance is None:
        _engine_instance = RAGEngine(model_name, index_path)
    
    return _engine_instance
