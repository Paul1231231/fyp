"""
Advanced FAISS features for RAG
- IVF index for larger databases
- PQ compression for memory efficiency
- Batch operations
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class AdvancedFAISSRAG:
    """RAG with advanced FAISS indexing strategies"""
    
    def __init__(self, index_type="flat"):
        """
        index_type options:
        - 'flat': Exact search, best for < 1M vectors
        - 'ivf':  Faster search with slight accuracy tradeoff, good for 1M-10M vectors
        - 'ivfpq': Memory efficient, good for 10M+ vectors
        """
        self. encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.dim = 384
        self.index_type = index_type
        self.docs = []
        
        # Create appropriate index
        if index_type == "flat":
            # Exact search using inner product (cosine similarity)
            self.index = faiss.IndexFlatIP(self.dim)
            print("Using Flat index (exact search)")
            
        elif index_type == "ivf":
            # IVF:  Inverted File Index for faster search
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss. IndexIVFFlat(quantizer, self.dim, nlist)
            self.needs_training = True
            print(f"Using IVF index with {nlist} clusters")
            
        elif index_type == "ivfpq":
            # IVF + Product Quantization for memory efficiency
            nlist = 100
            m = 8  # Number of subquantizers
            bits = 8  # Bits per subquantizer
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, bits)
            self.needs_training = True
            print(f"Using IVFPQ index (compressed)")
        else:
            raise ValueError(f"Unknown index type: {index_type}")
    
    def add_documents(self, texts, batch_size=32):
        """Add documents in batches"""
        # Encode in batches for efficiency
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        ).astype('float32')
        
        # Train index if needed
        if hasattr(self, 'needs_training') and self.needs_training:
            print("Training index...")
            self.index.train(embeddings)
            self.needs_training = False
        
        # Add to index
        self.index.add(embeddings)
        self.docs.extend(texts)
        
        print(f"Added {len(texts)} documents. Total: {len(self.docs)}")
    
    def search(self, query, k=5, nprobe=10):
        """
        Search with optional parameters
        nprobe: number of clusters to search (for IVF indices)
        """
        # Set search parameters for IVF
        if self. index_type in ["ivf", "ivfpq"]:
            self.index.nprobe = nprobe
        
        # Encode query
        query_vec = self. encoder.encode(
            [query],
            normalize_embeddings=True
        ).astype('float32')
        
        # Search
        scores, indices = self.index.search(query_vec, k)
        
        # Return results with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.docs):
                results.append({
                    'text': self.docs[idx],
                    'score': float(score),
                    'index': int(idx)
                })
        
        return results
    
    def save(self, path):
        """Save index and documents"""
        faiss.write_index(self. index, f"{path}. index")
        np.save(f"{path}_docs.npy", self.docs, allow_pickle=True)
        print(f"Saved to {path}")
    
    def load(self, path):
        """Load index and documents"""
        self.index = faiss.read_index(f"{path}.index")
        self.docs = np.load(f"{path}_docs.npy", allow_pickle=True).tolist()
        print(f"Loaded from {path}")
        print(f"Index has {self.index.ntotal} vectors")


def demo_index_comparison():
    """Compare different FAISS index types"""
    import time
    
    # Sample documents (expand this for real testing)
    docs = [
        "FAISS supports multiple index types for different use cases.",
        "Flat index provides exact search results.",
        "IVF index trades some accuracy for speed.",
        "Product Quantization compresses vectors to save memory.",
    ] * 100  # Duplicate to have more docs
    
    for index_type in ["flat", "ivf", "ivfpq"]:
        print(f"\n{'='*60}")
        print(f"Testing {index_type. upper()} index")
        print(f"{'='*60}")
        
        rag = AdvancedFAISSRAG(index_type=index_type)
        
        # Add documents
        start = time.time()
        rag.add_documents(docs)
        add_time = time.time() - start
        
        # Search
        start = time.time()
        results = rag.search("What is FAISS?", k=3)
        search_time = time.time() - start
        
        print(f"Add time: {add_time:.4f}s")
        print(f"Search time: {search_time:.4f}s")
        print(f"\nTop result: {results[0]['text'][: 80]}...")
        print(f"Score: {results[0]['score']:.4f}")


if __name__ == "__main__":
    demo_index_comparison()