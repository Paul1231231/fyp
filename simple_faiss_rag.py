"""
Minimal FAISS RAG example - easiest to understand
"""

import sglang as sgl
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class SimpleRAG:
    def __init__(self):
        # Embedding model
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # FAISS index (384 dimensions for all-MiniLM-L6-v2)
        self.index = faiss.IndexFlatIP(384)
        
        # Store documents
        self.docs = []
    
    def add(self, texts):
        """Add documents"""
        # Encode and normalize
        embeddings = self.encoder. encode(texts, normalize_embeddings=True)
        
        # Add to FAISS
        self.index.add(embeddings.astype('float32'))
        
        # Store texts
        self.docs.extend(texts)
    
    def search(self, query, k=3):
        """Search for similar documents"""
        # Encode query
        query_vec = self.encoder.encode([query], normalize_embeddings=True)
        
        # Search
        scores, indices = self.index.search(query_vec. astype('float32'), k)
        
        # Return documents
        return [self.docs[i] for i in indices[0]]


@sgl.function
def answer(s, question, context):
    """Generate answer from context"""
    s += sgl.user(f"Context: {context}\n\nQuestion: {question}")
    s += sgl.assistant(sgl.gen("response", max_tokens=200))


def quick_demo():
    # Setup
    rag = SimpleRAG()
    
    # Add knowledge
    rag.add([
        "FAISS is a library for efficient similarity search of dense vectors.",
        "SGLang is a framework for programming language models.",
        "RAG combines retrieval with generation for better answers."
    ])
    
    # Start LLM
    runtime = sgl.Runtime(model_path="meta-llama/Llama-3.1-8B-Instruct")
    sgl.set_default_backend(runtime)
    
    # Query
    question = "What is FAISS?"
    docs = rag.search(question, k=2)
    context = " ".join(docs)
    
    result = answer.run(question=question, context=context)
    print(f"Q: {question}")
    print(f"A: {result['response']}")
    
    runtime.shutdown()


if __name__ == "__main__":
    quick_demo()