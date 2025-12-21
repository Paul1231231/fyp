"""
Complete RAG System using SGLang and FAISS
Optimized for small to medium-sized databases
"""

import sglang as sgl
from typing import List, Dict, Optional
import numpy as np
import faiss
from dataclasses import dataclass
import pickle
import os


@dataclass
class Document:
    """Document with metadata"""
    id: str
    text: str
    metadata: Dict = None


class FAISSRAGSystem:
    """RAG system using FAISS for efficient vector search"""
    
    def __init__(
        self,
        model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        embedding_dim: int = 384,  # Dimension for all-MiniLM-L6-v2
        use_gpu: bool = False
    ):
        self.llm_model_path = model_path
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        
        # Initialize FAISS index
        # Using IndexFlatIP for small databases (inner product/cosine similarity)
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        # Move to GPU if requested and available
        if use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 
                0, 
                self.index
            )
            print("Using GPU for FAISS")
        
        # Store documents and metadata
        self.documents: List[Document] = []
        self.runtime = None
        self.embedding_model = None
        
    def load_embedding_model(self, model_name: str = "all-MiniLM-L6-v2"):
        """Load sentence transformer model for embeddings"""
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer(model_name)
        print(f"Loaded embedding model:  {model_name}")
        
    def add_documents(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict]] = None
    ):
        """Add documents to FAISS index"""
        if self.embedding_model is None:
            self.load_embedding_model()
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents
        start_id = len(self.documents)
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            doc = Document(
                id=f"doc_{start_id + i}",
                text=text,
                metadata=metadata
            )
            self.documents.append(doc)
        
        print(f"Added {len(texts)} documents. Total:  {len(self.documents)}")
        print(f"FAISS index size: {self. index.ntotal} vectors")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 3
    ) -> List[Dict]:
        """Retrieve most relevant documents using FAISS"""
        if self.embedding_model is None:
            raise ValueError("Embedding model not loaded")
        
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve documents
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):  # Valid index
                doc = self.documents[idx]
                results.append({
                    'document': doc,
                    'score': float(distance),
                    'text': doc.text,
                    'metadata': doc. metadata
                })
        
        print(f"Retrieved {len(results)} documents for query:  '{query[: 50]}...'")
        return results
    
    def start_runtime(self):
        """Start SGLang runtime"""
        self. runtime = sgl.Runtime(model_path=self.llm_model_path)
        sgl.set_default_backend(self.runtime)
        print(f"SGLang runtime started with model: {self.llm_model_path}")
    
    def save(self, path: str = "rag_system"):
        """Save FAISS index and documents to disk"""
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        if self.use_gpu:
            # Move to CPU before saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, f"{path}/faiss. index")
        else:
            faiss.write_index(self.index, f"{path}/faiss.index")
        
        # Save documents
        with open(f"{path}/documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        
        print(f"Saved RAG system to {path}/")
    
    def load(self, path: str = "rag_system"):
        """Load FAISS index and documents from disk"""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}/faiss.index")
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,
                self. index
            )
        
        # Load documents
        with open(f"{path}/documents.pkl", 'rb') as f:
            self.documents = pickle.load(f)
        
        print(f"Loaded RAG system from {path}/")
        print(f"Index size: {self.index.ntotal}, Documents: {len(self.documents)}")
    
    def shutdown(self):
        """Cleanup resources"""
        if self.runtime:
            self.runtime.shutdown()
            print("Runtime shutdown complete")


# SGLang functions for RAG
@sgl.function
def rag_query(s, query: str, contexts: List[str]):
    """Basic RAG query"""
    context_text = "\n\n".join([
        f"[Document {i+1}]\n{doc}" 
        for i, doc in enumerate(contexts)
    ])
    
    s += sgl.system("You are a helpful assistant that answers questions based on provided context.")
    s += sgl.user(f"""Context: 
{context_text}

Question: {query}

Answer based on the context above. If the information is not in the context, say so.""")
    
    s += sgl.assistant(sgl.gen("answer", max_tokens=512, temperature=0.7))


@sgl.function
def rag_with_citations(s, query: str, contexts: List[str]):
    """RAG with source citations"""
    context_text = "\n\n".join([
        f"[{i+1}] {doc}" 
        for i, doc in enumerate(contexts)
    ])
    
    s += sgl. system("""You are a helpful assistant that provides accurate answers with citations.
Always cite your sources using [1], [2], etc. when referencing the context.""")
    
    s += sgl.user(f"""Context:
{context_text}

Question: {query}

Provide a detailed answer and cite sources using [1], [2], etc.""")
    
    s += sgl.assistant(sgl.gen("answer", max_tokens=512, temperature=0.7))


@sgl.function
def rag_with_reasoning(s, query: str, contexts: List[str]):
    """RAG with chain-of-thought reasoning"""
    context_text = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(contexts)])
    
    s += sgl.system("You are a thoughtful assistant that reasons step-by-step.")
    s += sgl.user(f"""Context:
{context_text}

Question: {query}

Think through this step-by-step before answering.""")
    
    s += sgl.assistant("Let me think through this:\n\n" + sgl.gen("reasoning", max_tokens=256))
    s += sgl.assistant("\n\nFinal Answer:\n" + sgl.gen("answer", max_tokens=256))


def main():
    """Main demo function"""
    print("=" * 80)
    print("RAG System with FAISS and SGLang")
    print("=" * 80)
    
    # Initialize RAG system
    rag = FAISSRAGSystem(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        use_gpu=False  # Set to True if you have GPU
    )
    
    # Load embedding model
    rag.load_embedding_model("all-MiniLM-L6-v2")
    
    # Sample documents
    documents = [
        "Python is a high-level, interpreted programming language created by Guido van Rossum.  "
        "First released in 1991, Python emphasizes code readability with significant whitespace.  "
        "It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        
        "FAISS (Facebook AI Similarity Search) is a library developed by Meta Research for efficient "
        "similarity search and clustering of dense vectors. It contains algorithms that search in sets "
        "of vectors of any size, even ones that don't fit in RAM.  It's particularly useful for RAG systems.",
        
        "SGLang is a structured generation language for programming large language models. It provides "
        "primitives for constrained generation, multiple chained generation calls, advanced prompting, "
        "control flow, and efficient execution with RadixAttention for KV cache reuse.",
        
        "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with "
        "text generation. It works by retrieving relevant documents from a knowledge base and using them "
        "as context for the language model to generate more accurate, grounded responses.",
        
        "Vector databases store high-dimensional embeddings and enable semantic search through similarity "
        "comparisons. For small databases (< 100K documents), FAISS is often the best choice due to its "
        "simplicity and speed.  For larger production systems, consider Pinecone, Weaviate, or Qdrant.",
        
        "The sentence-transformers library provides an easy way to compute dense vector representations "
        "for sentences and paragraphs. Popular models include all-MiniLM-L6-v2 (fast, 384 dimensions) "
        "and all-mpnet-base-v2 (more accurate, 768 dimensions)."
    ]
    
    # Add metadata
    metadatas = [
        {"topic": "python", "source": "programming_guide"},
        {"topic": "faiss", "source": "ml_tools"},
        {"topic": "sglang", "source": "llm_frameworks"},
        {"topic": "rag", "source": "ml_techniques"},
        {"topic": "vector_db", "source": "ml_infrastructure"},
        {"topic": "embeddings", "source": "ml_tools"}
    ]
    
    # Add documents to RAG system
    rag.add_documents(documents, metadatas)
    
    # Save the index (optional)
    rag.save("my_rag_system")
    
    # Start SGLang runtime
    print("\nStarting SGLang runtime...")
    rag.start_runtime()
    
    # Example queries
    queries = [
        "What is FAISS and why is it good for small databases?",
        "Explain how RAG works",
        "What embedding models should I use?"
    ]
    
    print("\n" + "=" * 80)
    print("Running RAG Queries")
    print("=" * 80)
    
    for query in queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {query}")
        print(f"{'=' * 80}")
        
        # Retrieve relevant documents
        results = rag.retrieve(query, top_k=2)
        
        # Extract texts and show scores
        print("\nRetrieved Documents:")
        for i, result in enumerate(results):
            print(f"  [{i+1}] Score: {result['score']:.4f} | Topic: {result['metadata']. get('topic', 'N/A')}")
            print(f"      {result['text'][: 100]}...")
        
        contexts = [r['text'] for r in results]
        
        # Simple RAG
        print("\n--- Simple RAG ---")
        state = rag_query. run(query=query, contexts=contexts)
        print(f"\n{state['answer']}")
        
        # RAG with citations
        print("\n--- RAG with Citations ---")
        state = rag_with_citations.run(query=query, contexts=contexts)
        print(f"\n{state['answer']}")
    
    # Cleanup
    rag.shutdown()
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()