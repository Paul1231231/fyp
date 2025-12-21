"""
Setup script for FAISS-based RAG system
"""

import subprocess
import sys


def install():
    packages = [
        "faiss-cpu",  # Use "faiss-gpu" if you have CUDA
        "sentence-transformers",
        "sglang[all]",
        "numpy",
    ]
    
    print("Installing packages for FAISS RAG system...")
    for pkg in packages:
        print(f"\nInstalling {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    
    print("\n" + "="*80)
    print("Installation complete!")
    print("="*80)
    print("\nRun these commands:")
    print("  python simple_faiss_rag.py          # Minimal example")
    print("  python rag_sglang_faiss.py          # Full-featured RAG")
    print("  python faiss_advanced_features.py   # Advanced FAISS features")


if __name__ == "__main__":
    install()