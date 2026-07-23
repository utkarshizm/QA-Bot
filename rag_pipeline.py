import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import numpy as np
from typing import List, Tuple

class RAGPipeline:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel('gemini-1.5-flash')
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.chroma = chromadb.Client()
        self.collection = self.chroma.get_or_create_collection("docs")
        self.chunks = []
    
    def index_document(self, pdf_bytes: bytes, filename: str):
        # Extract text
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        
        # Chunk
        chunk_size = 1000
        overlap = 150
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 100:
                chunks.append({
                    "text": chunk,
                    "source": filename,
                    "page": i // chunk_size + 1
                })
        
        # Embed and store
        embeddings = self.embedder.encode([c["text"] for c in chunks])
        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=[c["text"] for c in chunks],
            metadatas=chunks,
            ids=ids
        )
        self.chunks.extend(chunks)
    
    def query(self, question: str, k: int = 3, rerank: bool = False) -> Tuple[str, List[dict]]:
        # Embed question
        q_embedding = self.embedder.encode([question])
        
        # Retrieve
        results = self.collection.query(
            query_embeddings=q_embedding.tolist(),
            n_results=k
        )
        
        contexts = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            contexts.append({
                "text": doc,
                "source": meta['source'],
                "page": meta['page']
            })
        
        # Build prompt
        context_text = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(contexts)])
        prompt = f"""Answer the question using ONLY the provided context. 
If the answer isn't in the context, say "I don't have enough information."
Cite sources using [1], [2], etc.

Context:
{context_text}

Question: {question}

Answer:"""
        
        response = self.llm.generate_content(prompt)
        return response.text, contexts
    
    def clear(self):
        self.chroma.delete_collection("docs")
        self.collection = self.chroma.get_or_create_collection("docs")
        self.chunks = []
