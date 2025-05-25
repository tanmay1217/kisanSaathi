import faiss
import numpy as np
import os

class VectorStore:
    def __init__(self, embedding_path: str, index_path: str):
        self.embedding_path = embedding_path
        self.index_path = index_path
        self.index = None

    def load_embeddings(self):
        if not os.path.exists(self.embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {self.embedding_path}")
        embeddings = np.load(self.embedding_path)
        print(f"Embeddings loaded. Shape: {embeddings.shape}")
        return embeddings

    def create_index(self, embeddings):
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        embedding_dim = embeddings.shape[1]

        # Create a FAISS index using Inner Product (IP)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(embeddings)
        print(f"Number of vectors indexed: {self.index.ntotal}")

    def save_index(self):
        if self.index is None:
            raise ValueError("Index has not been created. Call `create_index` first.")
        faiss.write_index(self.index, self.index_path)
        print(f"FAISS index saved to '{self.index_path}'.")

    def run(self):
        embeddings = self.load_embeddings()
        self.create_index(embeddings)
        self.save_index()