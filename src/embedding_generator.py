import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

class KCCEmbedder:
    def __init__(self, input_path: str, embedding_output_path: str, metadata_output_path: str):
        self.input_path = input_path
        self.embedding_output_path = embedding_output_path
        self.metadata_output_path = metadata_output_path
        self.df = None
        self.embeddings = None
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    def load_data(self):
        self.df = pd.read_csv(self.input_path)
        print(f"Loaded {len(self.df)} rows from {self.input_path}.")

    def generate_embeddings(self):
        chunks = self.df['chunk'].tolist()
        # print(f"Generating embeddings for {len(chunks)} chunks...")
        self.embeddings = self.model.encode(
            chunks,
            show_progress_bar=True,
            batch_size=64
        )

    def save_outputs(self):
        np.save(self.embedding_output_path, self.embeddings)
        self.df.to_csv(self.metadata_output_path, index=False)
        print(f"Embeddings saved to: {self.embedding_output_path}")
        print(f"Metadata saved to: {self.metadata_output_path}")

    def run(self):
        self.load_data()
        self.generate_embeddings()
        self.save_outputs()
