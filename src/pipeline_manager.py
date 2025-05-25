import time
import os
from .kcc_preprocessor import KCCPreprocessor
from tavily import TavilyClient
from dotenv import load_dotenv
load_dotenv()

class KisanSaathiPipeline:
    def __init__(self, raw_path, processed_path, embedding_path, metadata_path, index_path):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.embedding_path = embedding_path
        self.metadata_path = metadata_path
        self.index_path = index_path
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")  # Store key in env var
        self.tavily = TavilyClient(api_key=self.tavily_api_key)

    def preprocess_data(self):
        if os.path.exists(self.processed_path):
            print(f"[SKIP] Preprocessed data already exists at: {self.processed_path}")
        else:
            print("[RUN] Running preprocessor...")
            preprocessor = KCCPreprocessor(self.raw_path, self.processed_path)
            preprocessor.run()

    def generate_embeddings(self):
        if os.path.exists(self.embedding_path) and os.path.exists(self.metadata_path):
            print(f"[SKIP] Embeddings already exist at: {self.embedding_path}")
        else:
            print("[RUN] Generating embeddings...")
            from src.embedding_generator import KCCEmbedder
            embedder = KCCEmbedder(
                input_path=self.processed_path,
                embedding_output_path=self.embedding_path,
                metadata_output_path=self.metadata_path
            )
            embedder.run()

    def create_faiss_index(self):
        if os.path.exists(self.index_path):
            print(f"[SKIP] FAISS index already exists at: {self.index_path}")
        else:
            print("[RUN] Creating FAISS index...")
            from src.vector_store import VectorStore
            vector_store = VectorStore(
                embedding_path=self.embedding_path,
                index_path=self.index_path
            )
            vector_store.run()

    def query_llm_with_prompt(self, prompt):
        print("[RUN] Generating answer using RAG pipeline...")
        try:
            from .rag_pipeline import generate_answer
            result = generate_answer(prompt)

            if result.get("invoke_live_search"):  # No relevant context found
                print("No relevant local context found. Invoking live Internet search...")
                live_search_result = self.perform_live_internet_search(prompt)
                print(f"[LIVE SEARCH RESULT]\n{live_search_result}")

                # Return a dict including live search result and source info
                return {
                    "invoke_live_search": True,
                    "live_search_result": live_search_result,
                    "answer": None,
                    "context_used": None,
                    "source": "Fallback Internet search"
                }

            else:  # Context found and answer generated
                print(f"[ANSWER] {result['answer']}")
                return {
                    "invoke_live_search": False,
                    "live_search_result": None,
                    "answer": result["answer"],
                    "context_used": result.get("context_used", []),
                    "source": "KCC dataset"
                }

        except Exception as e:
            print(f"[ERROR] Failed to generate answer: {e}")
            return {
                "invoke_live_search": False,
                "live_search_result": None,
                "answer": None,
                "context_used": None,
                "source": "Error",
                "error": str(e)
            }


    def perform_live_internet_search(self, query):
        try:
            response = self.tavily.search(
                query=query,
                search_depth="advanced",  # deeper, better results
                include_answer=True,
                include_raw_content=False
            )
            return response.get("answer", "No answer found.")
        except Exception as e:
            print(f"[ERROR] Tavily search failed: {e}")
            return "Live internet search failed."

    def run(self, ask_query=True):
        self.preprocess_data()
        self.generate_embeddings()
        self.create_faiss_index()
