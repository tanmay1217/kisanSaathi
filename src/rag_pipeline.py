import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import os

# Check if FAISS index file exists
if not os.path.exists("data/Embeddings/kcc_faiss.index"):
    raise FileNotFoundError("FAISS index file not found at 'data/Embeddings/kcc_faiss.index'")

# Load vector store and data
index = faiss.read_index("data/Embeddings/kcc_faiss.index")

# Check if processed chunks file exists
if not os.path.exists("data/Processed/kcc_preprocessed_chunks.csv"):
    raise FileNotFoundError("Processed chunks file not found at 'data/Processed/kcc_preprocessed_chunks.csv'")
df_chunks = pd.read_csv("data/Processed/kcc_preprocessed_chunks.csv")

# Load sentence transformer model (same used for embeddings)
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Normalize function (must be same as during indexing)
def normalize_vector(vec):
    vec = np.asarray(vec)
    faiss.normalize_L2(vec)
    return vec

# Function to search local FAISS index WITHOUT threshold filtering
def semantic_search(query, top_k=5):
    query_vector = model.encode([query])
    query_vector = normalize_vector(query_vector)
    # print(f"[DEBUG] Query vector shape: {query_vector.shape}")

    distances, indices = index.search(query_vector, top_k)

    print(f"[DEBUG] distances: {distances}")
    # print(f"[DEBUG] indices: {indices}")

    context_results = []
    for i, dist in zip(indices[0], distances[0]):
        context_results.append({
            "score": float(dist),
            "text": df_chunks.iloc[i]["chunk"]
        })
    return context_results

# Function to query the local LLM (Ollama)
def query_local_llm(prompt, model_name="gemma:2b"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 300
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to query LLM: {e}")
        if response is not None:
            print(f"[ERROR] Response content: {response.text}")
        return "Error querying local LLM."

def generate_answer(query, top_k=5, threshold=0.72):
    results = semantic_search(query, top_k=top_k)

    if results:
        context_scores = [r["score"] for r in results]
        max_score = max(context_scores)

        if max_score < threshold:
            print("[DEBUG] No context passes the threshold. Triggering live internet search.")
            return {
                "answer": None,
                "source": None,
                "context_used": [],
                "invoke_live_search": True
            }

        # Use all contexts that meet threshold
        context_texts = [r["text"] for r in results if r["score"] >= threshold]

        context = "\n---\n".join(context_texts)
        prompt = f"""You are an agricultural assistant helping Indian farmers.

Use the following context from Kisan Call Center (KCC) data to answer the user's question. Be accurate and concise.
Return your answer in a clean, readable format using bullet points, numbered lists, or short paragraphs—whichever fits best.
Context:
{context}

Question: {query}
Answer:"""
        answer = query_local_llm(prompt)
        return {
            "answer": answer,
            "source": "KCC dataset",
            "context_used": context_texts,
            "invoke_live_search": False
        }

    else:
        print("[DEBUG] No context found in FAISS index.")
        return {
            "answer": None,
            "source": None,
            "context_used": [],
            "invoke_live_search": True
        }

