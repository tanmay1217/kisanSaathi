import os
import asyncio
import platform
import streamlit as st
from src.pipeline_manager import KisanSaathiPipeline

# Fix Windows event loop issue
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Disable Streamlit's file watcher for PyTorch
os.environ["STREAMLIT_WATCHDOG"] = "false"

# Define paths (adjust paths as per your project structure)
raw_path = "data/Raw/kcc_dataset.csv"
processed_path = "data/Processed/kcc_preprocessed_chunks.csv"
embedding_path = "data/Embeddings/kcc_embeddings.npy"
metadata_path = "data/Embeddings/kcc_metadata.csv"
index_path = "data/Embeddings/kcc_faiss.index"

# Initialize the pipeline
pipeline = KisanSaathiPipeline(
    raw_path=raw_path,
    processed_path=processed_path,
    embedding_path=embedding_path,
    metadata_path=metadata_path,
    index_path=index_path
)


# Streamlit UI
st.title("Kisan Saathi - Agricultural Assistant")
st.subheader("Get answers to your agricultural queries using local datasets or fallback Internet search.")

# Run the pipeline initialization steps
with st.spinner("Initializing pipeline..."):
    try:
        pipeline.run()  # This ensures preprocessing, embedding generation, and FAISS index creation are done.
    except Exception as e:
        st.error(f"Pipeline initialization failed: {e}")

# Input: Natural-language query
query = st.text_area("Enter your query:", height=100)

if st.button("Get Answer"):
    if query.strip():
        with st.spinner("Processing your query..."):
            try:
                result = pipeline.query_llm_with_prompt(query)

                # Handle the result
                if result.get("invoke_live_search"):  # No local context found
                    st.warning("No relevant local context found.")
                    st.markdown(f"**Live Search Result:**\n{result.get('live_search_result', 'No results found.')}")
                elif result.get("answer"):  # Context found and answer generated
                    st.success("Answer generated:")
                    st.markdown(f"**Answer:**\n{result['answer']}")
                else:
                    st.error("No answer could be generated.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.error("Please enter a query.")

# Footer
st.markdown("---")
st.markdown("**Kisan Saathi** | Powered by Local Context and Internet Search")
