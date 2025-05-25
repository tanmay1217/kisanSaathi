# Kisan Saathi - Agricultural Assistant

Kisan Saathi is an AI-powered agricultural assistant designed to help farmers and agricultural professionals get answers to their farming-related queries. The app uses a combination of a local dataset (KCC dataset) and live Internet search to provide accurate and relevant information.

---

## **Overview**

Kisan Saathi leverages advanced AI technologies such as semantic search and language models to provide context-based answers. The app integrates a local dataset (KCC dataset) for agricultural knowledge and falls back to live Internet search when no relevant local context is found. It features a user-friendly interface built with Streamlit, making it accessible to a wide range of users.

---

## **Features**

* **Local Dataset Integration**: Uses the KCC dataset for context-based answers.
* **Semantic Search**: Employs FAISS for efficient vector-based semantic search.
* **Fallback to Live Internet Search**: Provides answers from the Internet when local context is unavailable.
* **Interactive UI**: Built with Streamlit for a simple and user-friendly experience.
* **Dynamic Query Processing**: Real-time status updates during query handling.

---

## **Installation**

### **Prerequisites**

* Python 3.8 or higher
* Pip (Python package manager)

### **Steps**

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/kisan-saathi.git
   cd kisan-saathi
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up the `.env` file:

   * Create a `.env` file in the project root directory.
   * Add your API key for live Internet search:

     ```properties
     TAVILY_API_KEY="your-api-key-here"
     ```
     If you don't have yours you can use mine: TAVILY_API_KEY="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"


4. Preprocess the KCC dataset:

   ```bash
   python src/kcc_preprocessor.py
   ```

5. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

## **Dependencies**

The project uses the following Python libraries:

```plaintext
pandas==2.2.3
numpy==2.2.6
sentence-transformers==4.1.0
faiss-cpu==1.11.0
streamlit==1.45.1
torch==2.7.0
transformers==4.52.3
requests==2.32.3
python-dotenv==1.1.0
beautifulsoup4==4.13.4
duckduckgo_search==8.0.2
googlesearch-python==1.3.0
watchdog==6.0.0
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## **Launch Instructions**

1. **Preprocess the Dataset**:

   * Run the preprocessing script to clean and prepare the KCC dataset:

     ```bash
     python src/kcc_preprocessor.py
     ```

2. **Start the App**:

   * Launch the Streamlit app:

     ```bash
     streamlit run app.py
     ```

3. **Access the App**:

   * Open your browser and navigate to:

     ```
     http://localhost:8501
     ```

4. **Use the App**:

   * Enter your agricultural query in the text box.
   * Click the "Get Answer" button to process your query.
   * View the answer generated from the local dataset or fallback Internet search.

---

## **Project Structure**

```
KisanSaathi/
│
├── data/
│   ├── Raw/                # Raw KCC dataset
│   ├── Processed/          # Preprocessed dataset
│   └── Embeddings/         # Embeddings and FAISS index
│
├── src/
│   ├── pipeline_manager.py # Core pipeline logic
│   ├── kcc_preprocessor.py # Preprocessing script for KCC dataset
│   └── rag_pipeline.py     # Retrieval-Augmented Generation logic
│
├── app.py                  # Streamlit app entry point
├── requirements.txt        # Project dependencies
└── .env                    # Environment variables
```

---

## **Challenges Faced**

1. **System Limitations**:

   * The full KCC dataset could not be processed due to memory and computational constraints.
   * A subset of 50,000 records was processed to ensure meaningful results.

2. **Performance Optimization**:

   * Techniques like batching and incremental processing were explored to handle larger datasets.

---

## **Future Improvements**

1. **Scale to Full Dataset**:

   * Upgrade system resources or use cloud-based solutions to process the entire dataset.

2. **Enhanced Search**:

   * Integrate advanced search algorithms for better context retrieval.

3. **Multilingual Support**:

   * Add support for queries in regional languages.

---

## **Acknowledgments**

* The KCC dataset for providing valuable agricultural data.
* Open-source libraries like FAISS, PyTorch, and Streamlit for enabling this project.


