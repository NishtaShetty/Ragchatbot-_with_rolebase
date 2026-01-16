# Multilingual Role-Based RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with role-based access control, multilingual support, and advanced document processing capabilities including OCR and table extraction.

## 🌟 Features

- **Role-Based Access Control**: Separate document access for Field Team, Sales, and HR roles
- **Multilingual Support**: English, Kannada, Hindi, Tamil, Telugu, Malayalam, and more
- **Hybrid Search**: Combines semantic (vector) and keyword (BM25) search
- **Advanced OCR**: Handles scanned PDFs with table detection and extraction
- **Voice Input**: Speech-to-text support for multiple Indian languages
- **Real-time Translation**: Automatic query and response translation
- **Cross-Encoder Reranking**: Improves retrieval accuracy
- **Modern UI**: Clean, responsive web interfaces for chat and document ingestion

## 🏗️ Architecture

### Components

1. **Chat Server** (`server.py`) - Port 6001
   - Handles user queries with role-based filtering
   - Supports text and voice input
   - Multilingual query processing and response translation

2. **Ingestion Server** (`ingest.py`) - Port 6002
   - Document upload and processing
   - Role-based document organization
   - Vector store creation and management

3. **RAG Engine** (`rag/`)
   - `query_engine.py`: Query processing with role verification
   - `retriever.py`: Hybrid search (semantic + BM25)
   - `data_processing.py`: Document chunking and embedding
   - `groq_llm.py`: LLM integration via Groq API
   - `language_utils.py`: Multilingual ASR and translation
   - `ocr_loader.py`: OCR processing for scanned documents
   - `meta.py`: Advanced table extraction with LayoutLM

## 📋 Prerequisites

- Python 3.9+
- Tesseract OCR installed
- CUDA-compatible GPU (optional, for faster processing)
- Groq API key

### Install Tesseract OCR

**Windows:**
```bash
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Or use chocolatey:
choco install tesseract
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd <project-directory>
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your Groq API key from: https://console.groq.com/

5. **Prepare data directory structure**
```
data/
├── field_team/    # PDFs for field team
├── sales/         # PDFs for sales team
└── hr/            # PDFs for HR team
```

## 📊 Usage

### 1. Start the Ingestion Server

```bash
uvicorn ingest:app --host 0.0.0.0 --port 6002 --reload
```

Open `Documentingestion_frontend.html` in your browser and upload documents for each role.

### 2. Build Vector Store (Alternative)

Or run ingestion programmatically:
```bash
python -c "from rag import build_vector_store, Config; build_vector_store(Config())"
```

### 3. Start the Chat Server

```bash
uvicorn server:app --host 0.0.0.0 --port 6001 --reload
```

### 4. Access the Chat Interface

Open `chatbot_frontend.html` in your browser.

## 🔧 Configuration

Edit `rag/config.py` to customize:

```python
@dataclass
class Config:
    # Chunking
    chunk_size: int = 300
    overlap: int = 50
    
    # Embeddings
    embed_model_name: str = "all-MiniLM-L6-v2"
    
    # Retrieval
    top_k_retriever: int = 40
    rerank_top_k: int = 5
    semantic_weight: float = 0.5
    keyword_weight: float = 0.5
    
    # LLM
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
```

## 🎯 API Endpoints

### Chat Server (Port 6001)

**POST /chat**
```json
{
  "query": "What is the leave policy?",
  "language": "en",
  "role": "hr",
  "input_type": "text"
}
```

**Response:**
```json
{
  "answer": "According to the HR policy...",
  "sources": [
    {
      "source": "hr_policy.pdf",
      "page": 5,
      "section": "Leave Policy",
      "role": "hr"
    }
  ],
  "transcription": null,
  "original_language": "en",
  "was_translated": false
}
```

### Ingestion Server (Port 6002)

**POST /ingest**
- Form data with `role` (field_team/sales/hr) and `files` (PDF uploads)

**Response:**
```json
{
  "status": "success",
  "role": "hr",
  "message": "Ingested 3 document(s)",
  "files": ["policy.pdf", "handbook.pdf"],
  "total_chunks": 245
}
```

## 🌐 Supported Languages

- English (en)
- Kannada (kn)
- Hindi (hi)
- Tamil (ta)
- Telugu (te)
- Malayalam (ml)
- Marathi (mr)
- Bengali (bn)
- Gujarati (gu)
- Odia (or)
- Assamese (as)

## 🔒 Role-Based Access Control

The system enforces strict role-based access:

1. Documents are organized by role folders
2. Each chunk is tagged with its parent document's role
3. Retrieval filters by user role at multiple levels:
   - ChromaDB metadata filtering
   - BM25 role-specific indexes
   - Parent document verification
4. Cross-role access is prevented at query time

## 📁 Project Structure

```
.
├── server.py                    # Chat API server
├── ingest.py                    # Document ingestion server
├── chatbot_frontend.html        # Chat UI
├── Documentingestion_frontend.html  # Upload UI
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables
├── data/                        # Document storage (by role)
│   ├── field_team/
│   ├── sales/
│   └── hr/
├── chroma_db/                   # Vector database
└── rag/                         # RAG engine modules
    ├── config.py                # Configuration
    ├── query_engine.py          # Query processing
    ├── retriever.py             # Hybrid search
    ├── data_processing.py       # Document processing
    ├── groq_llm.py              # LLM integration
    ├── language_utils.py        # Multilingual support
    ├── ocr_loader.py            # OCR processing
    ├── meta.py                  # Table extraction
    ├── text_utils.py            # Text chunking
    ├── utils.py                 # Helper functions
    └── prompt.py                # Prompt templates
```

## 🐛 Troubleshooting

### Tesseract not found
```python
# Edit rag/meta.py and set the path manually:
tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### CUDA out of memory
```python
# In rag/language_utils.py, force CPU:
device = "cpu"
```

### ChromaDB collection not found
```bash
# Rebuild the vector store:
python -c "from rag import build_vector_store, Config; build_vector_store(Config())"
```

## 📝 Development

### Adding a New Role

1. Create folder in `data/` (e.g., `data/finance/`)
2. Add PDFs to the folder
3. Update frontend dropdowns in HTML files
4. Rebuild vector store

### Changing LLM Provider

Replace `rag/groq_llm.py` with your provider's implementation:
```python
class CustomLLMGenerator:
    def __init__(self, config: Config):
        # Initialize your LLM client
        pass
    
    def generate(self, prompt: str) -> str:
        # Call your LLM API
        pass
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- Groq for fast LLM inference
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- AI4Bharat for Indic language models
- Tesseract OCR for document processing

## 📧 Support

For issues and questions, please open a GitHub issue or contact [your-email@example.com]
