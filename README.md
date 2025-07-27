# Astra-rag-demo-
Interactive Streamlit app for semantic PDF Q&amp;A using AstraDB, Groq’s LLaMA3, and Hugging Face embeddings. Upload a PDF, ask questions, and get accurate answers with contextual chunk retrieval from a vector store. Built for fast RAG workflows and AI-powered document analysis.

# 📄 PDF Q&A App — Streamlit + AstraDB + Groq + HF Embeddings

Unlock answers from your PDFs with semantic search and LLM-powered intelligence. This project lets you upload any PDF, store meaningful chunks in AstraDB, and get accurate answers via Groq's lightning-fast LLaMA3 models—all wrapped in a sleek Streamlit interface.

## 🚀 Features

- 🔍 **Semantic Search** over PDF content using Hugging Face’s `bge-small-en`
- 🧠 **LLM Answers** using Groq’s blazing-fast LLaMA3 (via `langchain-groq`)
- 🗄️ **Persistent Chunk Storage** with AstraDB (NoSQL + Vectors)
- 📑 **Contextual Retrieval** for relevant document insights
- 🖼️ **Streamlit UI** for easy interaction and exploration

---

## 🧠 Tech Stack

| Component         | Tool/Library                 | Role                               |
|------------------|------------------------------|------------------------------------|
| LLM              | Groq LLaMA3                  | Generative answer generation       |
| Embedding Model  | Hugging Face `bge-small-en-v1.5` | Vector embedding of PDF chunks |
| Vector DB        | AstraDB (via Cassio)         | Semantic storage and retrieval     |
| UI Framework     | Streamlit                    | Interactive front-end              |
| PDF Parser       | PyPDF2                       | Text extraction from PDF           |

---

## ⚙️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/pdf-qa-streamlit.git
cd pdf-qa-streamlit
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

You’ll need:
- A PDF file (e.g. `attention.pdf`)
- Groq API key
- AstraDB credentials (Token + DB ID)

Use the Streamlit sidebar to input these securely.

---

## 💡 Usage

streamlit run app.py

## 🌐 Links

- [Groq API Docs](https://console.groq.com/docs)
- [AstraDB Vector DB](https://www.datastax.com/astra)
- [Hugging Face Embeddings](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [Streamlit](https://streamlit.io)
