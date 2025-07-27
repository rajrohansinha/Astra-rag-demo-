import streamlit as st
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import cassio

# --- Config ---
st.set_page_config(page_title="📄 PDF Q&A App", layout="centered")
st.title("📄 PDF-Based Q&A using AstraDB + Groq : Raj")

# --- API Input ---
with st.sidebar:
    st.header("🔑 Credentials")
    pdf_path = st.text_input("PDF file path (e.g. attention.pdf)", value="attention.pdf")
    groq_key = st.text_input("Groq API Key", type="password", value="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")
    astra_token = st.text_input("Astra DB Token", type="password", value="AstraCS:vCwxHYfhNjZQYDpsWroJxZvz:fa3914d1e5d696abd142d29aae29008e1c8113621dbed09dc4b1143f31f2bcf8")
    astra_db_id = st.text_input("Astra DB ID", value="3b0814ca-1f8e-4458-8a17-f0986718b395")
    st.markdown("---")
    st.caption("Upload your PDF and ask questions. Answers are generated from embedded chunks stored in AstraDB.")

# --- Load PDF ---
@st.cache_data
def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

raw_text = extract_text(pdf_path)

# --- Split and Embed ---
cassio.init(token=astra_token, database_id=astra_db_id)

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
llm = ChatGroq(groq_api_key=groq_key, model_name="llama3-8b-8192")

astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_streamlit_demo",
    session=None,
    keyspace=None,
)

splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200)
texts = splitter.split_text(raw_text)

if len(texts) > 0:
    astra_vector_store.add_texts(texts[:50])  # Only store top 50 chunks for demo

# --- Build Index ---
index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# --- Question Input ---
query = st.text_input("❓ Ask a question from the PDF")

if query:
    st.subheader("📣 Answer")
    try:
        answer = index.query(query, llm=llm).strip()
        st.success(answer)

        # Show top documents
        st.subheader("📄 Relevant Chunks")
        results = astra_vector_store.similarity_search_with_score(query, k=4)
        for doc, score in results:
            snippet = doc.page_content[:100].replace("\n", " ")
            st.write(f"🟡 [{score:.4f}] {snippet}...")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please enter a question above to get started.")
    if raw_text:
        st.subheader("📄 Extracted PDF Preview")
        st.write(raw_text[:1000] + ("..." if len(raw_text) > 1000 else ""))
    else:
        st.warning("No text could be extracted from the PDF. Please check the file path.")