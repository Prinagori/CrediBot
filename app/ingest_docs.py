from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path

def load_documents(folder_path):
    docs = []
    for file_path in Path(folder_path).glob("*.pdf"):
        loader = PyMuPDFLoader(str(file_path))
        docs.extend(loader.load())
    return docs

def create_vector_db(docs, persist_dir="db"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # ✅ force CPU to avoid GPU/meta tensor issues
    )

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb

if __name__ == "__main__":
    raw_docs = load_documents("data")
    create_vector_db(raw_docs)
