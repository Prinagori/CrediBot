from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def get_qa_chain():
    # Embedding setup
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="db", embedding_function=embeddings)

    # Top 2 most relevant chunks only
    retriever = vectordb.as_retriever(search_kwargs={"k": 6})

    # Define prompt
    prompt_template = PromptTemplate.from_template("""
You are a helpful assistant who answers user questions using the context below.

- Only answer using the context provided.
- Extract numbers, facts, or fees directly from the context when available.
- If the exact answer is not in the context, say: "Sorry, I couldn’t find that information in the documents provided."

Context:
{context}

Question: {question}
Answer:
""")

    # Groq LLM setup
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)


    return qa_chain
