# CrediBot: AI-Powered Credit Card Policy Assistant

CrediBot is a Retrieval-Augmented Generation (RAG) application that answers credit card policy-related questions for Chase and AMEX. It extracts accurate details such as APR, annual fees, and terms from official PDF documents and displays the response with relevant sources.

---

## Problem Statement

Can we build an intelligent system that can:

- Provide instant answers about credit card policies like APR, annual fees, and terms?
- Retrieve responses directly from official documents without hallucination?
- Enable users to access financial information quickly and transparently?

---

## Dataset

- **Source:** Official credit card agreement PDFs from **Chase** and **AMEX**, downloaded from CFPB (Consumer Financial Protection Bureau).
- **Content:** Details about APR, penalty fees, interest calculation, and terms.
- **Processing:**  
  - Extracted text from PDFs using `PyMuPDF`.
  - Split text into smaller chunks for semantic search.
  - Created embeddings for chunks using **Sentence Transformers** and stored in **ChromaDB**.

---

## Tech Stack

- **LangChain** – RAG pipeline implementation
- **Groq API (LLaMA 3)** – Fast and efficient language model
- **ChromaDB** – Vector database for semantic retrieval
- **Sentence Transformers** – For generating text embeddings
- **Streamlit** – Interactive frontend UI

---

## Features

- Ask natural language questions like:
  - *“What is the APR for Chase Freedom Unlimited?”*
  - *“What is the annual fee for AMEX Gold card?”*
- Answers are based **only** on provided official documents.
- Displays relevant source excerpts for transparency.

---

## Demo

[Watch the Demo](https://youtu.be/Hpqgh_RNh60)

---


