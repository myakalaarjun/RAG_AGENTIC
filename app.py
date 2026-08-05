import os
import faiss
from dotenv import load_dotenv

from fastapi import FastAPI
from langserve import add_routes
import uvicorn

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ----------------------------
# Load API Key
# ----------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ----------------------------
# Initialize LLM
# ----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
)

# ----------------------------
# Knowledge Base
# ----------------------------
big_paragraph = """
The Internet is a global system of interconnected computer networks that uses the Internet protocol suite (TCP/IP) to communicate between networks and devices.

The origins of the Internet date back to the development of packet switching and research commissioned by the United States Department of Defense in the 1960s. The primary precursor network was ARPANET.

Today, the Internet supports cloud computing, video conferencing, online gaming, social media, and much more.
"""

documents = [Document(page_content=big_paragraph)]

# ----------------------------
# Split Documents
# ----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = text_splitter.split_documents(documents)

# ----------------------------
# Embeddings
# ----------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

embedding_dim = len(embeddings.embed_query("hello"))

index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

vector_store.add_documents(chunks)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

# ----------------------------
# Prompt
# ----------------------------
rag_prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Use ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# ----------------------------
# LangServe
# ----------------------------
app = FastAPI(
    title="Internet RAG",
    version="1.0",
    description="RAG with LangServe",
)

add_routes(
    app,
    rag_chain,
    path="/agent",
)

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )
