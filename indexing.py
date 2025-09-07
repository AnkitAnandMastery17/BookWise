import os
os.environ['USER_AGENT'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"

from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient

load_dotenv()
import os, streamlit as st

QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL", ""))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY", ""))

if not QDRANT_URL:
    st.error("QDRANT_URL is not set. Add it to .streamlit/secrets.toml or your environment.")
    st.stop()

URL="https://docs.chaicode.com/youtube/getting-started/"
loader = WebBaseLoader(URL)
docs = loader.load()

#chunking
text_splitter =  RecursiveCharacterTextSplitter(
    chunk_size =1000,
    chunk_overlap=600,
    add_start_index=True
)
split_docs = text_splitter.split_documents(documents=docs)

for i, d in enumerate(split_docs, start=1):
    d.metadata["chunk_id"] = i  # stable per-document chunk index[18][10]
    d.metadata.setdefault("source", URL)  # ensure URL present for citing[21]
    d.metadata.setdefault("title", "ChaiCode: Getting Started") 

#vector embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name="BookWise",
    embedding=embedding_model
)



