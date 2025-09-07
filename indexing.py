import os
os.environ['USER_AGENT'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"

from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

loader = WebBaseLoader("https://docs.chaicode.com/youtube/getting-started/")
docs = loader.load()

#chunking
text_splitter =  RecursiveCharacterTextSplitter(
    chunk_size =1000,
    chunk_overlap=600
)
split_docs = text_splitter.split_documents(documents=docs)

#vector embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url="http://localhost:6333",
    collection_name="BookWise",
    embedding=embedding_model
)



