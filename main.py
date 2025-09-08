import os
os.environ['USER_AGENT'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"

import streamlit as st
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI


OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY", ""))

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Add it to .streamlit/secrets.toml or your environment.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

load_dotenv()
client = OpenAI()

#vector embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_db = QdrantVectorStore.from_existing_collection(
    api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
    collection_name="BookWise",
    embedding=embedding_model
)
st.title("🦜🔗 Book Wise Chat")
query = st.text_input("Enter your question:")

if query:
    search_results = vector_db.similarity_search(
        query=query
    )

    context = "\n\n\n".join(
        [
            f"Page Content: {doc.page_content}\n"
            f"Website: {doc.metadata.get('source', 'Unknown')}\n"
            f"Page Title: {doc.metadata.get('title', 'Unknown')}"
            for doc in search_results
        ]
    )
  

    SYSTEM_PROMPT = f"""
    You are a helpful AI Assistant who answers user queries based on the available context
    retrieved from web pages along with their source URLs and page titles.
    
    You should only answer the user based on the following context and guide the user
    to visit the source website for more detailed information.

    Context:
    {context}
    """

    chat_completion = client.chat.completions.create(
        model="gpt-4o",  # Fixed model name
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
    )

    # Replace print() with Streamlit display
    st.write("🤖:", chat_completion.choices[0].message.content)
