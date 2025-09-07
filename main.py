import os
os.environ['USER_AGENT'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"

import streamlit as st
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

load_dotenv()
client = OpenAI()

#vector embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="BookWise",
    embedding=embedding_model
)
st.title("🦜🔗 Quickstart App")
query = st.text_input("Enter your question:")

if query:
    search_results = vector_db.similarity_search(
        query=query
    )

    context = "\n\n\n".join([
        f"Page Content: {result.page_content}\n"
        f"Website: {result.metadata.get('source', 'Unknown')}\n"
        f"Page Title: {result.metadata.get('title', 'Unknown')}"
        for result in search_results
    ])

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
