# BookWise - AI Chatbot for “Getting Started | Chai aur Docs”
> A lightweight, retrieval‑augmented Streamlit chatbot that indexes content, stores embeddings in Qdrant, and answers questions grounded in the source with links back to the exact page/section.
## This project is configured as a chatbot for the website “[Getting Started | Chai aur Docs](https://docs.chaicode.com/youtube/getting-started/)”

# Tech Stack
 - __UI:__ Streamlit.
- __Retrieval:__ LangChain (loaders, splitters) + Qdrant vector store.
- __Models:__ OpenAI chat + embeddings (configurable in code).

# Quick Start
__Prerequisites:__ Python 3.9+, Git; Docker if running Qdrant locally.

1. __Clone and install:__
- `git clone https://github.com/AnkitAnandMastery17/BookWise`
- `cd BookWise`
- `python -m venv .venv`
  
2. __Start your virtual environment__
  
3. __Install all the required files__

    `pip install -r requirements.txt`
4. __Configuration__
 - Create a folder .streamlit
 - Inside the folder create a file secrets.toml.
 - Add 
 `OPENAI_API_KEY = "<Your Api Key>"`         
 `QDRANT_URL = "http://localhost:6333"`

5. __Run Qdrant locally via Docker:__
   
   `docker compose up -d`
7. __Run `python indexing.py`__

    > To confirm your qdrant is running. visit `http://localhost:6333/dashboard`. Go to collections, you will see a new collection named BookWise.
   
<img width="1822" height="892" alt="image" src="https://github.com/user-attachments/assets/b23aa1a9-54b9-433f-a3f1-4a043138587f" />

8. __Run the app__
`streamlit run main.py`
9. Go to http://localhost:8501

<img width="1898" height="939" alt="image" src="https://github.com/user-attachments/assets/a421be92-637c-4f60-96b1-ddf6e2afa9a4" />

