import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

def setup_pinecone(index_name="alkhemy-index", dimension=384):
    """Initialize Pinecone and create index if it doesn't exist."""
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")
    return pc

def store_embeddings_from_file(artifacts_dir, index_name="alkhemy-index"):
    """Read chunks from file, embed them, and store in Pinecone."""
    # Load chunks and metadata
    with open(os.path.join(artifacts_dir, "chunks_and_metadata.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Initialize embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Setup Pinecone
    setup_pinecone(index_name)
    embeddings = [model.encode(chunk["text"]) for chunk in chunks]
    ids = [str(chunk["id"]) for chunk in chunks]

    # Upsert embeddings into Pinecone
    vectors = [{"id": id_, "values": embedding, "metadata": {"text": chunk["text"]}} 
               for id_, embedding, chunk in zip(ids, embeddings, chunks)]
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    index.upsert(vectors=vectors)
    print("Embeddings successfully stored in Pinecone.")

