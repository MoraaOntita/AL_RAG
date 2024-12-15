import os
import streamlit as st
from dotenv import load_dotenv
from data_loader import load_and_split_document
from pinecone_indexer import store_embeddings_from_file
from rag_chatbot import create_qa_chain

# Load environment variables
load_dotenv()

# Directories and paths
document_path = "/home/moraa-ontita/Documents/GenAI/AL_RAG_CHATBOT/data/alkhemy_brands_overview.docx"
artifacts_dir = "/home/moraa-ontita/Documents/GenAI/AL_RAG_CHATBOT/artifacts"

# Streamlit App
def main():
    st.set_page_config(page_title="Alkhemy Brands RAG Chatbot", layout="wide")
    st.title("Alkhemy Brands Chatbot")
    st.write("Ask any questions about Alkhemy Brands Limited!")


    # Chatbot QA Chain
    st.header("Ask Questions")
    query = st.text_input("Enter your question here:")
    if query:
        try:
            qa_chain = create_qa_chain()
            response = qa_chain.run(query)
            st.write("**Answer:**")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
