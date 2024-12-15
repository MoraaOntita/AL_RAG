import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain.callbacks import StreamlitCallbackHandler
from pinecone import Pinecone as PineconeClient
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Setup Groq API Wrapper for LLaMA 3
def load_llama3_model():
    """Load the LLaMA 3 model using Groq API."""
    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model="llama3-70b-8192", api_key=api_key)
    return llm

# Load Pinecone Retriever
def load_pinecone_retriever(index_name="alkhemy-index"):
    """Load embeddings from Pinecone and create retriever."""
    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = LangchainPinecone(
        index=index, 
        embedding=embeddings,
        text_key="text"
    )
    
    return vectorstore.as_retriever()


# Define the RAG Chain
def create_qa_chain():
    """Create the Retrieval-Augmented Generation QA chain."""
    llm = load_llama3_model()
    retriever = load_pinecone_retriever()
    
    # Ensure retriever is explicitly called
    retriever_obj = retriever  # This should now be a valid retriever object
    
    prompt_template = """
    Use the following context to answer the question at the end.
    
    Context: {context}
    
    Question: {question}
    
    Answer in a clear and concise manner.
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever_obj, 
        chain_type="stuff", 
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Streamlit App
def main():
    st.set_page_config(page_title="Alkhemy Brands RAG Chatbot", layout="wide")
    st.title("Alkhemy Brands Chatbot")
    st.write("Ask any questions about Alkhemy Brands Limited!")
    
    # Load QA Chain
    qa_chain = create_qa_chain()

    # User Input Box
    with st.container():
        query = st.text_input("Enter your question here:", key="user_query")
        if query:
            st_callback = StreamlitCallbackHandler(st.container())
            with st.spinner("Generating response..."):
                response = qa_chain.run(query, callbacks=[st_callback])
            st.write("**Answer:**")
            st.write(response)

if __name__ == "__main__":
    main()
