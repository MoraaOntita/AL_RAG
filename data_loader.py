import os
import json
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Function to save text to file
def save_to_file(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(data)

# Function to save chunks with metadata
def save_chunks_and_metadata(chunks, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    chunk_data = []

    for i, chunk in enumerate(chunks):
        chunk_data.append({
            "id": i,
            "text": chunk.page_content,
            "metadata": chunk.metadata
        })

    # Save chunks and metadata as JSON
    with open(os.path.join(output_dir, "chunks_and_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, indent=4)

def load_and_split_document(doc_path, artifacts_dir):
    """Load, save, and split the document into chunks."""
    # Load Word document
    loader = Docx2txtLoader(doc_path)
    documents = loader.load()

    # Save full extracted text
    extracted_text = "\n".join([doc.page_content for doc in documents])
    save_to_file(extracted_text, os.path.join(artifacts_dir, "extracted_text.txt"))
    print("Extracted text saved to artifacts folder.")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Save chunks and metadata
    save_chunks_and_metadata(chunks, artifacts_dir)
    print("Chunks and metadata saved to artifacts folder.")

    return chunks
