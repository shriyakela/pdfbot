import argparse
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings  # Corrected import for embeddings
from dotenv import load_dotenv
import os
import shutil
from typing import List

CHROMA_PATH = "chroma"
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

def main():
    print("Starting main function...")
    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    print("Finished main function.")

def load_documents() -> List[Document]:
    directory_path = r"C:\Users\Shriya.Kela\genai\pdf"
    print(f"Checking files in directory: {directory_path}")
    files = os.listdir(directory_path)
    print(f"Files in directory: {files}")

    loader = DirectoryLoader(directory_path, glob="*.pdf")
    documents = loader.load()

    print("Documents loaded successfully.")
    return documents

# def split_documents(documents: List[Document]):
#     print("Splitting documents into chunks...")
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=400,
#         chunk_overlap=30,
#         length_function=len,
#         is_separator_regex=False,
#     )
#     chunks = text_splitter.split_documents(documents)
#     print(f"Number of chunks created: {len(chunks)}")
#     return chunks
def split_documents(documents: List[Document]):
    print("Splitting documents into chunks...")
    # Using SemanticChunker for better splitting
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    chunks = text_splitter.create_documents([doc.page_content for doc in documents])
    print(f"Number of chunks created: {len(chunks)}")
    return chunks

def add_to_chroma(chunks: List[Document]):
    print("Adding chunks to Chroma database...")
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks: List[Document]):
    print("Calculating chunk IDs...")
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    print("Clearing Chroma database...")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    print("Database cleared.")

def get_embedding_function():
    print("Initializing embedding function...")
    return OpenAIEmbeddings()

if __name__ == "__main__":
    main()
