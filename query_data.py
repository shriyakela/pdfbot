import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import openai
import os
from extract import get_embedding_function  # Importing the embedding function from extract.py
from openai import OpenAI



load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai.api_key)
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    print("Starting main function...")
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    print(f"Query Text: {query_text}")
    query_rag(query_text)
    print("Finished main function.")

def query_rag(query_text: str):
    print("Querying RAG...")
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=1)
    print(f"Number of results: {len(results)}")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(f"Generated Prompt: {prompt}")

    # Call OpenAI API
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or your preferred OpenAI model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    response_text = response.choices[0].message.content

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()
