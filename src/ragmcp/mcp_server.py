import chromadb
from sentence_transformers import SentenceTransformer
import needle
import os
from fastmcp import FastMCP

mcp = FastMCP("MyServer")

# Initialize Ephemeral Vector DB
chroma_client = chromadb.EphemeralClient()
collection = chroma_client.create_collection(name="mcp_kb", metadata={"hnsw:space": "cosine"})

# Initialize the Tiny Agent and Embedding Model
agent = needle.Needle(tools=[])
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to add documents to the vector database
@mcp.tool()
def add_documents_to_db(documents):
    for doc in documents:
        # Embed the document
        embedding = embedding_model.encode(doc['text']).tolist()
        
        # Add to ChromaDB
        collection.add(
            documents=[doc['text']],
            metadatas=[doc['metadata']],
            ids=[doc['id']]
        )

# Function to query the vector database
@mcp.tool()
def query_database(query_text):
    # Embed the query
    query_embedding = embedding_model.encode(query_text).tolist()
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    return results


@mcp.tool() 
def health_check():
    return {'status': 'healthy'}

if __name__ == "__main__":
    # Example documents
    documents = [
        {
            "id": "1",
            "text": "This is a sample document about RAG systems.",
            "metadata": {"source": "example"}
        },
        {
            "id": "2",
            "text": "Another example document about AI and machine learning.",
            "metadata": {"source": "example"}
        }
    ]

    # Add documents to the database
    add_documents_to_db(documents)

    # Query the database
    query_results = query_database("RAG systems")
    print(query_results)

    mcp.run(transport="http", host="127.0.0.1", port=9000)