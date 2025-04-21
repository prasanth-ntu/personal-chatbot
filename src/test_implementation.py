import os
from dotenv import load_dotenv
from data_processor import DataProcessor
from vector_db import FAISSDB, PineconeDB
from rag import RAGSystem

def test_data_processor():
    print("Testing Data Processor...")
    processor = DataProcessor(
        repo_url=os.getenv("GITHUB_REPO_URL"),
        branch=os.getenv("GITHUB_BRANCH"),
        subfolder="Knowledge/Tech"
    )
    
    # Test document retrieval
    documents = processor.get_documents()
    print(f"Retrieved {len(documents)} documents from Tech subfolder")
    
    # Test document chunking
    chunked_documents = processor.chunk_documents(documents)
    print(f"Created {len(chunked_documents)} chunks")
    
    return chunked_documents

def test_vector_db(documents, use_pinecone=False):
    print("\nTesting Vector Database...")
    
    if use_pinecone:
        print("Using Pinecone...")
        vector_db = PineconeDB(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT"),
            index_name=os.getenv("PINECONE_INDEX_NAME")
        )
    else:
        print("Using FAISS...")
        vector_db = FAISSDB()
    
    # Prepare documents for vector DB
    db_documents = [
        {
            "id": f"{doc.source}_{doc.metadata.get('chunk_index', 0)}",
            "content": doc.content,
            "metadata": doc.metadata
        }
        for doc in documents
    ]
    
    # Test adding documents
    vector_db.add_documents(db_documents)
    print("Documents added to vector database")
    
    # Test search with advanced technical queries
    test_queries = [
        "What are the key differences between BERT and RoBERTa?",
        "Explain the architecture of transformer models in detail",
        "How does self-attention mechanism work in transformer models?"
    ]
    
    for query in test_queries:
        print(f"\nSearch results for '{query}':")
        results = vector_db.search(query)
        for result in results:
            print(f"- Score: {result.score:.2f}, Source: {result.metadata['source']}")
    
    return vector_db

def test_rag_system(vector_db):
    print("\nTesting RAG System with GPT-4...")
    rag_system = RAGSystem(
        vector_db=vector_db,
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Test query with advanced technical questions
    test_questions = [
        "Compare and contrast the architectures of BERT and GPT models. What are their key differences and use cases?",
        # "Explain in detail how the attention mechanism works in transformer models, including the mathematical operations involved.",
        # "What are the advantages and limitations of using transformer models for natural language processing tasks?",
        # "How does transfer learning work in the context of transformer models, and what are its benefits?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        answer = rag_system.query(question)
        print(f"Answer: {answer.content}")
        if answer.citations:
            print("Citations:")
            for citation in answer.citations:
                print(f"- {citation['title']} ({citation['source']})")

def main():
    # Load environment variables
    load_dotenv()
    
    # Test data processor
    documents = test_data_processor()
    
    # Test vector database (use FAISS by default)
    use_pinecone = os.getenv("VECTOR_DB_TYPE", "faiss").lower() == "pinecone"
    vector_db = test_vector_db(documents, use_pinecone)
    
    # Test RAG system
    test_rag_system(vector_db)

if __name__ == "__main__":
    main() 