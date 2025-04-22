import streamlit as st
import os
from dotenv import load_dotenv
from data_processor import DataProcessor
from vector_db import PineconeDB, FAISSDB
from rag import RAGSystem

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

def initialize_system():
    """Initialize the RAG system with the appropriate components."""
    # Initialize data processor
    data_processor = DataProcessor(
        repo_url=os.getenv("GITHUB_REPO_URL"),
        branch=os.getenv("GITHUB_BRANCH")
    )

    # Get and process documents
    documents = data_processor.get_documents()
    chunked_documents = data_processor.chunk_documents(documents)

    # Prepare documents for vector DB
    db_documents = [
        {
            "id": f"{doc.source}_{doc.metadata.get('chunk_index', 0)}",
            "content": doc.content,
            "metadata": doc.metadata
        }
        for doc in chunked_documents
    ]

    # Initialize vector DB
    vector_db_type = os.getenv("VECTOR_DB_TYPE", "faiss")
    if vector_db_type == "pinecone":
        vector_db = PineconeDB(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT"),
            index_name=os.getenv("PINECONE_INDEX_NAME")
        )
    else:
        vector_db = FAISSDB()

    # Add documents to vector DB
    vector_db.add_documents(db_documents)

    # Initialize RAG system
    rag_system = RAGSystem(
        vector_db=vector_db,
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        api_key=os.getenv("OPENAI_API_KEY")
    )

    return rag_system

def main():
    st.set_page_config(
        page_title="Blog Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    st.title("Blog Content Chatbot 🤖")
    st.markdown("""
    Ask questions about the blog content and get answers with citations.
    The chatbot uses RAG (Retrieval Augmented Generation) to provide accurate and relevant answers.
    """)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = initialize_system()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                st.markdown("**Citations:**")
                for citation in message["citations"]:
                    st.markdown(f"- {citation['title']} ({citation['source']})")

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get answer from RAG system
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.rag_system.query(prompt)
                st.markdown(answer.content)
                
                if answer.citations:
                    st.markdown("**Citations:**")
                    for citation in answer.citations:
                        st.markdown(f"- {citation['title']} ({citation['source']})")

        # Add assistant message to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer.content,
            "citations": answer.citations
        })

if __name__ == "__main__":
    main() 