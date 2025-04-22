# Personal Blog Chatbot

A chatbot that helps search and answer questions about my blog content using RAG (Retrieval Augmented Generation) and LLMs.

## Features

- Search and answer questions about my blog content
- Citations for all answers
- Modular architecture for easy switching of components
- Support for multiple LLM providers
- Clean and intuitive UI

## Architecture

The system consists of several key components:

1. **Data Processing Pipeline**
   - GitHub content cloning
   - Markdown parsing
   - Document chunking
   - Embedding generation

2. **Vector Database**
   - Pinecone integration
   - Alternative: FAISS for local storage

3. **RAG Implementation**
   - Document retrieval
   - Context management
   - Citation tracking

4. **LLM Integration**
   - Multiple provider support
   - Conversation management
   - Response generation

5. **UI Layer**
   - Streamlit/Gradio interface
   - User interaction handling

## Setup

1. Clone the repository
2. Ensure atleast Python 3.11 is used
3. Install dependencies: 
   - `pip install -r requirements.txt`
4. Configure environment variables
5. Test the application: `python src/test_implementation.py`
6. Run the application: `streamlit run src/app.py`
   - To circumvent versioning issue if exists, `$(which python) -m streamlit run src/app.py`

## Configuration

Create a `.env` file with the following variables:

```env
GITHUB_REPO_URL=https://github.com/prasanth-ntu/prasanth.io
GITHUB_BRANCH=v4
VECTOR_DB_TYPE=pinecone  # or faiss
LLM_PROVIDER=openai  # or other providers
```

## Usage

1. Start the application
2. Enter your questions in the chat interface
3. Receive answers with citations

## Development

The project is structured to be modular and extensible. Key components can be swapped out by implementing the appropriate interfaces.

## To do's
- Find open source alternative for OpenAI as it's not cost-effectoive

## License
MIT License
