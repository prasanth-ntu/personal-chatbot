from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pinecone
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

@dataclass
class SearchResult:
    content: str
    metadata: Dict[str, Any]
    score: float

class VectorDB(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector database."""
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Search for similar documents."""
        pass

class PineconeDB(VectorDB):
    def __init__(self, api_key: str, environment: str, index_name: str):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)
        
        # Create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=384,  # Dimension of all-MiniLM-L6-v2 embeddings
                metric="cosine"
            )
        
        self.index = pinecone.Index(index_name)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        vectors = []
        for doc in documents:
            embedding = self.embedder.encode(doc['content']).tolist()
            vectors.append({
                'id': doc['id'],
                'values': embedding,
                'metadata': {**doc['metadata'], 'content': doc['content']}
            })
        
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        query_embedding = self.embedder.encode(query).tolist()
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        
        return [
            SearchResult(
                content=result.metadata['content'],
                metadata=result.metadata,
                score=result.score
            )
            for result in results.matches
        ]

class FAISSDB(VectorDB):
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.documents = []

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            return

        self.documents = documents
        embeddings = np.array([self.embedder.encode(doc['content']) for doc in documents])
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        if not self.index:
            return []

        query_embedding = self.embedder.encode(query)
        distances, indices = self.index.search(
            np.array([query_embedding]), k
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append(SearchResult(
                    content=doc['content'],
                    metadata=doc['metadata'],
                    score=float(1 / (1 + distances[0][i]))  # Convert distance to similarity score
                ))
        
        return results 