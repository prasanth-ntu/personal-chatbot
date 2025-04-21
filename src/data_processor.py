import os
from typing import List, Dict, Optional
import git
from pathlib import Path
import markdown
from bs4 import BeautifulSoup
from dataclasses import dataclass

@dataclass
class Document:
    content: str
    metadata: Dict
    source: str

class DataProcessor:
    def __init__(self, repo_url: str, branch: str, subfolder: Optional[str] = None, local_path: str = "data"):
        self.repo_url = repo_url
        self.branch = branch
        self.subfolder = subfolder
        self.local_path = local_path
        self.repo = None

    def clone_repo(self) -> None:
        """Clone the GitHub repository if not already cloned."""
        if not os.path.exists(self.local_path):
            self.repo = git.Repo.clone_from(self.repo_url, self.local_path, branch=self.branch)
        else:
            self.repo = git.Repo(self.local_path)
            self.repo.remotes.origin.pull()

    def process_markdown(self, file_path: str) -> str:
        """Process a markdown file and extract text content."""
        with open(file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML
        html = markdown.markdown(md_content)
        
        # Extract text from HTML
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()

    def get_documents(self) -> List[Document]:
        """Get all documents from the repository or specified subfolder."""
        if not self.repo:
            self.clone_repo()

        documents = []
        content_path = os.path.join(self.local_path, "content")
        
        # If subfolder is specified, append it to the content path
        if self.subfolder:
            content_path = os.path.join(content_path, self.subfolder)
        
        for root, _, files in os.walk(content_path):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.local_path)
                    
                    content = self.process_markdown(file_path)
                    metadata = {
                        "source": relative_path,
                        "title": os.path.splitext(file)[0],
                        "category": self.subfolder if self.subfolder else "general"
                    }
                    
                    documents.append(Document(
                        content=content,
                        metadata=metadata,
                        source=relative_path
                    ))
        
        return documents

    def chunk_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Split documents into chunks for better processing."""
        chunked_documents = []
        
        for doc in documents:
            words = doc.content.split()
            for i in range(0, len(words), chunk_size - chunk_overlap):
                chunk = ' '.join(words[i:i + chunk_size])
                chunked_documents.append(Document(
                    content=chunk,
                    metadata={**doc.metadata, "chunk_index": i // (chunk_size - chunk_overlap)},
                    source=doc.source
                ))
        
        return chunked_documents 