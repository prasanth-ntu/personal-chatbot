from typing import List, Dict, Any
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.globals import set_verbose

# Set verbose mode if needed
set_verbose(False)

@dataclass
class Answer:
    content: str
    citations: List[Dict[str, Any]]

class RAGSystem:
    def __init__(self, vector_db, llm_provider="openai", api_key=None):
        self.vector_db = vector_db
        self.llm = self._initialize_llm(llm_provider, api_key)
        self.prompt_template = self._create_prompt_template()

    def _initialize_llm(self, provider: str, api_key: str):
        if provider == "openai":
            return ChatOpenAI(
                model="gpt-4.1-nano", #"gpt-4-turbo-preview",
                temperature=0,
                api_key=api_key,
                max_tokens=2000,
                model_kwargs={"response_format": {"type": "text"}}
            )
        # Add support for other LLM providers here
        raise ValueError(f"Unsupported LLM provider: {provider}")

    def _create_prompt_template(self):
        return ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context from technical documentation.
            Your responses should be:
            1. Accurate and based only on the provided context
            2. Well-structured and easy to understand
            3. Include specific citations from the source material
            4. Technical but accessible
            5. If you cannot find the answer in the context, say "I don't have enough information to answer that question."
            
            Format your response with:
            - A clear answer to the question
            - Supporting details from the context
            - Citations to the specific sources used
            """),
            ("human", """Context: {context}
            
            Question: {question}
            
            Please provide a detailed answer based on the context above. Include specific citations from the sources."""),
        ])

    def _format_context(self, search_results: List[Dict[str, Any]]) -> str:
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"Source {i} ({result.metadata['title']}):\n{result.content}\n")
        return "\n".join(context_parts)

    def _extract_citations(self, answer: str, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        citations = []
        for result in search_results:
            if result.content in answer:
                citations.append({
                    "source": result.metadata["source"],
                    "title": result.metadata["title"],
                    "score": result.score
                })
        return citations

    def query(self, question: str, k: int = 5) -> Answer:
        # Retrieve relevant documents
        search_results = self.vector_db.search(question, k)
        
        if not search_results:
            return Answer(
                content="I don't have enough information to answer that question.",
                citations=[]
            )

        # Format context for the LLM
        context = self._format_context(search_results)

        # Create the chain
        chain = (
            {"context": lambda _: context, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

        # Generate answer
        answer_text = chain.invoke(question)

        # Extract citations
        citations = self._extract_citations(answer_text, search_results)

        return Answer(
            content=answer_text,
            citations=citations
        ) 