import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Any, Dict

class Evaluator:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = "https://openrouter.ai/api/v1" if api_key and api_key.startswith("sk-or-v1") else None
        model_prefix = "openai/" if api_key and api_key.startswith("sk-or-v1") else ""
        self.llm = ChatOpenAI(
            model=f"{model_prefix}gpt-4o-mini", 
            temperature=0,
            base_url=base_url
        )

    def check_faithfulness(self, query: str, response: str, context: str) -> str:
        """Checks if the response is supported by the context."""
        prompt = ChatPromptTemplate.from_template(
            "You are an evaluator for a RAG system. "
            "Given the following context and an answer, determine if the answer is fully supported by the context. "
            "Reply only with 'Yes' or 'No', followed by a brief one-sentence reason.\n\n"
            "Context: {context}\n\n"
            "Question: {query}\n\n"
            "Answer: {response}\n\n"
            "Is the answer supported? [Yes/No]"
        )
        
        chain = prompt | self.llm
        result = chain.invoke({"context": context, "query": query, "response": response})
        return result.content

    def evaluate(self, query: str, response: str, docs: List[Any]) -> Dict[str, Any]:
        context = "\n".join([d.page_content for d in docs])
        score = self.check_faithfulness(query, response, context)
        return {"faithfulness": score}
