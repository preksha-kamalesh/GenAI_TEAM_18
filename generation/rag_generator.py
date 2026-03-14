"""
RAG Generator Module — LLM-based Answer Generation
Navya's Module | genai-team-18

Takes retrieved documents and a question, generates an answer using an LLM.
Supports multiple LLM backends: GPT, Llama 3, Mistral, or local models.

Pipeline: Question + Retrieved Docs → LLM → Generated Answer
"""

import os
from typing import Optional, List
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate text from a prompt."""
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI GPT backend (GPT-3.5/GPT-4 via API)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name ('gpt-3.5-turbo', 'gpt-4', etc.)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY env var not set")

        try:
            import openai
            openai.api_key = self.api_key
            self.client = openai
        except ImportError:
            raise ImportError("Install 'openai' package: pip install openai")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate using OpenAI API."""
        import openai
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response["choices"][0]["message"]["content"].strip()


class HuggingFaceBackend(LLMBackend):
    """HuggingFace local LLM backend (Llama 3, Mistral, etc.)."""

    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1", device: str = "cuda"):
        """
        Args:
            model_name: HuggingFace model ID
            device: 'cuda' or 'cpu'
        """
        self.model_name = model_name
        self.device = device

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate using local HuggingFace model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the output
        return response[len(prompt):].strip()


class RAGGenerator:
    """High-level RAG answer generator."""

    def __init__(
        self,
        llm_backend: Optional[LLMBackend] = None,
        backend_type: str = "huggingface",
        **backend_kwargs
    ):
        """
        Initialize RAG Generator.

        Args:
            llm_backend: Pre-initialized LLM backend. If None, creates one based on backend_type.
            backend_type: 'openai', 'huggingface', or 'ollama'
            **backend_kwargs: Additional args for the backend (api_key, model_name, etc.)
        """
        if llm_backend:
            self.backend = llm_backend
        elif backend_type == "openai":
            self.backend = OpenAIBackend(**backend_kwargs)
        elif backend_type == "huggingface":
            self.backend = HuggingFaceBackend(**backend_kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend_type}")

    def generate_answer(
        self,
        question: str,
        retrieved_docs: List[dict],
        top_k: int = 5,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> dict:
        """Generate an answer based on question and retrieved documents.

        Args:
            question: User's question
            retrieved_docs: List of dicts with 'text' and 'source' keys
            top_k: Use only top-k documents (if more provided)
            max_tokens: Max tokens in response
            temperature: LLM temperature (0.0 = deterministic, 1.0 = random)

        Returns:
            Dict with keys: 'question', 'answer', 'context', 'num_docs_used'
        """
        # Prepare context from documents
        context = self._build_context(retrieved_docs[:top_k])

        # Build RAG prompt
        prompt = self._build_rag_prompt(question, context)

        # Generate answer
        answer = self.backend.generate(prompt, max_tokens=max_tokens, temperature=temperature)

        return {
            "question": question,
            "answer": answer,
            "context": context,
            "num_docs_used": len(retrieved_docs[:top_k]),
            "documents": retrieved_docs[:top_k],
        }

    def _build_context(self, docs: List[dict]) -> str:
        """Format retrieved documents into context string."""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.get("source", "Unknown")
            text = doc.get("text", "")
            context_parts.append(f"[Document {i}] ({source})\n{text}")
        return "\n\n".join(context_parts)

    def _build_rag_prompt(self, question: str, context: str) -> str:
        """Build the complete RAG prompt."""
        prompt = f"""You are a helpful assistant providing accurate answers based on provided context.

Context (Retrieved Documents):
{context}

Question:
{question}

Instructions:
- Answer ONLY based on the provided context.
- If the context doesn't contain enough information to answer, say "The provided documents do not contain sufficient information."
- Be concise but thorough.
- Include relevant source references when applicable.
- Do NOT make up information or hallucinate facts not in the documents.

Answer:
"""
        return prompt

    def __repr__(self) -> str:
        return f"RAGGenerator(backend={type(self.backend).__name__})"


# Example usage
if __name__ == "__main__":
    # Example 1: Using HuggingFace (local) backend
    print("Initializing RAG Generator with HuggingFace backend...")
    try:
        generator = RAGGenerator(
            backend_type="huggingface",
            model_name="mistralai/Mistral-7B-Instruct-v0.1",
            device="cuda"  # or "cpu"
        )

        # Sample retrieved documents
        sample_docs = [
            {
                "text": "Diabetes mellitus is a group of metabolic diseases characterized by high blood sugar levels.",
                "source": "medical_db"
            },
            {
                "text": "Type 1 diabetes must be managed with insulin injections.",
                "source": "medical_db"
            },
        ]

        # Generate answer
        result = generator.generate_answer(
            question="What is diabetes and how is it managed?",
            retrieved_docs=sample_docs,
            max_tokens=256,
            temperature=0.5
        )

        print(f"\n❓ Question: {result['question']}")
        print(f"✅ Answer: {result['answer']}")
        print(f"📚 Documents used: {result['num_docs_used']}")

    except ImportError as e:
        print(f"Note: {e}")
        print("Install required packages: pip install transformers torch")
