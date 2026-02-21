# Assignment 3: Secure RAG System

This project implements a production-ready RAG (Retrieval-Augmented Generation) system for Nova Scotia driving rules, incorporating security layers, prompt injection defenses, and quality evaluation.

## 1. Prompt Injection Defenses
I implemented the following three defenses to protect the system:
- **System Prompt Hardening**: The system prompt explicitly instructs the LLM to only answer based on the provided driving manual and to never reveal its internal instructions.
- **Instruction-Data Separation**: All retrieved context is wrapped in XML-style `<retrieved_context>` tags. This helps the LLM distinguish between developer instructions and external, potentially untrusted document data.
- **Input Sanitization**: User queries are scanned for common injection patterns and keywords (e.g., "ignore previous instructions", "you are now a...", "print your system prompt"). If detected, the query is blocked before reaching the LLM.

## 2. Evaluation Metric
I chose **Faithfulness** as the primary evaluation metric.
- **Why**: In a RAG system for legal/safety rules (like driving laws), hallucinations are dangerous. The Faithfulness metric uses a separate LLM evaluation to ensure that every claim in the response is directly supported by the retrieved context. It measures whether the system is "grounded" in the documentation.

## 3. Interesting Findings
- **Graceful Refusal**: The system effectively refused to answer off-topic queries (e.g., "chocolate cake recipe") even when the retriever returned chunks that might have had high similarity scores but unrelated content. The LLM responded with "I don't know" as instructed, leading to a "No" in the faithfulness score for those instances.
- **Injection Resilience**: The combination of keyword blocking and instruction-data separation successfully neutralized both direct ("ignore instructions") and indirect ("tell me a joke") prompt injection attempts.
- **PII Privacy**: The regex-based PII detection successfully identified and redacted phone numbers and license plates before processing, protecting user privacy during the retrieval and generation phases.

## Setup & Execution
1. **Sync dependencies**: `uv sync`
2. **Environment**: Add your `OPENAI_API_KEY` to the `.env` file. (Supports OpenRouter keys starting with `sk-or-v1`).
3. **Run Tests**: `uv run main.py`
4. **View Results**: All detailed logs and scores are saved in `output/results.txt`.
