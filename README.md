# Assignment 3: Secure RAG System

This project implements a secure RAG (Retrieval-Augmented Generation) system for Nova Scotia driving rules, focusing on guardrails, prompt injection defenses, and evaluation.

## Defenses Implemented
1. **System Prompt Hardening**: Strict instructions to the LLM to only answer based on driving rules and never deviate.
2. **Instruction-Data Separation**: Using specialized delimiters (`<retrieved_context>`) to help the LLM distinguish between control instructions and untrusted data.
3. **Input Sanitization**: Detection and blocking of known injection patterns and keywords.

## Evaluation Metric
- **Faithfulness**: The system uses a separate LLM call to verify if the generated answer is strictly supported by the retrieved context.

## Setup
1. Install dependencies: `uv sync`
2. Configure API Key: Add `OPENAI_API_KEY` to `.env`. This system supports both standard OpenAI keys and OpenRouter (`sk-or-v1...`) keys automatically.
3. Run tests: `uv run main.py`
4. View results: `output/results.txt`
