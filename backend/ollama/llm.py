# backend/ollama/llm.py
from ollama import chat
import logging

logging.basicConfig(level=logging.INFO)

def ask_ollama(prompt: str, model: str = "llama3.1:8b-instruct-q3_K_S") -> str:
    """
    Send a prompt to Ollama LLM and return the response text.

    Args:
        prompt (str): The user prompt or query.
        model (str): The Ollama model to use (default "llama3.1:8b-instruct-q3_K_S").

    Returns:
        str: LLM response text.
    """
    try:
        response = chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.get("content", "")
        logging.info("Ollama responded successfully.")
        return answer
    except Exception as e:
        logging.error(f"Ollama query failed: {e}")
        return f"Error: {e}"
