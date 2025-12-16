from groq import Groq
import logging
import os
import re
import dotenv 

logger = logging.getLogger(__name__)

# Initialize Groq client using your API key
GROQ_API_KEY = "GROQ_API_KEY"

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

def query_gemini(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    """
    Query the Groq API with a text prompt and return the response text.
    Now uses Groq instead of Gemini for higher rate limits.
    """
    try:
        # Send the prompt to Groq
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048
        )
        
        # Extract the text output
        answer = response.choices[0].message.content.strip()
        
        logger.info("Groq query successful")
        return answer

    except Exception as e:
        msg = str(e)
        # Friendly handling for rate limits / quota exceeded (HTTP 429)
        if "rate" in msg.lower() or "quota" in msg.lower() or "429" in msg:
            wait_match = re.search(r"retry in ([0-9]+)", msg.lower())
            wait_hint = f" Please retry in ~{wait_match.group(1)}s." if wait_match else " Please retry after a short wait."
            friendly = (
                "LLM rate limit exceeded." +
                wait_hint +
                " Consider reducing concurrent requests or waiting for reset."
            )
            logger.warning(friendly + f" Raw error: {msg}")
            return friendly
        logger.error(f"Groq API failed: {e}")
        return f"LLM Error: {e}"
