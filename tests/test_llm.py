# tests/test_llm.py
def test_ollama_client_import():
    from backend.llm.ollama_client import OllamaClient
    c = OllamaClient()
    assert hasattr(c, "generate")
