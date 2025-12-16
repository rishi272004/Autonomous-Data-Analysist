# backend/query_engine/memory.py
import hashlib

class ConversationMemory:
    def __init__(self, max_history: int = 5):
        self.history = []
        self.max_history = max_history
        self.cache = {}

    def _key(self, query: str, table_name: str, file_hash: str) -> str:
        raw = f"{query}|{table_name}|{file_hash}"
        return hashlib.md5(raw.encode()).hexdigest()

    def add(self, query: str, result: dict, table_name: str = "", file_hash: str = ""):
        self.history.append({"query": query, "result": result, "table": table_name, "file_hash": file_hash})
        if len(self.history) > self.max_history:
            self.history.pop(0)
        key = self._key(query or "", table_name or "", file_hash or "")
        self.cache[key] = result

    def get_cached(self, query: str, table_name: str = "", file_hash: str = ""):
        key = self._key(query or "", table_name or "", file_hash or "")
        return self.cache.get(key)

    def get_context(self, n: int = 2) -> str:
        recent = self.history[-n:]
        return "\n".join([f"Q: {h['query']} | R: {len(h['result'])} items" for h in recent])

    def get_last_result(self):
        return self.history[-1]["result"] if self.history else None

memory = ConversationMemory()