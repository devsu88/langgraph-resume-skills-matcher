def _strip_markdown_json(text: str) -> str:
    """Rimuove wrapper markdown (```json ... ```) dall'output LLM per ottenere JSON valido."""
    if not text:
        return text
    s = text.strip()
    if s.startswith("```"):
        first = s.find("\n")
        if first != -1:
            s = s[first + 1 :]
        if s.endswith("```"):
            s = s[: s.rfind("```")].rstrip()
    return s