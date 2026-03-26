import ollama
from typing import List, Dict, Any, AsyncIterator

from config import get_settings

settings = get_settings()

class OllamaService:
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or settings.llm.model
        host = base_url or settings.llm.base_url
        self.client = ollama.AsyncClient(host=host)

    async def invoke(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Invoke the Ollama model with a list of messages.
        """
        response = await self.client.chat(model=self.model, messages=messages, tools=tools)
        return response

    async def stream(self, messages: List[Dict[str, Any]]) -> AsyncIterator[str]:
        """
        Stream the response from the Ollama model.
        """
        async for part in self.client.chat(model=self.model, messages=messages, stream=True):
            yield part['message']['content']
