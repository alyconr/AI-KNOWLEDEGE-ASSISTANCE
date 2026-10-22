
from typing import Protocol


class LLMProvider(Protocol):

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        top_p: float = 1.0,
        max_tokens: int = 300,
    ) -> str:
        ... # estos puntos suspensivos indican que la implementación de este método debe ser proporcionada por cualquier clase que implemente este protocolo
