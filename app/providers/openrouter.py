
import httpx

from app.providers.base import LLMProvider


class LLMProviderError(Exception):
    pass


class OpenRouterProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        top_p: float = 1.0,
        max_tokens: int = 300,
    ) -> str:

        if not self.api_key:
            raise LLMProviderError(
                "No se configuró OPENROUTER_API_KEY"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False, # Deshabilitar el streaming para obtener la respuesta completa de una vez
        }

        try:
            async with httpx.AsyncClient(
                timeout=30.0
            ) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )

                response.raise_for_status()
                data = response.json()

        except httpx.TimeoutException as exc:
            raise LLMProviderError(
                "Tiempo de espera agotado"
            ) from exc

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise LLMProviderError(
                f"OpenRouter respondió HTTP {status}"
            ) from exc

        except (httpx.RequestError, ValueError) as exc:
            raise LLMProviderError(
                "Error de comunicación con el proveedor"
            ) from exc

        try:
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Respuesta vacía")
            return content

        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMProviderError(
                "Formato de respuesta LLM inválido"
            ) from exc
