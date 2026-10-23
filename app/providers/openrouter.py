
from typing import Any

import httpx


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

    async def _request(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:

        if not self.api_key:
            raise LLMProviderError(
                "No se configuró OPENROUTER_API_KEY"
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            async with httpx.AsyncClient(
                timeout=30.0
            ) as client:

                response = await client.post(
                    (
                        f"{self.base_url}"
                        "/chat/completions"
                    ),
                    headers={
                        "Authorization": (
                            f"Bearer {self.api_key}"
                        ),
                        "Content-Type": (
                            "application/json"
                        ),
                    },
                    json=payload,
                )

                response.raise_for_status()

                data = response.json()

        except httpx.TimeoutException as exc:
            raise LLMProviderError(
                "Tiempo de espera agotado"
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                "OpenRouter respondió HTTP "
                f"{exc.response.status_code}"
            ) from exc

        except (
            httpx.RequestError,
            ValueError,
        ) as exc:
            raise LLMProviderError(
                "Error de comunicación "
                "con el proveedor"
            ) from exc

        try:
            message = data["choices"][0]["message"]

            if not isinstance(message, dict):
                raise TypeError(
                    "Respuesta inválida"
                )

            return message

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            raise LLMProviderError(
                "Formato de respuesta LLM inválido"
            ) from exc

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        top_p: float = 1.0,
        max_tokens: int = 300,
    ) -> str:

        message = await self._request(
            messages,
            temperature,
            top_p,
            max_tokens,
        )

        content = message.get("content")

        if (
            not isinstance(content, str)
            or not content.strip()
        ):
            raise LLMProviderError(
                "Respuesta de texto vacía"
            )

        return content

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.3,
        top_p: float = 1.0,
        max_tokens: int = 300,
    ) -> dict[str, Any]:

        return await self._request(
            messages,
            temperature,
            top_p,
            max_tokens,
            tools=tools,
        )
