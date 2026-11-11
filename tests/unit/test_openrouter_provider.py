
import json

import httpx
import pytest

from app.providers.openrouter import (
    OpenRouterProvider,
)


@pytest.mark.asyncio
async def test_openrouter_sends_tools_and_reads_tool_calls(
    monkeypatch,
):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(
            request.content
        )

        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {
                                        "name": (
                                            "lookup_ai_concept"
                                        ),
                                        "arguments": (
                                            '{"concept":"RAG"}'
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    original = httpx.AsyncClient

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(
            transport=httpx.MockTransport(
                handler
            ),
            **kwargs,
        ),
    )

    provider = OpenRouterProvider(
        "test-key",
        "openrouter/free",
        "https://example.test/api/v1",
    )

    message = await provider.complete_with_tools(
        messages=[
            {
                "role": "user",
                "content": "¿Qué es RAG?",
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_ai_concept",
                    "description": "consulta",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "concept": {
                                "type": "string"
                            }
                        },
                    },
                },
            }
        ],
    )

    assert (
        message["tool_calls"][0]["function"]["name"]
        == "lookup_ai_concept"
    )

    assert (
        seen["body"]["tool_choice"]
        == "auto"
    )

    assert (
        seen["body"]["tools"][0]["type"]
        == "function"
    )


@pytest.mark.asyncio
async def test_generate_preserves_existing_contract(
    monkeypatch,
):
    original = httpx.AsyncClient

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "content": (
                                        "respuesta del modelo"
                                    )
                                }
                            }
                        ]
                    },
                )
            ),
            **kwargs,
        ),
    )

    provider = OpenRouterProvider(
        "test-key",
        "openrouter/free",
        "https://example.test/api/v1",
    )

    answer = await provider.generate([
        {
            "role": "user",
            "content": "Hola",
        }
    ])

    assert answer == "respuesta del modelo"
