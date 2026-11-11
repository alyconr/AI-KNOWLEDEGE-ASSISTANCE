
import pytest

from app.services.assistant_service import (
    AssistantService,
)


class FakeProvider:

    async def generate(
        self,
        messages,
        temperature=0.3,
        top_p=1,
        max_tokens=300,
    ):
        assert (
            messages[1]["content"]
            == "¿Qué es FastAPI?"
        )

        return (
            "Es un framework de APIs en Python."
        )


@pytest.mark.asyncio
async def test_returns_answer():

    service = AssistantService(
        FakeProvider()
    )

    response = await service.ask(
        "¿Qué es FastAPI?"
    )

    assert (
        "framework"
        in response.lower()
    )
