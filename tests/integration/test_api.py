
import pytest

from httpx import (
    ASGITransport,
    AsyncClient,
)

from app.main import app

from app.api.agent import (
    get_agent_service,
)

from app.services.agent_service import (
    AgentService,
)


class FakeProvider:

    async def complete_with_tools(
        self,
        messages,
        tools,
        temperature=0.3,
        top_p=1,
        max_tokens=300,
    ):
        return {
            "content": "Respuesta educativa"
        }


@pytest.mark.asyncio
async def test_agent_endpoint():

    app.dependency_overrides[
        get_agent_service
    ] = lambda: AgentService(
        FakeProvider()
    )

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:

            response = await client.post(
                "/api/v1/agent/chat",
                json={
                    "question": "¿Qué es RAG?"
                },
            )

        assert response.status_code == 200

        assert (
            response.json()["answer"]
            == "Respuesta educativa"
        )

        assert (
            response.json()["tools_used"]
            == []
        )

    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_agent_rejects_invalid():

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:

        response = await client.post(
            "/api/v1/agent/chat",
            json={
                "question": "x",
                "max_steps": 6,
            },
        )

    assert response.status_code == 422
