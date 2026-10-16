
import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import get_settings
from app.providers.openrouter import (
    LLMProviderError,
    OpenRouterProvider,
)
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.assistant_service import AssistantService


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Chat"])


def get_assistant_service() -> AssistantService:
    settings = get_settings()

    provider = OpenRouterProvider(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model,
        base_url=settings.openrouter_base_url,
    )

    return AssistantService(provider)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: AssistantService = Depends(
        get_assistant_service
    ),
):
    try:
        answer = await service.ask(
            question=request.question,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

    except LLMProviderError as exc:
        logger.error("Fallo del proveedor LLM: %s", exc, exc_info=exc.__cause__)
        raise HTTPException(
            status_code=502,
            detail=f"No fue posible obtener una respuesta del LLM: {exc}",
        )

    return ChatResponse(
        answer=answer,
        model=get_settings().openrouter_model,
    )
