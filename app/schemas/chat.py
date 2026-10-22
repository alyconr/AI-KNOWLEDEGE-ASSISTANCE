
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=2000,
        description="Pregunta del usuario",
    )

    temperature: float = Field(
        default=0.3,
        ge=0,
        le=1.5,
    )

    top_p: float = Field(
        default=1.0,
        gt=0,
        le=1,
    )

    max_tokens: int = Field(
        default=300,
        ge=1,
        le=2000,
    )


class ChatResponse(BaseModel):
    answer: str
    model: str
