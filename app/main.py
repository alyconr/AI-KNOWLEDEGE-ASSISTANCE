
from fastapi import FastAPI

from app.api.chat import router as chat_router


app = FastAPI(
    title="AI Knowledge Assistant",
    description="API educativa con integración LLM",
    version="1.0.0",
)

app.include_router(chat_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
