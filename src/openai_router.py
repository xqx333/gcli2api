"""OpenAI Router - Handles OpenAI-format model listing."""
from fastapi import APIRouter

from config import get_available_models
from .models import ModelList, Model

router = APIRouter()


@router.get("/v1/models", response_model=ModelList)
async def list_models():
    """Return available models using the OpenAI schema."""
    models = get_available_models("openai")
    return ModelList(data=[Model(id=model_id) for model_id in models])
