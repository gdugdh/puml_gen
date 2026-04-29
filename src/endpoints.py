from __future__ import annotations

from src.api_models import GeneratePumlRequest, GeneratePumlResponse
from src.generator import generate_from_request

try:
    from fastapi import APIRouter
except ImportError:  # pragma: no cover
    APIRouter = None


if APIRouter is not None:
    router = APIRouter()

    @router.post("/generate", response_model=GeneratePumlResponse)
    def generate_puml_docs(request: GeneratePumlRequest) -> GeneratePumlResponse:
        return generate_from_request(request)
else:  # pragma: no cover
    router = None
