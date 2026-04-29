from __future__ import annotations

try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover
    FastAPI = None

try:
    import uvicorn
except ImportError:  # pragma: no cover
    uvicorn = None

from src.endpoints import router


def create_app() -> FastAPI:
    if FastAPI is None or router is None:
        raise RuntimeError("fastapi is not installed")

    app = FastAPI(title="puml_gen")
    app.include_router(router)
    return app


app = create_app() if FastAPI is not None and router is not None else None


def main() -> int:
    if uvicorn is None:
        raise RuntimeError("uvicorn is not installed")
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
