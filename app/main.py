from fastapi import FastAPI

app = FastAPI(title="RAG-Assistant")


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the application is running."""
    return {"status": "ok"}
