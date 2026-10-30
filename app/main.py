from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.chunking import chunk_text

app = FastAPI(title="RAG-Assistant")

SUPPORTED_EXTENSIONS = {".txt"}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the application is running."""
    return {"status": "ok"}


async def read_text_upload(file: UploadFile) -> str:
    """Return the decoded text of an uploaded `.txt` file.

    Raises 400 when the extension is not `.txt`, the content is not valid
    UTF-8, or the file is empty or contains only whitespace, since there is
    nothing to ingest.
    """
    filename = file.filename or ""
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    content = await file.read()
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty.")

    return text


@app.post("/documents/extract")
async def extract_document(file: UploadFile = File(...)) -> dict[str, str]:
    """Return the text of an uploaded `.txt` file along with its filename.

    The file is read in memory and not stored. See `read_text_upload` for the
    400 responses.
    """
    text = await read_text_upload(file)
    return {"filename": file.filename, "text": text}


@app.post("/documents/ingest")
async def ingest_document(file: UploadFile = File(...)) -> dict[str, Any]:
    """Split an uploaded `.txt` file into overlapping chunks.

    Uses chunks of 1000 characters with 200 characters of overlap. The file is
    read in memory and not stored. See `read_text_upload` for the 400 responses.
    """
    text = await read_text_upload(file)
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return {"filename": file.filename, "chunk_count": len(chunks), "chunks": chunks}
