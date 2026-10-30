from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI(title="RAG-Assistant")

SUPPORTED_EXTENSIONS = {".txt"}


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the application is running."""
    return {"status": "ok"}


@app.post("/documents/extract")
async def extract_document(file: UploadFile = File(...)) -> dict[str, str]:
    """Return the text of an uploaded `.txt` file along with its filename.

    The file is read in memory and not stored. Responds 400 when the extension
    is not `.txt`, the content is not valid UTF-8, or the file is empty or
    contains only whitespace, since there is nothing to ingest.
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

    return {"filename": filename, "text": text}
