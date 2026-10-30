import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_extract_returns_text_and_filename() -> None:
    response = client.post(
        "/documents/extract",
        files={"file": ("notes.txt", b"Retrieval-Augmented Generation", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "notes.txt",
        "text": "Retrieval-Augmented Generation",
    }


def test_extract_rejects_unsupported_extension() -> None:
    response = client.post(
        "/documents/extract",
        files={"file": ("report.pdf", b"%PDF-1.7", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Only .txt files are supported."}


def test_extract_rejects_empty_file() -> None:
    response = client.post(
        "/documents/extract",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "File is empty."}


def test_extract_rejects_invalid_utf8() -> None:
    response = client.post(
        "/documents/extract",
        files={"file": ("latin1.txt", b"caf\xe9", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "File must be UTF-8 encoded text."}


def test_ingest_returns_single_chunk_for_short_text() -> None:
    response = client.post(
        "/documents/ingest",
        files={"file": ("notes.txt", b"Retrieval-Augmented Generation", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "notes.txt",
        "chunk_count": 1,
        "chunks": ["Retrieval-Augmented Generation"],
    }


def test_ingest_splits_long_text_into_overlapping_chunks() -> None:
    text = "".join(str(i % 10) for i in range(2500))

    response = client.post(
        "/documents/ingest",
        files={"file": ("long.txt", text.encode(), "text/plain")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "long.txt"
    assert body["chunk_count"] == 3
    assert body["chunks"] == [text[0:1000], text[800:1800], text[1600:2500]]


@pytest.mark.parametrize(
    ("filename", "content", "detail"),
    [
        ("report.pdf", b"%PDF-1.7", "Only .txt files are supported."),
        ("latin1.txt", b"caf\xe9", "File must be UTF-8 encoded text."),
        ("empty.txt", b"", "File is empty."),
        ("blank.txt", b"  \n\t ", "File is empty."),
    ],
)
def test_ingest_rejects_invalid_files(
    filename: str, content: bytes, detail: str
) -> None:
    response = client.post(
        "/documents/ingest",
        files={"file": (filename, content, "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": detail}
