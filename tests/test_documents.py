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
