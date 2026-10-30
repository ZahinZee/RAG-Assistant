import pytest

from app.chunking import chunk_text


def test_splits_text_into_chunks_of_chunk_size() -> None:
    chunks = chunk_text("abcdefghij", chunk_size=5, chunk_overlap=0)

    assert chunks == ["abcde", "fghij"]


def test_consecutive_chunks_share_overlap() -> None:
    chunks = chunk_text("abcdefghij", chunk_size=4, chunk_overlap=2)

    assert chunks == ["abcd", "cdef", "efgh", "ghij"]
    for current, following in zip(chunks, chunks[1:]):
        assert current[-2:] == following[:2]


def test_final_chunk_can_be_shorter() -> None:
    chunks = chunk_text("abcdefgh", chunk_size=3, chunk_overlap=0)

    assert chunks == ["abc", "def", "gh"]


def test_empty_text_gives_no_chunks() -> None:
    assert chunk_text("", chunk_size=5, chunk_overlap=1) == []


def test_whitespace_is_preserved() -> None:
    text = "  first line\n\n\tsecond line  "

    chunks = chunk_text(text, chunk_size=6, chunk_overlap=0)

    assert "".join(chunks) == text


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "message"),
    [
        (0, 0, "chunk_size must be greater than zero."),
        (-1, 0, "chunk_size must be greater than zero."),
        (5, -1, "chunk_overlap must not be negative."),
        (5, 5, "chunk_overlap must be smaller than chunk_size."),
        (5, 6, "chunk_overlap must be smaller than chunk_size."),
    ],
)
def test_rejects_invalid_parameters(
    chunk_size: int, chunk_overlap: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        chunk_text("some text", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
