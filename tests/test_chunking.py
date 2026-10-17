import pytest

from app.rag.chunking import chunk_text


def test_respects_size_and_overlaps():
    text = " ".join(f"Oración número {i} con algunas palabras de relleno." for i in range(40))
    chunks = chunk_text(text, size=40, overlap=10)
    assert len(chunks) > 3
    assert all(len(c.split()) <= 40 for c in chunks)
    # solapamiento: la última oración de un fragmento reaparece al inicio del siguiente
    assert chunks[0].split(".")[-2].strip().split()[-1] in chunks[1]


def test_short_text_single_chunk():
    assert chunk_text("Una sola oración corta.", 100, 10) == ["Una sola oración corta."]


def test_very_long_sentence_is_split():
    text = " ".join(["palabra"] * 500)
    chunks = chunk_text(text, size=100, overlap=10)
    assert len(chunks) == 5 and all(len(c.split()) <= 100 for c in chunks)


def test_empty_and_invalid():
    assert chunk_text("   \n ", 50, 5) == []
    with pytest.raises(ValueError):
        chunk_text("hola", 10, 10)
