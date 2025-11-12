from rag_pipeline.chunking import split_text_fixed_size


def test_empty_text():
    assert split_text_fixed_size("", chunk_size=100) == []


def test_small_text_no_overlap():
    t = "Hello world. This is a test."
    chunks = split_text_fixed_size(t, chunk_size=100, overlap=0)
    assert len(chunks) == 1
    assert "Hello world" in chunks[0]


def test_overlap_and_multiple_chunks():
    t = "".join([f"Sentence {i}. " for i in range(50)])
    chunks = split_text_fixed_size(t, chunk_size=50, overlap=10)
    # expect multiple chunks
    assert len(chunks) >= 2
