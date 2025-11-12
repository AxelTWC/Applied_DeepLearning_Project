import re
from typing import List


def split_text_fixed_size(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    r"""Split text into fixed-size chunks with optional overlap.

    This baseline uses character-based chunking. It will try to avoid
    splitting in the middle of sentences by searching for a sentence boundary
    (., !, or ?) within the chunk end window, but will fall back to a hard
    split when necessary.

    Implementation notes:
    - Uses simple rfind on the substring to pick the last sentence boundary
      which is faster and avoids building large intermediate lists from
      regex.finditer on long documents.
    - Adds a progress guard so `start` always advances and cannot get stuck.

    Args:
        text: Input document string.
        chunk_size: Maximum characters per chunk.
        overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    # Use simple rfind on the substring to locate the last sentence-ending
    # punctuation followed by a space. This is fast and avoids regex overhead.
    SENT_ENDS = [". ", "! ", "? "]

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to find a sentence boundary before 'end' to make nicer chunks
        sub = text[start:end]
        last_pos = -1
        for se in SENT_ENDS:
            pos = sub.rfind(se)
            if pos > last_pos:
                last_pos = pos

        if last_pos != -1:
            # split after the sentence-ending token (include the following space)
            split_pos = start + last_pos + len(se)
            # only accept the split if the chunk won't be tiny
            if split_pos - start >= chunk_size // 3:
                end = split_pos

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # If we've reached or passed the end, stop immediately
        if end >= text_len:
            break

        # advance by chunk_size - overlap, but ensure progress
        new_start = end - overlap
        if new_start <= start:
            new_start = start + 1
        start = max(0, new_start)

    return chunks


if __name__ == "__main__":
    sample = "Hello world. " * 50
    print(len(split_text_fixed_size(sample, chunk_size=100, overlap=10)))
