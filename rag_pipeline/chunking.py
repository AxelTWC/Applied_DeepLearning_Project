import re
from typing import List, Tuple
from tqdm.auto import tqdm

class Chunk:
    
    def __init__(self):
        pass
    
    def chunk(self, text:str, start_pos: int, chunk_size: int = 500, overlap: int = 50) -> Tuple[int, List[str]]:
        pass
        
    def apply(self, text: str) -> List[str]:
        pass
    
class FixedSizeChunk(Chunk):
    
    def __init__(self):
        pass
    
    def chunk(self, text: str, start_pos: int, chunk_size: int = 500, overlap: int = 50) -> Tuple[int, str]:
        """
        Split text into fixed-size chunks with optional overlap.
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
            text (str): Input document string
            start_pos (int): Chunking start position
            chunk_size (int): Chunk size
            overlap (int): Characters of overlap between consecutive chunks
        
        Returns:
            Tuple of next start position and current chunk text
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        
        text = text.strip()
        if not text:
            return start_pos, ""
        text_len = len(text)
        
        # Use simple rfind on the substring to locate the last sentence-ending
        # punctuation followed by a space. This is fast and avoids regex overhead.
        SENT_ENDS = [". ", "! ", "? "]
        end = min(start_pos + chunk_size, text_len)
        # Try to find a sentence boundary before 'end' to make nicer chunks
        sub = text[start_pos:end]
        last_pos = -1
        for se in SENT_ENDS:
            pos = sub.rfind(se)
            if pos > last_pos:
                last_pos = pos

        if last_pos != -1:
            # split after the sentence-ending token (include the following space)
            split_pos = start_pos + last_pos + len(se)
            # only accept the split if the chunk won't be tiny
            if split_pos - start_pos >= chunk_size // 3:
                end = split_pos
        
        # last chunk
        if end >= text_len:
            return text_len, text[start_pos:].strip()
        
        chunk = text[start_pos:end].strip()
        
        next_start = end - overlap
        if next_start <= start_pos:
            next_start = start_pos + 1

        next_start = max(0, next_start)
        return next_start, chunk
        
        
    
    def apply(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
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

        with tqdm(total=text_len, desc="Chunking text", unit="char") as progress:
            while start < text_len:
                next_start, chunk = self.chunk(text, start, chunk_size, overlap)
                chunks.append(chunk)
                progress.update(next_start - start)
                start = next_start

        return chunks


if __name__ == "__main__":
    sample = """
Title: Small Sample Document
This is a short sample document used to demonstrate the Retrieval-Augmented Generation (RAG)
baseline. The goal is to split this text into chunks, index them, and then retrieve relevant
pieces in response to a user query.

The quick brown fox jumps over the lazy dog. This sentence contains every letter in the English
alphabet and is often used as example text. The RAG baseline will not be perfect but will
illustrate the flow: chunk -> embed -> index -> retrieve -> generate.

End of sample document.
    """
    chunk = FixedSizeChunk()
    res = chunk.apply(sample, chunk_size=300, overlap=20)
    print(len(res))
    for r in res:
        print(r)
        print("-"*100)