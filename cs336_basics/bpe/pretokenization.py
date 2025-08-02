import os
import mmap
from typing import BinaryIO, List, Tuple, Iterable
import regex as re
from tqdm import tqdm
from collections import Counter


PAT = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_BYTES = re.compile(
    rb"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    special_tokens_bytes: List[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(special_tokens_bytes, list), (
        "Must represent special token as a bytestring"
    )
    if any(not isinstance(t, (bytes, bytearray)) for t in special_tokens_bytes):
        raise TypeError("special_tokens_bytes must contain bytes")

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
    
    if not special_tokens_bytes:
        return sorted(set(chunk_boundaries))

    max_tok_len = len(special_tokens_bytes)

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess

        tail = b""
        absolute_pos = initial_position
        boundary_set = False
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            window = tail + mini_chunk
            window_start = absolute_pos - len(tail)

            best_abs = None
            for tok in special_tokens_bytes: # Make sure all special tokens are not divided
                idx = window.find(tok)
                if idx != -1:
                    abs_idx = window_start + idx
                    if best_abs is None or abs_idx < best_abs:
                        best_abs = abs_idx

            if best_abs is not None:
                chunk_boundaries[bi] = best_abs
                boundary_set = True
                break

            absolute_pos += len(mini_chunk)
            if max_tok_len > 1:
                tail = window[-(max_tok_len - 1):]
            else:
                tail = b""

        if not boundary_set and chunk_boundaries[bi] != file_size:
            chunk_boundaries[bi] = file_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

## Usage
# pretokenize_file("./data/TinyStoriesV2-GPT4-valid.txt",["<|endoftext|>"], 8)
def pretokenize_file(
    input_path: str, 
    special_tokens: List[str],
    num_processes: int = 1
) -> Tuple[Iterable[List[bytes]], Counter]:
    """
    Pre-tokenize a file by finding boundaries for special tokens.
    Returns an iterable of (start, end) byte offsets for each chunk.
    """
    special_tokens_bytes: List[bytes] = [s.encode("utf-8") for s in special_tokens]
    special_re = re.compile(b"|".join(re.escape(t) for t in special_tokens_bytes)) if special_tokens_bytes else None
    word_freq: Counter = Counter()

    def emit_normal_segment(seg: bytes, toks_out: List[bytes]):
        for m in PAT_BYTES.finditer(seg):
            w = m.group()
            toks_out.append(w)
            word_freq[w] += 1

    def gen():
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, special_tokens_bytes)
            for start, end in tqdm(list(zip(boundaries[:-1], boundaries[1:])),desc="Pre-tokenizing file", unit="chunk"):
                f.seek(start)
                chunk = f.read(end - start)
                toks: List[bytes] = []

                if not chunk:
                    yield toks
                    continue

                if special_re is None:
                    emit_normal_segment(chunk, toks)
                    yield toks
                    continue

                pos = 0
                for m in special_re.finditer(chunk):
                    s, e = m.start(), m.end()
                    if s > pos:
                        emit_normal_segment(chunk[pos:s], toks)
                    w = m.group(0)
                    toks.append(w)
                    word_freq[w] += 1
                    pos = e
                if pos < len(chunk):
                    emit_normal_segment(chunk[pos:], toks)

                yield toks

    return gen(), word_freq