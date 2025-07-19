import os
import mmap
from typing import BinaryIO, List, Tuple, Iterable
import regex as re
from multiprocessing import Pool

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

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

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

_MMAP: mmap.mmap

def _init_worker_mmap(input_path: str):
    global _MMAP
    f = open(input_path, "rb")
    _MMAP = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

def _process_bounds(bounds: Tuple[int, int]) -> List[bytes]:
    start, end = bounds
    chunk_bytes = _MMAP[start:end]
    text = chunk_bytes.decode("utf-8", errors="ignore")
    toks = re.findall(PAT, text)
    return [t.encode("utf-8") for t in toks]

## Usage
# pretokenize_file("./data/TinyStoriesV2-GPT4-valid.txt",["<|endoftext|>"], 8)
def pretokenize_file(
    input_path: str, 
    special_tokens: List[str],
    num_processes: int = 1
) -> Iterable[List[bytes]]:
    """
    Pre-tokenize a file by finding boundaries for special tokens.
    Returns an iterable of (start, end) byte offsets for each chunk.
    """
    split_special_token = ''.join(special_tokens).encode('utf-8') if isinstance(special_tokens, list) else special_tokens.encode('utf-8')
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token)
    
    chunks = [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    
    if num_processes > 1:
        with Pool(num_processes, initializer=_init_worker_mmap, initargs=(input_path,)) as pool:
            for tok_bytes in pool.map(_process_bounds, chunks):
                yield tok_bytes
    else:
        f = open(input_path, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for start, end in chunks:
            chunk = mm[start:end].decode("utf-8", errors="ignore")
            toks = re.findall(PAT, chunk)
            yield [t.encode("utf-8") for t in toks]
        mm.close()
        f.close()
        
