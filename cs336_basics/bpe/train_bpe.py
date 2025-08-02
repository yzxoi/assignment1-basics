import heapq
import time
from typing import List, Tuple, Dict, Set, Union
from .pretokenization import pretokenize_file
from .tokenizer import BPETokenizer
from tqdm import tqdm
from collections import defaultdict

Pos = Tuple[int, int]                 # (doc_id, index)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    num_processes: int = 1
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Args:
        input_path: Path to training text file.
        vocab_size: Maximum vocabulary size (including initial bytes and special tokens).
        special_tokens: List of strings to include verbatim in the vocabulary.
        num_processes: Number of parallel processes for pretokenization.

    Returns:
        vocab: Mapping from token ID to token bytes.
        merges: Ordered list of byte-pair merges.
    """ 
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    t = time.perf_counter()
    tokenizer = BPETokenizer(special_tokens)
    special_ids = {
        tokenizer.byte2id[token.encode("utf-8")]
        for token in special_tokens
    }
    timings["tokenizer_init"] = time.perf_counter() - t

    t = time.perf_counter()
    chunks, word_freq = pretokenize_file(
        input_path, special_tokens, num_processes
    )
    for _ in chunks:
        pass
    total_types = len(word_freq)
    print(f"Total types: {total_types}")
    # print(word_freq)
    timings["pretokenize"] = time.perf_counter() - t
    print(f"[BPE] Found {total_types:,} unique pre-tokens")

    t = time.perf_counter()
    type_byteids: Dict[bytes, List[int]] = {}
    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)

    for word_bytes, freq in word_freq.items():
        if word_bytes in tokenizer.byte2id:
            byte_ids = [tokenizer.byte2id[word_bytes]]
        else:
            byte_ids = [tokenizer.byte2id[bytes([b])] for b in word_bytes]
        if len(byte_ids) == 1 and byte_ids[0] in special_ids:
            type_byteids[word_bytes] = byte_ids
            continue
        type_byteids[word_bytes] = byte_ids
        for i in range(len(byte_ids) - 1):
            pair = (byte_ids[i], byte_ids[i+1])
            if pair[0] in special_ids or pair[1] in special_ids:
                continue
            pair_counts[pair] += freq
    timings["initial_count"] = time.perf_counter() - t

    heap = [(-cnt, pair) for pair, cnt in pair_counts.items()]
    heapq.heapify(heap)

    merges: List[Tuple[bytes, bytes]] = []
    t = time.perf_counter()
    while len(tokenizer.vocab) < vocab_size and heap:
        negcnt, pair = heapq.heappop(heap)
        freq = -negcnt
        # Skip stale or zero-count entries
        if pair_counts.get(pair, 0) != freq or freq == 0:
            continue
        
        tied = [pair]
        while heap and heap[0][0] == negcnt:
            _, p2 = heapq.heappop(heap)
            if pair_counts.get(p2, 0) == freq:
                tied.append(p2)
        tied.sort(key=lambda pr: (tokenizer.vocab[pr[0]], tokenizer.vocab[pr[1]]), reverse=True)

        selected = tied[0]
        for p_other in tied[1:]:
            heapq.heappush(heap, (-pair_counts[p_other], p_other))

        pair = selected
        if pair[0] in special_ids or pair[1] in special_ids:
            pair_counts[pair] = 0
            continue

        id_a, id_b = pair
        # Record merge as byte pair
        merges.append((tokenizer.vocab[id_a], tokenizer.vocab[id_b]))
        new_id = tokenizer.add_merge(pair)

        pair_counts[pair] = 0

        for word_bytes, byte_ids in list(type_byteids.items()):
            freq = word_freq[word_bytes]
            i = 0
            while i < len(byte_ids) - 1:
                if byte_ids[i] == id_a and byte_ids[i+1] == id_b:
                    if i > 0:
                        left = (byte_ids[i-1], id_a)
                        pair_counts[left] -= freq
                        heapq.heappush(heap, (-pair_counts[left], left))
                    if i+2 < len(byte_ids):
                        right = (id_b, byte_ids[i+2])
                        pair_counts[right] -= freq
                        heapq.heappush(heap, (-pair_counts[right], right))
                    byte_ids[i : i+2] = [new_id]
                    if i > 0:
                        new_left = (byte_ids[i-1], new_id)
                        pair_counts[new_left] += freq
                        heapq.heappush(heap, (-pair_counts[new_left], new_left))
                    if i+1 < len(byte_ids):
                        new_right = (new_id, byte_ids[i+1])
                        pair_counts[new_right] += freq
                        heapq.heappush(heap, (-pair_counts[new_right], new_right))
                    # i += 1
                i += 1
        type_byteids = {w: ids for w, ids in type_byteids.items() if len(ids) > 1}

    timings["merge_loop"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - t0

    print("\n[BPE] Timing (seconds)")
    for k, v in timings.items():
        print(f"  {k:<14}: {v:.4f}")

    print(merges[:10])
    return tokenizer.vocab, merges