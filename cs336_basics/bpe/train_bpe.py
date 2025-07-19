import heapq
import time
from typing import List, Tuple, Dict, Set, Union
from pretokenization import pretokenize_file
from tokenizer import BPETokenizer
from tqdm import tqdm
from data_structures import DoublyLinkedList, Node
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
    timings["tokenizer_init"] = time.perf_counter() - t

    t = time.perf_counter()
    docs: List[List[bytes]] = []
    for tok_bytes in pretokenize_file(input_path, special_tokens, num_processes):
        docs.append(list(tok_bytes))
    timings["pretokenize"] = time.perf_counter() - t
    print(f"[BPE] Loaded {len(docs):,} documents")

    t = time.perf_counter()
    pair_cnt : Dict[Tuple[bytes, bytes], int]  = defaultdict(int)
    pair_pos : Dict[Tuple[bytes, bytes], Set[Pos]] = defaultdict(set)

    for d, seq in enumerate(docs):
        for i, (a, b) in enumerate(zip(seq, seq[1:])):
            pair = (a, b)
            pair_cnt[pair] += 1
            pair_pos[pair].add((d, i))
    timings["initial_count"] = time.perf_counter() - t

    heap = [(-c, p) for p, c in pair_cnt.items()]
    heapq.heapify(heap)

    merges: List[Tuple[bytes, bytes]] = []
    pbar = tqdm(total=vocab_size - len(tokenizer.vocab), desc="Training BPE")

    t_merge = time.perf_counter()
    while len(tokenizer.vocab) < vocab_size and heap:
        neg_freq, pair = heapq.heappop(heap)
        freq = -neg_freq
        if freq == 0 or pair_cnt[pair] != freq:
            continue

        a, b = pair
        merged_token = a + b
        tokenizer.add_merge(pair)
        merges.append(pair)
        pbar.update(1)

        positions = list(pair_pos[pair])
        del pair_pos[pair]
        pair_cnt[pair] = 0

        for d, idx in positions:
            seq = docs[d]
            if idx >= len(seq)-1 or seq[idx] != a or seq[idx+1] != b:
                continue

            if idx > 0:
                left_pair = (seq[idx-1], a)
                pair_cnt[left_pair] -= 1
                pair_pos[left_pair].discard((d, idx-1))
                heapq.heappush(heap, (-pair_cnt[left_pair], left_pair))

            if idx+2 < len(seq):
                right_pair = (b, seq[idx+2])
                pair_cnt[right_pair] -= 1
                pair_pos[right_pair].discard((d, idx+1))
                heapq.heappush(heap, (-pair_cnt[right_pair], right_pair))

            seq[idx:idx+2] = [merged_token]

            if idx > 0:
                new_left = (seq[idx-1], merged_token)
                pair_cnt[new_left] += 1
                pair_pos[new_left].add((d, idx-1))
                heapq.heappush(heap, (-pair_cnt[new_left], new_left))

            if idx+1 < len(seq):
                new_right = (merged_token, seq[idx+1])
                pair_cnt[new_right] += 1
                pair_pos[new_right].add((d, idx))
                heapq.heappush(heap, (-pair_cnt[new_right], new_right))

    timings["merge_loop"] = time.perf_counter() - t_merge
    timings["total"] = time.perf_counter() - t0

    print("\n[BPE] Timing (seconds)")
    for k, v in timings.items():
        print(f"  {k:<14}: {v:.4f}")

    return tokenizer.vocab, merges