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
    timings, t0 = {}, time.perf_counter()
    tok = BPETokenizer(special_tokens)

    docs: List[List[int]] = []
    byte2id: Dict[bytes, int] = {v: k for k, v in tok.vocab.items()}

    for doc in pretokenize_file(input_path, special_tokens, num_processes):
        seq: List[int] = []
        for tok_bytes in doc:
            if tok_bytes in byte2id:
                seq.append(byte2id[tok_bytes])
            else:
                seq.extend(tok_bytes)
        if seq:
            docs.append(seq)

    pair_cnt: Dict[Tuple[int,int], int]           = defaultdict(int)
    pair_pos: Dict[Tuple[int,int], Set[Pos]]      = defaultdict(set)
    for d, seq in enumerate(docs):
        for i, (a, b) in enumerate(zip(seq, seq[1:])):
            pair_cnt[(a, b)] += 1
            pair_pos[(a, b)].add((d, i))

    heap = [(-c, p) for p, c in pair_cnt.items()]
    heapq.heapify(heap)
    merges: List[Tuple[bytes, bytes]] = []

    pbar = tqdm(total=vocab_size - len(tok.vocab), desc="Training BPE")
    while len(tok.vocab) < vocab_size and heap:
        neg, pair = heapq.heappop(heap)
        if -neg != pair_cnt.get(pair, 0):
            continue
        if -neg == 0:
            break
        pbar.update(1)

        a, b = pair
        a_bytes, b_bytes = tok.vocab[a], tok.vocab[b]
        tok.add_merge((a_bytes, b_bytes))
        new_id = len(tok.vocab) - 1
        new_token = tok.vocab[new_id]
        merges.append((a_bytes, b_bytes))

        positions = pair_pos.pop(pair)
        for d, idx in positions:
            seq = docs[d]
            if idx >= len(seq) - 1 or seq[idx] != a or seq[idx+1] != b:
                continue
            seq[idx:idx+2] = [new_id]

            if idx > 0:
                L = seq[idx-1]
                old = (L, a); new = (L, new_id)
                pair_cnt[old] -= 1
                pair_pos[old].discard((d, idx-1))
                pair_cnt[new] += 1
                pair_pos[new].add((d, idx-1))
                heapq.heappush(heap, (-pair_cnt[new], new))

            if idx < len(seq)-1:
                R = seq[idx+1]
                old = (b, R); new = (new_id, R)
                pair_cnt[old] -= 1
                pair_pos[old].discard((d, idx+1))
                pair_cnt[new] += 1
                pair_pos[new].add((d, idx))
                heapq.heappush(heap, (-pair_cnt[new], new))

        pair_cnt[pair] = 0

    timings["total"] = time.perf_counter() - t0
    return tok.vocab, merges