import heapq
import time
from typing import List, Tuple, Dict, Set, Union
from pretokenization import pretokenize_file
from tokenizer import BPETokenizer
from tqdm import tqdm
from data_structures import DoublyLinkedList, Node


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
    
    init_start = time.perf_counter()
    tokenizer = BPETokenizer(special_tokens=special_tokens)
    timings['tokenizer_init'] = time.perf_counter() - init_start
    print("Initialized tokenizer with special tokens:", special_tokens)

    pre_start = time.perf_counter()
    docs: List[DoublyLinkedList] = []
    for tok_bytes in pretokenize_file(str(input_path), special_tokens, num_processes):
        print(tok_bytes[:10])
        dll = DoublyLinkedList()
        for b in tok_bytes:
            dll.append(b)
        docs.append(dll)
    timings['pretokenization_and_build'] = time.perf_counter() - pre_start
    print(f"Pre-tokenization complete, {len(docs)} documents processed.")

    count_start = time.perf_counter()
    pair_counts: Dict[Tuple[bytes,bytes], int] = {}
    pair_positions: Dict[Tuple[bytes,bytes], Set[Node]] = {}
    for dll in docs:
        node = dll.head
        while node and node.next:
            pair = (node.value, node.next.value)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            pair_positions.setdefault(pair, set()).add(node)
            node = node.next
    timings['initial_count'] = time.perf_counter() - count_start
    print(f"Initial pair counts computed, {len(pair_counts)} pairs found.")

    heap_start = time.perf_counter()
    heap: List[Tuple[int, Tuple[bytes,bytes]]] = [(-cnt, pair) for pair, cnt in pair_counts.items()]
    heapq.heapify(heap)
    timings['heap_build'] = time.perf_counter() - heap_start
    print(f"Heap built with {len(heap)} pairs.")

    merge_start = time.perf_counter()
    merges: List[Tuple[bytes, bytes]] = []

    pbar = tqdm(total=vocab_size - len(tokenizer.vocab), desc="Training BPE", unit="merge")
    while len(tokenizer.vocab) < vocab_size and heap:
        negcnt, pair = heapq.heappop(heap)
        cnt = -negcnt
        if pair_counts.get(pair, 0) != cnt or cnt == 0:
            continue
        pbar.update(1)

        a, b = pair
        # print(a, b, "pair:", pair, "count:", cnt)
        
        merges.append(pair)
        tokenizer.add_merge(pair)

        nodes = list(pair_positions[pair])
        del pair_positions[pair]
        pair_counts[pair] = 0

        for node in nodes:
            if not node.next or node.value != a or node.next.value != b:
                continue

            left = node.prev
            right = node.next.next

            if left:
                old = (left.value, a)
                if pair_counts.get(old, 0) == 0:
                    continue
                pair_counts[old] -= 1
                pair_positions[old].discard(left)
                heapq.heappush(heap, (-pair_counts[old], old))
            if right:
                old = (b, right.value)
                if pair_counts.get(old, 0) == 0:
                    continue
                pair_counts[old] -= 1
                pair_positions[old].discard(node.next)
                heapq.heappush(heap, (-pair_counts[old], old))

            to_rm = node.next
            node.value = a + b
            node.next = to_rm.next
            if to_rm.next:
                to_rm.next.prev = node

            if left:
                newp = (left.value, a + b)
                pair_counts[newp] = pair_counts.get(newp, 0) + 1
                pair_positions.setdefault(newp, set()).add(left)
                heapq.heappush(heap, (-pair_counts[newp], newp))
            if right:
                newq = (a + b, right.value)
                pair_counts[newq] = pair_counts.get(newq, 0) + 1
                pair_positions.setdefault(newq, set()).add(node)
                heapq.heappush(heap, (-pair_counts[newq], newq))
    timings['merge_loop'] = time.perf_counter() - merge_start
    print(f"Completed merging, {len(merges)} merges created.")

    timings['total_time'] = time.perf_counter() - t0

    print("BPE Training timings (seconds):")
    for step, duration in timings.items():
        print(f"  {step}: {duration:.4f}")

    return tokenizer.vocab, merges