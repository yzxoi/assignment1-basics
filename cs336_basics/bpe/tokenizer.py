from typing import Dict, List, Tuple, Optional, Iterable, Iterator
import json
import regex as re


class BPETokenizer:
    """
    Byte-level BPE tokenizer class.
    """
    def __init__(self, special_tokens: List[str]):
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.byte2id: Dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        for token in special_tokens:
            tok_bytes = token.encode('utf-8')
            tid = len(self.vocab)
            self.vocab[tid] = tok_bytes
            self.byte2id[tok_bytes] = tid
        self.merges: Dict[Tuple[int, int], int] = {}

    def add_merge(self, pair: Tuple[int, int]) -> int:
        """
        Add a new merge token to the vocabulary, given a pair of symbol IDs.
        Returns the new token ID.
        """
        id_a, id_b = pair
        new_bytes = self.vocab[id_a] + self.vocab[id_b]
        new_id = len(self.vocab)
        self.vocab[new_id] = new_bytes
        self.byte2id[new_bytes] = new_id
        self.merges[pair] = new_id
        return new_id

class Tokenizer:
    """
    Byte-level tokenizer class.
    """
    def __init__(self, vocab: dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] | None = None):
        """
        vocab: dict[int, bytes]        # id -> token bytes
        merges: list of (bytes, bytes) # in learned order
        special_tokens: list of string tokens, e.g. ["<|endoftext|>"]
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens: List[str] = special_tokens or []
        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        self.byte2id: Dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        self.merge_ranks: Dict[Tuple[int, int], int] = {}
        for idx, (ba, bb) in enumerate(merges):
            if ba not in self.byte2id or bb not in self.byte2id:
                continue
            ida = self.byte2id[ba]
            idb = self.byte2id[bb]
            merged_bytes = ba + bb
            if merged_bytes not in self.byte2id:
                raise ValueError(f"Merged bytes {merged_bytes} from {(ba, bb)} not in vocab")
            self.merge_ranks[(ida, idb)] = idx

        # Pre-tokenization pattern: special tokens OR the byte-level regex
        self.PAT_BYTES = re.compile(
            rb"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        )

    def pretokenize(self, text: str) -> Iterator[Tuple[bool, str]]:
        """
        Pre-tokenize a string.
        """
        btext = text.encode("utf-8")
        N = len(btext)
        i = 0
        special_bytes_list = [tok.encode("utf-8") for tok in self.special_tokens]

        def _yield_normal_span(b: bytes, start: int, end: int):
            """Yield normal pieces within [start, end) using PAT_BYTES, without crossing 'end'."""
            j = start
            while j < end:
                m = self.PAT_BYTES.match(b, j)
                if m is None:
                    piece = b[j:j+1]
                    j += 1
                else:
                    m_end = m.end()
                    if m_end > end:
                        piece = b[j:end]
                        j = end
                    else:
                        piece = m.group(0)
                        j = m_end
                yield False, piece.decode("utf-8")

        while i < N:
            best_pos = None
            best_tok = None
            for tok in special_bytes_list:
                pos = btext.find(tok, i)
                if pos != -1:
                    if (best_pos is None) or (pos < best_pos) or (pos == best_pos and len(tok) > len(best_tok)):
                        best_pos, best_tok = pos, tok

            if best_pos is None:
                yield from _yield_normal_span(btext, i, N)
                break

            if best_pos > i:
                yield from _yield_normal_span(btext, i, best_pos)

            yield True, best_tok.decode("utf-8")
            i = best_pos + len(best_tok)

    def _encode_bytes(self, substring: str) -> List[int]:
        """
        Given a non-special substring (pretoken), perform byte-level initialization
        and greedy BPE merges by smallest rank first within this pretoken only.
        """
        data = substring.encode("utf-8")
        if data in self.byte2id:
            seq = [self.byte2id[data]]
        else:
            seq = [self.byte2id[bytes([b])] for b in data]

        while True:
            best_idx = None
            best_rank = None
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_idx = i
            if best_idx is None:
                break
            pair = (seq[best_idx], seq[best_idx + 1])
            byte_a, byte_b = self.vocab[pair[0]], self.vocab[pair[1]]
            merged_id = self.byte2id[byte_a + byte_b]
            seq[best_idx: best_idx + 2] = [merged_id]
        return seq

    def encode(self, text: str) -> List[int]:
        """
        Full encode pipeline:
          - pretokenize with special tokens atomic (longest-first)
          - apply merges within each pretoken independently
        """
        output_ids: List[int] = []
        for is_special, piece in self.pretokenize(text):
            print(f"Processing piece: {piece} (special={is_special})")
            if is_special:
                if piece in self.special_tokens:
                    output_ids.append(self.byte2id[piece.encode('utf-8')])
                else:
                    output_ids.extend(self._encode_bytes(piece))
            else:
                output_ids.extend(self._encode_bytes(piece))
        return output_ids
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of
        strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-eﬀicient tokenization of large files that we cannot directly load into
        memory.
        """
        for text in iterable:
            yield from self.encode(text)
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        output_bytes = b"".join(self.vocab[t] for t in ids)
        return output_bytes.decode('utf-8', errors='ignore')

def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
    """
    Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
    (in the same format that your BPE training code output) and (optionally) a list of special
    tokens. This method should accept the following additional parameters:
    vocab_filepath: str
    merges_filepath: str
    special_tokens: list[str] | None = None
    """
    with open(vocab_filepath, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    with open(merges_filepath, 'r', encoding='utf-8') as f:
        merges = [tuple(line.strip().split()) for line in f]
    tokenizer = cls(special_tokens)
    tokenizer.vocab = {int(k): v.encode('utf-8') for k, v in vocab.items()}
    tokenizer.byte2id = {v: k for k, v in tokenizer.vocab.items()}
    for a, b in merges:
        a_id = tokenizer.byte2id[a.encode('utf-8')]
        b_id = tokenizer.byte2id[b.encode('utf-8')]
        tokenizer.merges[(a_id, b_id)] = len(tokenizer.vocab)
        tokenizer.vocab[len(tokenizer.vocab)] = a.encode('utf-8') + b.encode('utf-8')
    return tokenizer