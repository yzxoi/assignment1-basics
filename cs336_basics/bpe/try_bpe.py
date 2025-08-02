from train_bpe import train_bpe
from typing import Dict, List, Tuple
import json

def save_vocab_json(vocab: Dict[int, bytes], path: str):
    """
    Save `vocab` (id -> bytes) as a JSON file mapping id -> string.
    We decode each token’s bytes with backslash‐escaping so that
    non‐UTF8 or control bytes survive round‐trip.
    """
    serializable = {
        str(idx): token_bytes.decode('utf-8', errors='backslashreplace')
        for idx, token_bytes in vocab.items()
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

def save_merges_txt(merges: List[Tuple[bytes, bytes]], path: str):
    """
    Save your merges as lines of two tokens separated by a space.
    We decode with backslashreplace for safety.
    """
    with open(path, 'w', encoding='utf-8') as f:
        for a, b in merges:
            a_str = a.decode('utf-8', errors='backslashreplace')
            b_str = b.decode('utf-8', errors='backslashreplace')
            f.write(f"{a_str} {b_str}\n")

def main():
    input_str = "./data/owt_train.txt"
    vocab, merges = train_bpe(
        input_path=input_str,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
        num_processes=10000,
    )
    save_vocab_json(vocab, "./trained_vocab.json")
    save_merges_txt(merges, "./trained_merges.txt")
    print("Wrote ./trained_vocab.json and ./trained_merges.txt")

if __name__ == "__main__":
	main()