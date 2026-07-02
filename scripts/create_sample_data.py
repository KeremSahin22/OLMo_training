"""
Create a small local training sample from C4 for OLMo-1B replication.

Usage:
    python scripts/create_sample_data.py \
        --output /disk/u/kerem.sahin/data/olmo1b_sample.npy \
        --num-docs 10000

The output is a uint16 numpy memmap file compatible with MemMapDataset.
~10k docs from C4 yields ~5-10M tokens (~20MB on disk).
"""

import argparse
import os

import numpy as np
from datasets import load_dataset

from olmo.tokenizer import Tokenizer

TOKENIZER_PATH = os.path.join(
    os.path.dirname(__file__),
    "../olmo_data/tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json",
)
EOS_TOKEN_ID = 50279  # matches OLMo-1B config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Path to output .npy file")
    parser.add_argument("--num-docs", type=int, default=10_000, help="Number of C4 documents to use")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"Loading tokenizer from {TOKENIZER_PATH}")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH, eos_token_id=EOS_TOKEN_ID)

    print(f"Streaming {args.num_docs} documents from allenai/c4 ...")
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

    all_tokens = []
    for i, doc in enumerate(dataset):
        if i >= args.num_docs:
            break
        ids = tokenizer.encode(doc["text"], add_special_tokens=True)
        all_tokens.extend(ids)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{args.num_docs} docs, {len(all_tokens):,} tokens so far")

    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(args.output)
    print(f"\nSaved {len(arr):,} tokens to {args.output}")
    print(f"File size: {arr.nbytes / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
