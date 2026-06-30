"""
Randomly select a subset of the official olmo-mix v1_5 shards (the same
pre-tokenized, pre-shuffled data used by configs/official-0724/OLMo-1B.yaml)
that together total approximately a target number of tokens.

Shards are raw uint16 token memmaps (2 bytes/token), so token count is
derived directly from each file's byte size via an HTTP HEAD request. Sizes
are cached locally so repeated runs don't re-query the server.

Usage:
    python scripts/select_shards.py \
        --target-tokens 50_000_000_000 \
        --seed 6198 \
        --output configs/data/olmo-mix-50b-shards.yaml
"""

import argparse
import json
import os
import random

import requests
import yaml

OFFICIAL_CONFIG = os.path.join(os.path.dirname(__file__), "../configs/official-0724/OLMo-1B.yaml")
SIZE_CACHE = os.path.join(os.path.dirname(__file__), "../configs/data/shard_sizes_cache.json")
BYTES_PER_TOKEN = 2  # uint16


def load_official_shard_paths():
    with open(OFFICIAL_CONFIG) as f:
        cfg = yaml.safe_load(f)
    return cfg["data"]["paths"]


def load_size_cache():
    if os.path.exists(SIZE_CACHE):
        with open(SIZE_CACHE) as f:
            return json.load(f)
    return {}


def save_size_cache(cache):
    os.makedirs(os.path.dirname(SIZE_CACHE), exist_ok=True)
    with open(SIZE_CACHE, "w") as f:
        json.dump(cache, f, indent=2)


def get_shard_sizes(paths, cache):
    sizes = {}
    missing = [p for p in paths if p not in cache]
    print(f"Fetching sizes for {len(missing)}/{len(paths)} shards not in cache...")
    for i, path in enumerate(missing):
        resp = requests.head(path, timeout=30)
        resp.raise_for_status()
        cache[path] = int(resp.headers["content-length"])
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(missing)}")
            save_size_cache(cache)
    save_size_cache(cache)
    for p in paths:
        sizes[p] = cache[p]
    return sizes


def select_shards(paths, sizes, target_tokens, seed):
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)

    selected = []
    total_tokens = 0
    for path in shuffled:
        if total_tokens >= target_tokens:
            break
        selected.append(path)
        total_tokens += sizes[path] // BYTES_PER_TOKEN
    return selected, total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-tokens", type=int, default=50_000_000_000)
    parser.add_argument("--seed", type=int, default=6198)
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "../configs/data/olmo-mix-50b-shards.yaml"))
    args = parser.parse_args()

    all_paths = load_official_shard_paths()
    cache = load_size_cache()
    sizes = get_shard_sizes(all_paths, cache)

    selected, total_tokens = select_shards(all_paths, sizes, args.target_tokens, args.seed)

    print(f"\nSelected {len(selected)}/{len(all_paths)} shards")
    print(f"Total tokens: {total_tokens:,} (target: {args.target_tokens:,})")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        yaml.safe_dump(
            {
                "seed": args.seed,
                "target_tokens": args.target_tokens,
                "total_tokens": total_tokens,
                "paths": selected,
            },
            f,
            sort_keys=False,
        )
    print(f"Wrote shard list to {args.output}")


if __name__ == "__main__":
    main()
