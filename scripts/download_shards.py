"""
Download the shards selected by scripts/select_shards.py to local disk, and
rewrite a training config's data.paths to point at the local copies.

Streaming reads shards directly over HTTP works (see MemMapDataset / get_bytes_range
in olmo/util.py), but for a full-scale, possibly multi-day, multi-GPU run, local
files are more reliable and avoid putting sustained load on olmo-data.org.

Usage:
    python scripts/download_shards.py \
        --shard-list configs/data/olmo-mix-50b-shards.yaml \
        --output-dir /disk/u/kerem.sahin/data/olmo-mix-50b \
        --update-config configs/olmo1b-50b-run.yaml
"""

import argparse
import os

import requests
import yaml

CHUNK_SIZE = 64 * 1024 * 1024  # 64MB


def download_file(url, dest_path):
    expected_size = int(requests.head(url, timeout=30).headers["content-length"])

    if os.path.exists(dest_path) and os.path.getsize(dest_path) == expected_size:
        print(f"  already downloaded: {dest_path}")
        return

    tmp_path = dest_path + ".partial"
    resume_from = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0

    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        mode = "ab" if resume_from else "wb"
        downloaded = resume_from
        with open(tmp_path, mode) as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                downloaded += len(chunk)
                print(f"\r  {downloaded / 1e9:.2f} / {expected_size / 1e9:.2f} GB", end="", flush=True)
    print()

    actual_size = os.path.getsize(tmp_path)
    if actual_size != expected_size:
        raise IOError(f"Download incomplete for {url}: got {actual_size} bytes, expected {expected_size}")
    os.rename(tmp_path, dest_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-list", default=os.path.join(os.path.dirname(__file__), "../configs/data/olmo-mix-50b-shards.yaml"))
    parser.add_argument("--output-dir", default="/disk/u/kerem.sahin/data/olmo-mix-50b")
    parser.add_argument("--update-config", default=os.path.join(os.path.dirname(__file__), "../configs/olmo1b-50b-run.yaml"))
    args = parser.parse_args()

    with open(args.shard_list) as f:
        shard_info = yaml.safe_load(f)
    paths = shard_info["paths"]

    os.makedirs(args.output_dir, exist_ok=True)
    local_paths = []
    for i, url in enumerate(paths):
        dest_path = os.path.join(args.output_dir, os.path.basename(url))
        print(f"[{i + 1}/{len(paths)}] {url}")
        download_file(url, dest_path)
        local_paths.append(dest_path)

    print(f"\nAll {len(local_paths)} shards downloaded to {args.output_dir}")

    if args.update_config:
        with open(args.update_config) as f:
            lines = f.readlines()

        out_lines = []
        in_paths_block = False
        for line in lines:
            stripped = line.strip()
            if stripped == "paths:":
                in_paths_block = True
                out_lines.append(line)
                for p in local_paths:
                    out_lines.append(f"    - {p}\n")
                continue
            if in_paths_block:
                if stripped.startswith("- "):
                    continue  # drop old (remote) path entries
                in_paths_block = False
            out_lines.append(line)

        with open(args.update_config, "w") as f:
            f.writelines(out_lines)
        print(f"Updated {args.update_config} data.paths to point at local files")


if __name__ == "__main__":
    main()
