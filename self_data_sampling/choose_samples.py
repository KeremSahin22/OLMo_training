import random
import requests
import yaml

TARGET_TOKENS = 900_000_000_000
SEED=1234
with open("configs/official-0724/OLMo-1B.yaml", "r") as f:  
    config = yaml.safe_load(f)

paths_list = config["data"]["paths"]

rng = random.Random(SEED)
shuffled_paths = paths_list.copy()
rng.shuffle(shuffled_paths)

selected_paths = []
total_tokens = 0
for path in shuffled_paths:
    size_bytes = int(requests.head(path).headers["Content-Length"])
    total_tokens += size_bytes // 2  # uint16
    selected_paths.append(path)
    if total_tokens >= TARGET_TOKENS:
        break

print(f"Selected {len(selected_paths)} paths")
print(f"Total tokens: {total_tokens}")