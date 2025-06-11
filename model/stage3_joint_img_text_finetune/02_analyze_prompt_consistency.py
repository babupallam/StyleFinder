import os
import json
from collections import defaultdict

# === Set your file paths ===
image_meta_path = "data/processed/metadata/image_paths.json"
prompt_json_path = "data/processed/metadata/image_prompts_per_identity.json"
image_root = "data/processed/splits/train/img"

# === Load files ===
with open(image_meta_path, "r") as f:
    image_meta = json.load(f)

with open(prompt_json_path, "r") as f:
    id_to_prompt = json.load(f)

# === Collect image paths per identity ===
identity_to_paths = defaultdict(list)
identity_prompt_set = defaultdict(set)

for rel_path, meta in image_meta.items():
    if meta["split"] != "train":
        continue

    item_id = meta["item_id"]
    full_path = os.path.join(image_root, rel_path[4:] if rel_path.startswith("img/") else rel_path)

    if not os.path.exists(full_path):
        print(f"[Missing] {full_path}")
        continue

    identity_to_paths[item_id].append(full_path)

    if item_id in id_to_prompt:
        identity_prompt_set[item_id].add(id_to_prompt[item_id])
    else:
        identity_prompt_set[item_id].add("[Missing Prompt]")

# === Analyze consistency ===
too_few = []
multiple_prompts = []
total_identities = len(identity_to_paths)

print("\n=== Dataset Analysis ===\n")
for item_id in identity_to_paths:
    n_images = len(identity_to_paths[item_id])
    n_prompts = len(identity_prompt_set[item_id])

    if n_images < 2:
        too_few.append((item_id, n_images))
    if n_prompts > 1:
        multiple_prompts.append((item_id, n_prompts))

print(f"Total identities: {total_identities}")
print(f"Identities with < 2 images: {len(too_few)}")
print(f"Identities with inconsistent prompts: {len(multiple_prompts)}\n")

# === Show examples ===
if too_few:
    print("  Identities with fewer than 2 images:")
    for item_id, n in too_few[:10]:
        print(f"  {item_id}: {n} images")

if multiple_prompts:
    print("\n  Identities with inconsistent prompts:")
    for item_id, n in multiple_prompts[:10]:
        print(f"  {item_id}: {n} unique prompts")
