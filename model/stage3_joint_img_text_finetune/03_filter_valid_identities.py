import os
import json
from collections import defaultdict

# Input files
image_meta_path = "data/processed/metadata/image_paths.json"
prompt_json_path = "data/processed/metadata/image_prompts_per_identity.json"

# Output files
filtered_meta_path = "data/processed/metadata/image_paths_clean.json"
filtered_prompt_path = "data/processed/metadata/image_prompts_clean.json"

# Load data
with open(image_meta_path, "r") as f:
    image_meta = json.load(f)
with open(prompt_json_path, "r") as f:
    prompts = json.load(f)

# Map item_id to all its images
id_to_paths = defaultdict(list)
for path, meta in image_meta.items():
    item_id = meta["item_id"]
    id_to_paths[item_id].append(path)

# Filter: at least 2 images + prompt exists
filtered_ids = [
    item_id for item_id, paths in id_to_paths.items()
    if len(paths) >= 2 and item_id in prompts
]

# Build filtered versions
filtered_meta = {
    path: meta for path, meta in image_meta.items()
    if meta["item_id"] in filtered_ids
}
filtered_prompts = {
    item_id: prompts[item_id] for item_id in filtered_ids
}

# Save
os.makedirs(os.path.dirname(filtered_meta_path), exist_ok=True)
with open(filtered_meta_path, "w") as f:
    json.dump(filtered_meta, f, indent=2)
with open(filtered_prompt_path, "w") as f:
    json.dump(filtered_prompts, f, indent=2)

# Print summary
print("Filtering complete!")
print(f"Original identities: {len(id_to_paths)}")
print(f"Filtered identities: {len(filtered_ids)}")
print(f"Images before: {len(image_meta)} | after: {len(filtered_meta)}")
print(f"Prompt entries before: {len(prompts)} | after: {len(filtered_prompts)}")
