import json
from pathlib import Path

# ========== CONFIG ==========
DESC_FILE = "data/raw/Anno/list_description_inshop.json"
META_FILE = "data/processed/metadata/image_paths.json"
OUT_FILE = "data/processed/metadata/image_prompts_per_identity.json"

# ========== SETUP ==========
Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)

# ========== LOAD FILES ==========
with open(DESC_FILE, 'r', encoding='utf-8') as f:
    desc_data = json.load(f)

with open(META_FILE, 'r', encoding='utf-8') as f:
    meta_data = json.load(f)

# ========== BUILD IDDESCRIPTION MAP ==========
desc_map = {}
for entry in desc_data:
    item_id = entry["item"]
    color = entry.get("color", "").strip()
    desc_lines = entry.get("description", [])
    description = " ".join(line.strip() for line in desc_lines if line.strip())
    prompt = f"A clothing item in {color}. {description}".strip()
    desc_map[item_id] = prompt

# ========== GATHER ALL ITEM IDs FROM METADATA ==========
all_item_ids = {meta["item_id"] for meta in meta_data.values()}

# ========== FILL PROMPTS FOR ALL ITEM IDs ==========
identity_prompts = {}
for item_id in sorted(all_item_ids):
    if item_id in desc_map:
        identity_prompts[item_id] = desc_map[item_id]
    else:
        identity_prompts[item_id] = f"A photo of {item_id.replace('_', ' ')}"

# ========== SAVE ==========
with open(OUT_FILE, "w", encoding='utf-8') as f:
    json.dump(identity_prompts, f, indent=2)

print(f" Saved {len(identity_prompts)} prompts to {OUT_FILE}")
