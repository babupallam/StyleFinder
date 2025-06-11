import json
import os
from tqdm import tqdm

# === Input/Output paths ===
input_path = "data/raw/Anno/list_description_inshop.json"
output_path = "data/processed/metadata/image_prompts_cleaned_for_v4.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# === Load original item-level description JSON ===
with open(input_path, "r", encoding="utf-8") as f:
    raw_items = json.load(f)

print(f"Loaded {len(raw_items)} items from {input_path}")

# === Build cleaned prompt dictionary ===
cleaned_prompts = {}

for item in tqdm(raw_items, desc="Processing descriptions"):
    item_id = item["item"]
    color = item.get("color", "").strip()
    description_list = item.get("description", [])

    if not description_list:
        continue

    # Remove first sentence from the first paragraph
    first_para = description_list[0]
    if '.' in first_para:
        _, *remaining = first_para.split('.', 1)
        description_list[0] = remaining[0].strip() if remaining else ''
    else:
        description_list = description_list[1:]

    # Filter out empty lines
    filtered_lines = [line.strip() for line in description_list if line.strip()]

    # Compose prompt
    prompt = f"A clothing item in {color}." if color else "A clothing item."
    if filtered_lines:
        prompt += " " + " ".join(filtered_lines)

    cleaned_prompts[item_id] = prompt.strip()

# === Save to file ===
with open(output_path, "w") as f:
    json.dump(cleaned_prompts, f, indent=2)

print(f"\nDone! Saved cleaned prompts to: {output_path}")
print(f"Total prompts written: {len(cleaned_prompts)}")
