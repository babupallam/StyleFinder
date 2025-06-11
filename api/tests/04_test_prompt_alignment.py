import os
import json
import clip
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# === Config ===
CKPT_PATH = "train/stage3_joint_img_text_finetune/checkpoints/vitb16_joint_FINAL.pth"
PROMPT_JSON = "data/processed/metadata/image_prompts_per_identity.json"
IMAGE_META = "data/processed/metadata/image_paths.json"
IMAGE_DIR = "data/processed/images_cropped/train/img"

# === Load CLIP ===
def load_model():
    assert os.path.exists(CKPT_PATH), "Checkpoint not found"
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model, preprocess = clip.load(ckpt["model_name"], device="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, preprocess

# === Load Prompt JSON ===
def load_prompt_data():
    with open(PROMPT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

# === Load Metadata ===
def load_metadata():
    with open(IMAGE_META, "r", encoding="utf-8") as f:
        return json.load(f)

# === Feature extraction ===
def get_image_tensor(image_path, preprocess):
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)

# === Test Prompt Presence & Token Length ===
def test_prompt_format_and_tokenization():
    prompt_data = load_prompt_data()
    total = len(prompt_data)
    too_long = 0

    for item_id, text in prompt_data.items():
        assert isinstance(text, str), f"Prompt for {item_id} is not a string"
        assert text.strip(), f"Prompt for {item_id} is empty"

        tokens = clip.tokenize([text], truncate=True)
        if tokens.shape[1] > 77:
            too_long += 1

    print(f"Checked {total} prompts.")
    print(f"{too_long} prompts exceed 77 tokens.")

# === Test Visual-Text Embedding Similarity (Optional) ===
def test_prompt_image_alignment(sample_size=10):
    model, preprocess = load_model()
    meta = load_metadata()
    prompts = load_prompt_data()

    image_paths_by_id = {}
    for rel_path, info in meta.items():
        if info["split"] != "train":
            continue
        item_id = info["item_id"]
        full_path = os.path.join(IMAGE_DIR, rel_path[4:] if rel_path.startswith("img/") else rel_path)
        if os.path.exists(full_path):
            image_paths_by_id.setdefault(item_id, []).append(full_path)

    # Sample test cases
    tested = 0
    aligned = 0

    for item_id, paths in list(image_paths_by_id.items())[:sample_size]:
        if item_id not in prompts:
            continue
        prompt = prompts[item_id]
        if not paths:
            continue

        image = get_image_tensor(paths[0], preprocess)
        token = clip.tokenize([prompt], truncate=True)

        with torch.no_grad():
            image_feat = model.encode_image(image).squeeze(0)
            text_feat = model.encode_text(token).squeeze(0)

        image_feat = image_feat / image_feat.norm()
        text_feat = text_feat / text_feat.norm()
        sim = cosine_similarity(image_feat.unsqueeze(0), text_feat.unsqueeze(0)).item()

        print(f"{item_id} — Cosine similarity: {sim:.4f}")
        if sim > 0.25:
            aligned += 1
        tested += 1

    print(f"\nPrompt-image alignment passed: {aligned}/{tested}")

# === Entry point ===
if __name__ == "__main__":
    test_prompt_format_and_tokenization()
    test_prompt_image_alignment(sample_size=20)
    print("\nPrompt alignment tests completed.")
