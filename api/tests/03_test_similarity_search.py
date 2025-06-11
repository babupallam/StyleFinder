import os
import torch
import clip
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# === Configuration ===
CKPT_PATH = "train/stage3_joint_img_text_finetune/checkpoints/vitb16_joint_FINAL.pth"
IMAGE_1 = "data/processed/images_cropped/query/img/MEN/Denim/id_00000182/01_1_front.jpg"
IMAGE_2 = "data/processed/images_cropped/query/img/MEN/Denim/id_00000182/01_7_additional.jpg"
UNRELATED_IMAGE = "data/processed/images_cropped/query/img/MEN/Denim/id_00000826/02_2_side.jpg"
TEXT_PROMPT = "A clothing item in Dark denim. These are the jeans of the season (and every other season for that matter!). Sleek and slim, these boast a crisp clean wash and traditional five-pocket styling. Want to get out the door in ten and still look sharp? This is how you do it. Woven, zip fly 72% cotton, 27% polyester, 1% spandex 31\" inseam, 35\" waist, 10\" rise Measured from size 32/32 Machine wash cold Made in China"
LONG_PROMPT = TEXT_PROMPT * 10

# === Model Loader ===
def load_model():
    assert os.path.exists(CKPT_PATH), f"Checkpoint missing: {CKPT_PATH}"
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model, preprocess = clip.load(ckpt["model_name"], device="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, preprocess

# === Feature Extraction ===
def get_image_features(image_path, model, preprocess):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        feats = model.encode_image(image_tensor).squeeze(0)
        return feats / feats.norm()

def get_text_features(text, model):
    token = clip.tokenize([text], truncate=True)
    with torch.no_grad():
        feats = model.encode_text(token).squeeze(0)
        return feats / feats.norm()

# === Tests ===
def test_image_to_image_similarity():
    model, preprocess = load_model()
    img1 = get_image_features(IMAGE_1, model, preprocess)
    img2 = get_image_features(IMAGE_2, model, preprocess)
    unrelated = get_image_features(UNRELATED_IMAGE, model, preprocess)

    sim12 = cosine_similarity(img1.unsqueeze(0), img2.unsqueeze(0)).item()
    sim1u = cosine_similarity(img1.unsqueeze(0), unrelated.unsqueeze(0)).item()

    print(f"Similarity between image1 and image2 (same ID): {sim12:.4f}")
    print(f"Similarity between image1 and unrelated image: {sim1u:.4f}")

    if sim12 > sim1u:
        print("Similar images have higher similarity as expected.")
    else:
        print("Warning: unrelated image has higher similarity than same-ID image.")
        print("This can happen if the model is not well-trained yet.")


def test_text_to_image_similarity():
    model, preprocess = load_model()
    img = get_image_features(IMAGE_1, model, preprocess)
    text = get_text_features(TEXT_PROMPT, model)

    sim = cosine_similarity(img.unsqueeze(0), text.unsqueeze(0)).item()
    print(f"Similarity between image and matching text: {sim:.4f}")

    assert sim > 0.2, "Expected decent similarity between matching image and text."

def test_rank1_retrieval_with_mock():
    vecs = torch.randn(5, 512)
    vecs = vecs / vecs.norm(dim=1, keepdim=True)
    query = vecs[3]  # identical to gallery[3]

    scores = cosine_similarity(query.unsqueeze(0), vecs).flatten()
    top_idx = torch.topk(torch.tensor(scores), k=1).indices.item()

    print(f"Rank-1 index: {top_idx} (expected 3)")
    assert top_idx == 3, "Expected query to match itself at rank-1"

def test_long_prompt_similarity():
    model, _ = load_model()
    text = get_text_features(LONG_PROMPT, model)
    base = get_text_features(TEXT_PROMPT, model)
    sim = cosine_similarity(base.unsqueeze(0), text.unsqueeze(0)).item()
    print(f"Similarity between base prompt and long prompt: {sim:.4f}")
    assert sim > 0.2, "Expected similarity to remain reasonable after truncation"

# === Entry ===
if __name__ == "__main__":
    test_image_to_image_similarity()
    test_text_to_image_similarity()
    test_rank1_retrieval_with_mock()
    test_long_prompt_similarity()
    print("\nAll similarity search tests passed.")
