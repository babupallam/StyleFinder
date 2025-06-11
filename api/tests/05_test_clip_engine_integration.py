import os
import sys
import time
import torch
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add root path so `backend.services.clip_engine` can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from api.backend.services.clip_engine import CLIPInferenceEngine


# === Configuration ===
MODEL_PATH = "train/stage3_joint_img_text_finetune/checkpoints/vitb16_joint_FINAL.pth"
SAMPLE_IMAGE = "data/processed/images_cropped/query/img/WOMEN/Dresses/id_00000008/02_7_additional.jpg"
SAMPLE_TEXT = "A black floral dress with sleeveless design and fitted waist, commonly worn in spring."

def test_image_to_image_inference():
    print("\n[Test] Image-to-Image Inference")

    engine = CLIPInferenceEngine()
    engine.load_model(MODEL_PATH)

    assert os.path.exists(SAMPLE_IMAGE), "Sample image not found"

    img = Image.open(SAMPLE_IMAGE).convert("RGB")

    start = time.time()
    query_feat = engine.encode_image(img)
    gallery_feat = engine.encode_image(img)  # using same image to simulate match
    elapsed = time.time() - start

    sim = cosine_similarity(query_feat.cpu().numpy(), gallery_feat.cpu().numpy())[0, 0]
    print(f"Image→Image similarity: {sim:.4f}")
    print(f"Embedding shape: {query_feat.shape}")
    print(f"Time taken: {elapsed:.4f}s")

    assert query_feat.shape == (1, 512), "Expected 512-D feature"
    assert abs(query_feat.norm().item() - 1.0) < 1e-3, "Query embedding not normalized"
    assert sim > 0.8, "Expected identical image similarity to be high"

def test_text_to_image_inference():
    print("\n[Test] Text-to-Image Inference")

    engine = CLIPInferenceEngine()
    engine.load_model(MODEL_PATH)

    img = Image.open(SAMPLE_IMAGE).convert("RGB")

    start = time.time()
    text_feat = engine.encode_text(SAMPLE_TEXT)
    img_feat = engine.encode_image(img)
    elapsed = time.time() - start

    sim = cosine_similarity(img_feat.cpu().numpy(), text_feat.cpu().numpy())[0, 0]
    print(f"Text→Image similarity: {sim:.4f}")
    print(f"Text shape: {text_feat.shape}, Image shape: {img_feat.shape}")
    print(f"Time taken: {elapsed:.4f}s")

    assert text_feat.shape == (1, 512), "Expected 512-D feature"
    assert abs(text_feat.norm().item() - 1.0) < 1e-3, "Text embedding not normalized"
    assert sim > 0.2, "Expected similarity between matching text and image"

# === Main Runner ===
if __name__ == "__main__":
    test_image_to_image_inference()
    test_text_to_image_inference()
    print("\nIntegration test of clip_engine completed successfully.")
