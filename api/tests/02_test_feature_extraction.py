import os
import torch
import clip
from PIL import Image
import numpy as np

# === Configuration ===
CKPT_PATH = "train/stage3_joint_img_text_finetune/checkpoints/vitb16_joint_FINAL.pth"
SAMPLE_IMAGE = "data/processed/images_cropped/query/img/WOMEN/Dresses/id_00000008/02_7_additional.jpg"
SAMPLE_PROMPT = "A clothing item in Black. Busy mornings and stacked social calls are no match for this throw-on-and-go tunic! Its boxy silhouette and ultra-soft knit fabrication will have you taking on the day in breezy comfort, while a scoop back lends it some unexpected sartorial side-eye. Plus, you can just as easily throw on this short-sleeved number over leggings to effortlessly transition from day to night. Semi-sheer, lightweight 95% rayon, 5% spandex 30.5\" full length, 42\" chest, 40\" waist, 8\" sleeve length Measured from Small Hand wash cold USA"
LONG_PROMPT = SAMPLE_PROMPT * 10  # Stress test for token limit


def load_model():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model, preprocess = clip.load(ckpt["model_name"], device="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Loaded model from: {CKPT_PATH}")
    return model, preprocess


def test_image_feature_extraction():
    model, preprocess = load_model()

    if not os.path.exists(SAMPLE_IMAGE):
        raise FileNotFoundError(f"Sample image not found: {SAMPLE_IMAGE}")

    image = Image.open(SAMPLE_IMAGE).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        image_feat = model.encode_image(image_tensor).squeeze(0)
        image_feat = image_feat / image_feat.norm()

    assert image_feat.shape[0] == 512, f"Unexpected image feature dimension: {image_feat.shape}"
    assert abs(image_feat.norm().item() - 1.0) < 1e-3, "Image feature is not normalized"
    print("Image feature extracted and normalized.")


def test_text_feature_extraction():
    model, _ = load_model()

    token = clip.tokenize(SAMPLE_PROMPT,truncate=True)
    with torch.no_grad():
        text_feat = model.encode_text(token).squeeze(0)
        text_feat = text_feat / text_feat.norm()

    assert text_feat.shape[0] == 512, f"Unexpected text feature dimension: {text_feat.shape}"
    assert abs(text_feat.norm().item() - 1.0) < 1e-3, "Text feature is not normalized"
    print("Text feature extracted and normalized.")


def test_long_prompt_truncation():
    model, _ = load_model()

    try:
        token = clip.tokenize(LONG_PROMPT, truncate=True)
        with torch.no_grad():
            text_feat = model.encode_text(token).squeeze(0)
            text_feat = text_feat / text_feat.norm()
        print("Long prompt was truncated and encoded successfully.")
    except Exception as e:
        print("Error encoding long prompt:", str(e))


def test_blank_image_handling():
    model, preprocess = load_model()

    blank_image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    image_tensor = preprocess(blank_image).unsqueeze(0)

    try:
        with torch.no_grad():
            feat = model.encode_image(image_tensor).squeeze(0)
            feat = feat / feat.norm()
        print("Blank image encoded successfully.")
    except Exception as e:
        print("Error encoding blank image:", str(e))


if __name__ == "__main__":
    test_image_feature_extraction()
    test_text_feature_extraction()
    test_long_prompt_truncation()
    test_blank_image_handling()
    print("\nAll feature extraction tests completed.")
