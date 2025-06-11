import os
from PIL import Image
import numpy as np
import torch
import clip
from torch.serialization import safe_globals
from clip.model import CLIP  # Make sure CLIP is importable

from sklearn.metrics.pairwise import cosine_similarity
from backend.config import MODEL_CONFIGS


def load_model_by_name(arch: str, model_variant: str):
    if arch not in MODEL_CONFIGS:
        print(f"[ERROR] Architecture '{arch}' not found.")
        return None

    variant_cfg = MODEL_CONFIGS[arch].get(model_variant)
    if not variant_cfg:
        print(f"[ERROR] Variant '{model_variant}' not found in '{arch}'.")
        return None

    clip_arch = variant_cfg["clip_arch"]
    checkpoint = variant_cfg["checkpoint"]
    use_finetuned = variant_cfg["use_finetuned"]
    gallery_path = variant_cfg["gallery_feats"]

    try:
        model, preprocess = load_model(
            model_name=clip_arch,
            checkpoint=checkpoint,
            use_finetuned=use_finetuned
        )
        gallery_feats, gallery_paths = load_gallery_features(gallery_path)
        return model, preprocess, gallery_feats, gallery_paths
    except Exception as e:
        print(f"[ERROR] Failed to load: {e}")
        return None


def load_model(model_name, checkpoint=None, use_finetuned=False):
    print("inside load_model")

    # default init
    model, preprocess = clip.load(model_name, device="cpu")

    if use_finetuned:
        if checkpoint and os.path.exists(checkpoint):
            try:
                print(f"Loading fine-tuned checkpoint from: {checkpoint}")

                # load entire saved model safely
                from torch.serialization import safe_globals
                from clip.model import CLIP

                with safe_globals({"CLIP": CLIP}):
                    data = torch.load(checkpoint, map_location="cpu", weights_only=False)

                #  CASE 1: Only state_dict (usual fine-tuned vision head)
                if "model_state_dict" in data:
                    print("Detected Stage 3 format")

                    # Extract only visual encoder keys
                    visual_state_dict = {
                        k.replace("visual.", ""): v
                        for k, v in data["model_state_dict"].items()
                        if k.startswith("visual.")
                    }

                    model.visual.load_state_dict(visual_state_dict, strict=True)
                #  CASE 2: Full model object saved directly (e.g. torch.save({"model": model}))
                elif "model" in data:
                    print("Detected Stage 2 format (full model object)")
                    saved_model = data["model"]
                    model.visual.load_state_dict(saved_model.visual.state_dict(), strict=True)

                else:
                    raise ValueError(" Unknown checkpoint structure.")

            except Exception as e:
                print(f"[ERROR] Failed to load: {e}")
                return None
        else:
            raise FileNotFoundError(f" Checkpoint file not found: {checkpoint}")

    model.eval()
    return model, preprocess




def load_gallery_features(path):
    gallery_data = torch.load(path, map_location="cpu")
    if "features" in gallery_data and "paths" in gallery_data:
        return gallery_data["features"], gallery_data["paths"]
    elif "features" in gallery_data and "image_paths" in gallery_data:
        return gallery_data["features"], gallery_data["image_paths"]
    raise ValueError("Unsupported gallery feature format.")

def embed_query_image(image_bytes, model, preprocess):
    image = Image.open(image_bytes).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze(0)

def retrieve_top_k(query_feat, gallery_feats, gallery_paths, k=5):
    query_feat = query_feat / query_feat.norm()
    gallery_feats = gallery_feats / gallery_feats.norm(dim=1, keepdim=True)
    scores = cosine_similarity(query_feat.unsqueeze(0), gallery_feats)[0]
    topk_indices = np.argsort(-scores)[:k]

    results = []
    for rank, idx in enumerate(topk_indices):
        results.append({
            "url": f"/static/gallery/{gallery_paths[idx]}",  # Adjust path if needed
            "score": float(scores[idx]),
            "rank": rank + 1,
            "label": os.path.basename(os.path.dirname(gallery_paths[idx]))
        })
    return results


# This must match what you mounted in FastAPI

def run_inference(model, preprocess, query_image_file, gallery_feats, gallery_paths, top_k=50):
    """
    Given a query image file and gallery features, return top-k similar results.
    """
    import torch
    from PIL import Image
    import clip
    import os

    # Load image and preprocess
    image = Image.open(query_image_file).convert("RGB")
    image_input = preprocess(image).unsqueeze(0)

    # Compute features and normalize
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        # Ensure both tensors are float32
        image_features = image_features.to(dtype=torch.float32)
        gallery_feats = gallery_feats.to(dtype=torch.float32)

        similarities = (image_features @ gallery_feats.T).squeeze(0)

    # Get top-k most similar indices
    top_scores, top_indices = similarities.topk(top_k)

    print(similarities)
    # Prepare output list
    results = []
    for i, idx in enumerate(top_indices):
        img_path = gallery_paths[idx]
        score = top_scores[i].item()
        label = os.path.basename(os.path.dirname(img_path))  # or however you want to extract label

        # Ensure this works with both absolute and relative paths
        # Assuming `img_path` is relative to "img/"
        # Don't prepend "gallery" if `img_path` already starts with "img/"
        if img_path.startswith("img/"):
            web_url = f"/static/{img_path}"
        else:
            web_url = f"/static/gallery/{img_path}"

        print(web_url)
        results.append({
            "url": web_url,
            "score": score,
            "rank": i + 1,
            "label": label,
        })

    return results
