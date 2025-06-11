import os
import torch
import clip
import argparse
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import shutil

# === Default paths ===
DEFAULT_IMAGE = "api/tests/uploads/query1.jpg"

#FALLBACK_IMAGE = "data/processed/splits/query/img/WOMEN/Dresses/id_00000008/02_7_additional.jpg"
FALLBACK_IMAGE = "data/processed/splits/query/img/MEN/Shorts/id_00000638/01_3_back.jpg"

GALLERY_FEATS = "data/processed/features_clip/gallery.pt"
FINETUNED_CHECKPOINT = "train/stage2_img_encoder_finetune/checkpoints/vitb16_subset_ep1_bs32_lr5e-05_maxSamples100%.pth"

def load_model(model_name="ViT-B/16", checkpoint=None, use_finetuned=False):
    model, preprocess = clip.load(model_name, device="cpu")

    if use_finetuned:
        print("🔧 Loading fine-tuned image encoder weights...")
        state_dict = torch.load(checkpoint, map_location="cpu")
        visual_state_dict = {
            k.replace("visual.", ""): v for k, v in state_dict.items() if k.startswith("visual.")
        }
        model.visual.load_state_dict(visual_state_dict, strict=True)
    else:
        print(" Using original CLIP (pretrained only)")

    model.eval()
    return model, preprocess

def load_gallery_features(path):
    gallery_data = torch.load(path, map_location="cpu")
    if isinstance(gallery_data, dict):
        if "features" in gallery_data and "paths" in gallery_data:
            return gallery_data["features"], gallery_data["paths"]
        elif "features" in gallery_data and "image_paths" in gallery_data:
            return gallery_data["features"], gallery_data["image_paths"]
    raise ValueError("Unsupported gallery feature format.")

def embed_query_image(image_path, model, preprocess):
    image = Image.open(image_path).convert("RGB")
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
    return [(gallery_paths[i], scores[i]) for i in topk_indices]


import math

def display_results(query_img, results, cols=5):
    root_path = "data/processed/splits/gallery"

    total_images = len(results) + 1  # +1 for the query image
    rows = math.ceil(total_images / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axs = axs.flatten()  # Flatten in case of single row

    # Show the query image in the first cell
    axs[0].imshow(Image.open(query_img))
    axs[0].set_title("Query")
    axs[0].axis("off")

    # Show results in the rest
    for i, (img_path, score) in enumerate(results):
        ax = axs[i + 1]
        normalized_path = os.path.join(root_path, img_path.replace("\\", os.sep))
        if not os.path.exists(normalized_path):
            ax.axis("off")
            ax.set_title(f"Top {i + 1}\nNot Found")
            continue

        ax.imshow(Image.open(normalized_path))
        ax.set_title(f"Top {i + 1}\nScore: {score:.2f}")
        ax.axis("off")

    # Hide any unused subplots
    for i in range(len(results) + 1, len(axs)):
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--gallery_feats", type=str, default=GALLERY_FEATS)
    parser.add_argument("--checkpoint", type=str, default=FINETUNED_CHECKPOINT)
    parser.add_argument("--model", type=str, default="ViT-B/16")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--use_finetuned", action="store_true", help="Use fine-tuned image encoder (Stage 2)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.image_path), exist_ok=True)

    # Force fallback image usage every time
    if os.path.exists(args.image_path):
        print(f"Removing existing query image: {args.image_path}")
        os.remove(args.image_path)



    print(f"Replacing query image with fallback image...")
    shutil.copyfile(FALLBACK_IMAGE, args.imcage_path)

    print("Loading model...")
    model, preprocess = load_model(
        model_name=args.model,
        checkpoint=args.checkpoint,
        use_finetuned=args.use_finetuned
    )

    print("Loading gallery features...")
    gallery_feats, gallery_paths = load_gallery_features(args.gallery_feats)

    print(f"Embedding query image: {args.image_path}")
    query_feat = embed_query_image(args.image_path, model, preprocess)

    print("Searching top-k similar images...")
    results = retrieve_top_k(query_feat, gallery_feats, gallery_paths, args.top_k)

    print("\nTop Matches:")
    for path, score in results:
        print(f"{path} | Score: {score:.4f}")

    display_results(args.image_path, results)



if __name__ == "__main__":
    main()



'''
==> Use original CLIP (default):

python 07_infer_image_to_gallery.py
 
==> Use fine-tuned image encoder:

python 07_infer_image_to_gallery.py --use_finetuned
'''