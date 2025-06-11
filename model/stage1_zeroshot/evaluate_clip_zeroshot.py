import os
import json
import argparse
import torch
from datetime import datetime
from utils import cosine_similarity, compute_map, compute_rank_k

def load_features(path):
    data = torch.load(path)
    return data["features"], data["image_paths"]

def extract_item_ids(image_paths, meta_path):
    with open(meta_path, "r") as f:
        mapping = json.load(f)
    # Return the item_id as a string for each image
    return [mapping[name]["item_id"] for name in image_paths]

def main(args):
    # === Load features ===
    print(" Loading query and gallery features...")
    query_feats, query_paths = load_features(os.path.join(args.feature_dir, "query.pt"))
    gallery_feats, gallery_paths = load_features(os.path.join(args.feature_dir, "gallery.pt"))

    # === Load item IDs from metadata ===
    print(" Extracting item IDs...")
    meta_file = os.path.join(args.meta_dir, "image_paths.json")
    query_ids = extract_item_ids(query_paths, meta_file)
    gallery_ids = extract_item_ids(gallery_paths, meta_file)

    # === Similarity matrix ===
    print(" Computing cosine similarity...")
    sim_matrix = cosine_similarity(query_feats, gallery_feats)

    # === Evaluation ===
    print(" Calculating mAP and Rank@K...")
    mAP = compute_map(sim_matrix, query_ids, gallery_ids)
    ranks = compute_rank_k(sim_matrix, query_ids, gallery_ids, k_list=[1, 5, 10])

    # === Logging ===
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"zeroshot_eval_{timestamp}.txt")

    with open(log_path, "w") as f:
        f.write(f"Zero-shot CLIP Evaluation ({args.feature_dir})\n")
        f.write(f"mAP: {mAP:.4f}\n")
        for k, v in ranks.items():
            f.write(f"Rank-{k}: {v*100:.2f}%\n")
    print(f"Saved log to {log_path}")

    print("\nEvaluation Results")
    print(f"mAP: {mAP:.4f}")
    for k, v in ranks.items():
        print(f"Rank-{k}: {v*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot CLIP evaluation (OpenAI CLIP)")
    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Path to directory containing query.pt, gallery.pt, etc. (e.g., data/processed/features_clip)")
    parser.add_argument("--meta_dir", type=str, default="data/processed/metadata",
                        help="Path to metadata folder containing image_paths.json")
    parser.add_argument("--log_dir", type=str, default="model/stage1_zeroshot/logs",
                        help="Output directory for evaluation logs")
    args = parser.parse_args()
    main(args)
