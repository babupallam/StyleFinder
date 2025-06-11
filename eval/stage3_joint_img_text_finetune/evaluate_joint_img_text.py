import os
import json
import argparse
from tqdm import tqdm
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import clip

# ========== Load Helpers ==========
def load_image(path, preprocess):
    try:
        image = Image.open(path).convert("RGB")
        return preprocess(image)
    except Exception as e:
        print(f"[⚠️] Could not process {path}: {e}")
        return None


def compute_metrics(similarity_matrix, query_ids, gallery_ids, top_k=(1, 5, 10)):
    ranks = {k: 0 for k in top_k}
    mAP = 0.0
    num_queries = similarity_matrix.shape[0]

    for i in range(num_queries):
        sims = similarity_matrix[i]
        indices = sims.argsort()[::-1]  # descending
        gt_id = query_ids[i]
        retrieved_ids = gallery_ids[indices]

        # mAP
        correct = 0
        avg_precision = 0
        for rank, pred_id in enumerate(retrieved_ids):
            if pred_id == gt_id:
                correct += 1
                avg_precision += correct / (rank + 1)
        if correct > 0:
            mAP += avg_precision / correct

        # Rank-k
        for k in top_k:
            if gt_id in retrieved_ids[:k]:
                ranks[k] += 1

    results = {
        "mAP": mAP / num_queries,
        **{f"Rank-{k}": ranks[k] / num_queries * 100 for k in top_k}
    }
    return results

# ========== Main ==========

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.log_dir, exist_ok=True)

    # Load metadata
    with open(args.metadata, "r") as f:
        meta = json.load(f)

    with open(args.prompt_json, "r") as f:
        id2text = json.load(f)

    subset_items = None
    if args.subset_json:
        with open(args.subset_json, "r") as f:
            subset_data = json.load(f)
        subset_items = set(subset_data.keys())
        print(f"📃 Using subset: {len(subset_items)} identities")

    # Filter query/gallery entries
    query_paths, gallery_paths, query_ids, gallery_ids = [], [], [], []

    for rel_path, entry in meta.items():
        item_id = entry["item_id"]
        split = entry["split"]

        if item_id not in id2text:
            continue
        if subset_items and item_id not in subset_items:
            continue

        abs_path = os.path.join(args.image_dir, split, "img", rel_path[4:] if rel_path.startswith("img/") else rel_path)
        if not os.path.exists(abs_path):
            continue

        if split == "query":
            query_paths.append(abs_path)
            query_ids.append(item_id)
        elif split == "gallery":
            gallery_paths.append(abs_path)
            gallery_ids.append(item_id)

    print(f"📸 Query images: {len(query_paths)}, Gallery images: {len(gallery_paths)}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = ckpt.get("model_name", args.model)
    model, preprocess = clip.load(model_name, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Extract text features (unique prompts only)
    item_ids = sorted(set(query_ids + gallery_ids))
    text_feats = {}
    for item_id in item_ids:
        text = id2text[item_id][:300]  # truncate long prompts
        tokens = clip.tokenize([text], truncate=True).to(device)
        with torch.no_grad():
            feat = model.encode_text(tokens)
            feat = F.normalize(feat, dim=-1)
        text_feats[item_id] = feat.squeeze(0)

    # Extract image features
    def extract_feats(paths, ids):
        feats = []
        kept_ids = []
        for path, item_id in tqdm(zip(paths, ids), total=len(paths), desc=f"Extracting {len(paths)} images"):
            img_tensor = load_image(path, preprocess)
            if img_tensor is None:
                continue
            with torch.no_grad():
                feat = model.encode_image(img_tensor.unsqueeze(0).to(device))
                feat = F.normalize(feat, dim=-1)
            feats.append(feat.squeeze(0).cpu())
            kept_ids.append(item_id)
        return torch.stack(feats), kept_ids

    q_feats, q_ids = extract_feats(query_paths, query_ids)
    g_feats, g_ids = extract_feats(gallery_paths, gallery_ids)

    # Compute cosine similarity
    sims = q_feats @ g_feats.T
    results = compute_metrics(sims.numpy(), q_ids, g_ids)

    # Log results
    log_file = os.path.join(args.log_dir, f"eval_log_{args.split_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(log_file, "w") as f:
        f.write("🎯 Evaluation Results (Stage 3 Joint CLIP)\n")
        f.write(f"Checkpoint: {args.checkpoint}\n\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")

    print("\n✅ Evaluation Complete:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    print(f"\n📝 Saved to: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--prompt_json", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="ViT-B/16")
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--split_tag", type=str, required=True)
    parser.add_argument("--subset_json", type=str, default=None)
    args = parser.parse_args()
    main(args)
