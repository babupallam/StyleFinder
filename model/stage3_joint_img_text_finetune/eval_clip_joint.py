import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from PIL import Image
import clip


def extract_features(model, preprocess, split_name, split_dict, output_path):
    model.eval()
    all_features, all_names = [],[]
    base_dir = os.path.join("data", "processed", "splits", split_name, "img")

    for rel_name in tqdm(split_dict[split_name], desc=f"Extracting {split_name}"):
        rel_path = rel_name.replace("\\", "/").lstrip("img/")
        img_path = os.path.normpath(os.path.join(base_dir, rel_path))
        if not os.path.exists(img_path):
            print(f"Missing: {img_path}")
            continue
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(model.device)
            with torch.no_grad():
                feat = model.encode_image(image_tensor)
                feat = nn.functional.normalize(feat, dim=1)
                all_features.append(feat.cpu())
                all_names.append(rel_name)
        except Exception as e:
            print(f"[Warning] Skipping {rel_name}: {e}")
            continue

    if not all_features:
        raise RuntimeError(f"No features extracted for split: {split_name}")

    torch.save({
        "features": torch.cat(all_features, dim=0),
        "image_paths": all_names
    }, output_path)


def load_features(path):
    data = torch.load(path)
    return data["features"], data["image_paths"]


def compute_metrics(sim_matrix, query_ids, gallery_ids, topk=(1, 5, 10)):
    mAP_total = 0
    rank_hits = {k: 0 for k in topk}
    total = len(query_ids)
    for i, q_id in enumerate(query_ids):
        sims = sim_matrix[i]
        sorted_indices = torch.argsort(sims, descending=True)
        ranked_ids = [gallery_ids[j] for j in sorted_indices.tolist()]
        relevant = [1 if r == q_id else 0 for r in ranked_ids]
        correct, ap = 0, 0
        for rank, rel in enumerate(relevant, 1):
            if rel:
                correct += 1
                ap += correct / rank
        mAP_total += ap / (relevant.count(1) + 1e-8)
        for k in topk:
            if q_id in ranked_ids[:k]:
                rank_hits[k] += 1
    mAP = mAP_total / total
    rank_results = {f"Rank-{k}": rank_hits[k] / total for k in topk}
    return mAP, rank_results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(args.model_arch, device=device)
    model = model.float()
    model.device = device

    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load metadata
    with open(os.path.join(args.meta_dir, "image_splits.json"), "r") as f:
        split_dict = json.load(f)
    with open(os.path.join(args.meta_dir, "image_paths.json"), "r") as f:
        item_id_map = json.load(f)

    # Output folder
    model_id = os.path.splitext(os.path.basename(args.ckpt))[0]
    out_dir = os.path.join("data", "processed", model_id)
    os.makedirs(out_dir, exist_ok=True)

    query_path = os.path.join(out_dir, "query.pt")
    gallery_path = os.path.join(out_dir, "gallery.pt")

    extract_features(model, preprocess, "query", split_dict, query_path)
    extract_features(model, preprocess, "gallery", split_dict, gallery_path)

    query_feats, query_paths = load_features(query_path)
    gallery_feats, gallery_paths = load_features(gallery_path)

    query_ids = [item_id_map[p]["item_id"] for p in query_paths]
    gallery_ids = [item_id_map[p]["item_id"] for p in gallery_paths]

    sim_matrix = torch.matmul(query_feats, gallery_feats.T)
    mAP, rank_k = compute_metrics(sim_matrix, query_ids, gallery_ids)

    log_dir = os.path.join("model", "stage3_joint_img_text_finetune", "logs", model_id)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    log_path = os.path.join(log_dir, f"eval_log_{model_name}_{timestamp}.txt")

    # Save detailed results to JSON
    metrics_out = {
        "mAP": round(mAP, 4),
        "Rank-1": round(rank_k["Rank-1"] * 100, 2),
        "Rank-5": round(rank_k["Rank-5"] * 100, 2),
        "Rank-10": round(rank_k["Rank-10"] * 100, 2),
        "timestamp": timestamp,
        "model_id": model_id,
        "query_size": len(query_ids),
        "gallery_size": len(gallery_ids),
        "ckpt": args.ckpt
    }
    json_path = os.path.join(log_dir, f"eval_metrics_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    # Also save a summary log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Evaluation for model: {model_id}\n")
        f.write(f"mAP: {mAP:.4f}\n")
        for k, v in rank_k.items():
            f.write(f"{k}: {v * 100:.2f}%\n")

    print("\n Evaluation Results")
    print(f"mAP: {mAP:.4f}")
    for k, v in rank_k.items():
        print(f"{k}: {v * 100:.2f}%")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to saved .pth model checkpoint")
    parser.add_argument("--meta_dir", type=str, default="data/processed/metadata", help="Directory with metadata files")
    parser.add_argument("--image_dir", type=str, default="data/processed/splits", help="Base image directory")
    parser.add_argument("--model_arch", type=str, default="ViT-B/16", help="CLIP model architecture")
    parser.add_argument("--store_path", type=str, default="eval_results", help="Directory to save results")

    args = parser.parse_args()
    main(args)

