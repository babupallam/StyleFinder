import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import torch
import numpy as np

from torchvision import transforms

# ========= PREPROCESSING ========= #
def get_preprocess():
    return transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4815, 0.4578, 0.4082],
                             std=[0.2686, 0.2613, 0.2758]),
    ])

# ========= LOAD IMAGE PATHS FROM METADATA ========= #
def load_image_paths(split_name, image_root, split_dict):
    paths = []
    for rel_path in split_dict.get(split_name, []):
        img_path = os.path.join(image_root, split_name, rel_path)
        if os.path.exists(img_path):
            paths.append((rel_path, img_path))
        else:
            print(f"[Missing] {img_path}")
    return paths

# ========= MAIN EXTRACTOR ========= #
def extract_features_finetuned(model, image_root, meta_dir, out_dir, batch_size, device):
    preprocess = get_preprocess()
    model.eval().to(device)

    # Load split metadata
    with open(os.path.join(meta_dir, "image_splits.json"), "r") as f:
        split_dict = json.load(f)

    os.makedirs(out_dir, exist_ok=True)

    for split in ["train", "val", "query", "gallery"]:
        print(f"\n Processing split: {split}")
        image_list = load_image_paths(split, image_root, split_dict)
        print(f"   Found {len(image_list)} images")

        all_features = []
        all_names = []
        batch = []
        batch_names = []

        for rel_name, full_path in tqdm(image_list):
            try:
                image = Image.open(full_path).convert("RGB")
                tensor = preprocess(image)
                batch.append(tensor)
                batch_names.append(rel_name)

                if len(batch) >= batch_size:
                    with torch.no_grad():
                        batch_tensor = torch.stack(batch).to(device)
                        features = model.encode_image(batch_tensor).cpu()
                        all_features.append(features)
                        all_names.extend(batch_names)
                    batch = []
                    batch_names = []

            except Exception as e:
                print(f"[Warning] Skipping {rel_name}  {e}")

        if batch:
            with torch.no_grad():
                batch_tensor = torch.stack(batch).to(device)
                features = model.encode_image(batch_tensor).cpu()
                all_features.append(features)
                all_names.extend(batch_names)

        if all_features:
            all_features = torch.cat(all_features, dim=0)
            torch.save(
                {"features": all_features, "image_paths": all_names},
                os.path.join(out_dir, f"{split}.pt")
            )
            print(f" Saved {split} features to {out_dir}/{split}.pt")
        else:
            print(f" No features saved for split: {split}")

# ========= ENTRY POINT ========= #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features using fine-tuned CLIP model")
    parser.add_argument("--img_dir", type=str, required=True, help="Folder like data/processed/splits")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for .pt files")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to your fine-tuned CLIP checkpoint")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--meta_dir", type=str, default="data/processed/metadata", help="Path to metadata folder")

    args = parser.parse_args()

    # Load model from checkpoint (you must define how your model loads)
    checkpoint = torch.load(args.ckpt, map_location=args.device)
    model = checkpoint["model"] if "model" in checkpoint else checkpoint  # adjust if needed

    extract_features_finetuned(
        model=model,
        image_root=args.img_dir,
        meta_dir=args.meta_dir,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        device=args.device
    )
