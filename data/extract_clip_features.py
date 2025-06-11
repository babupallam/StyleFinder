import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
import torch
import clip
from torchvision import transforms

# ========= UTILITY ========= #

def load_image_paths(split_name, image_root, split_dict):
    paths = []
    for rel_path in split_dict[split_name]:
        img_path = os.path.join(image_root, split_name, rel_path)
        if os.path.exists(img_path):
            paths.append((rel_path, img_path))
        else:
            print(f"[Missing] {img_path}")
    return paths

def preprocess_image(image, preprocess):
    try:
        return preprocess(image.convert("RGB"))
    except Exception as e:
        print(f"[Warning] Failed to process image: {e}")
        return None

# ========= MAIN FEATURE EXTRACTOR ========= #

def extract_features(image_root, meta_path, out_dir, model_name, batch_size, device):
    # Note: the following line should change for new models
    # Load model
    model, preprocess = clip.load(model_name, device=device)
    #model = load_my_finetuned_clip("path/to/checkpoint.pt")

    model.eval()

    # Load split metadata
    with open(os.path.join(meta_path, "image_splits.json"), "r") as f:
        split_dict = json.load(f)

    os.makedirs(out_dir, exist_ok=True)

    for split in ["train", "val", "query", "gallery"]:
        print(f"\nProcessing split: {split}")
        image_list = load_image_paths(split, image_root, split_dict)
        print(f"   Found {len(image_list)} images")

        all_features = []
        all_names = []

        batch = []
        batch_names = []

        for rel_name, full_path in tqdm(image_list):
            try:
                image = Image.open(full_path)
                tensor = preprocess_image(image, preprocess)
                if tensor is None:
                    continue
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

        # Final batch
        if batch:
            with torch.no_grad():
                batch_tensor = torch.stack(batch).to(device)
                features = model.encode_image(batch_tensor).cpu()
                all_features.append(features)
                all_names.extend(batch_names)

        # Concatenate all features
        if all_features:
            all_features = torch.cat(all_features, dim=0)
            torch.save(
                {"features": all_features, "image_paths": all_names},
                os.path.join(out_dir, f"{split}.pt")
            )
            print(f" Saved {split} features to {out_dir}/{split}.pt")
        else:
            print(f" No features saved for split: {split}")

# ========= CLI ENTRY ========= #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CLIP image features (OpenAI CLIP)")

    parser.add_argument("--img_dir", type=str, required=True, help="Path to image root (e.g., data/processed/splits)")
    parser.add_argument("--out_dir", type=str, required=True, help="Where to save output .pt files")
    parser.add_argument("--model", type=str, default="ViT-B/16", help="CLIP model name (ViT-B/16 or RN50)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu")
    parser.add_argument("--meta_dir", type=str, default="data/processed/metadata", help="Path to metadata folder")

    args = parser.parse_args()

    extract_features(
        image_root=args.img_dir,
        meta_path=args.meta_dir,
        out_dir=args.out_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device
    )
