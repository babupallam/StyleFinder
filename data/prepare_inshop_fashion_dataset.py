import os
import json
import argparse
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ========= DEFAULT CONFIG ========= #
BASE_RAW = "data/raw"
BASE_PROCESSED = "data/processed"
IMG_FOLDER = os.path.join(BASE_RAW, "Img")
SPLIT_FILE = os.path.join(BASE_RAW, "Eval", "list_eval_partition.txt")
DESC_FILE = os.path.join(BASE_RAW, "Anno", "list_description_inshop.json")

# ========== LOADING FUNCTIONS ========== #

def load_split_file():
    df = pd.read_csv(SPLIT_FILE, sep=r'\s+', skiprows=2, header=None)
    df.columns = ["image_name", "item_id", "split"]
    return df

def load_descriptions():
    with open(DESC_FILE, 'r', encoding='utf-8') as f:
        desc_json = json.load(f)

    desc_map = {}
    for entry in desc_json:
        item_id = entry.get("item_id") // baseid = entry.get("item_id")
        color = entry.get("color", "") // 000895_cream
        00897_red

        desc = entry.get("description", "")
        text = f"a {color} {desc}".strip()
        if item_id and text != "a":
            desc_map[item_id] = text
    return desc_map

# ========== MAIN PROCESS FUNCTION ========== #

def process_dataset(out_img_dir):
    os.makedirs(out_img_dir, exist_ok=True)
    out_meta_dir = os.path.join(BASE_PROCESSED, "metadata")
    os.makedirs(out_meta_dir, exist_ok=True)

    df = load_split_file()
    desc_map = load_descriptions()

    # Add val split (20% of train)
    train_df = df[df["split"] == "train"]
    # Filter out item_ids with only 1 image
    counts = train_df["item_id"].value_counts()
    valid_ids = counts[counts > 1].index
    train_df_strat = train_df[train_df["item_id"].isin(valid_ids)]

    # Stratified split on valid item_ids
    train_part, val_part = train_test_split(
        train_df_strat,
        test_size=0.2,
        random_state=42,
        stratify=train_df_strat["item_id"]
    )

    # Keep untouched items with only 1 image in train
    remainder = train_df[~train_df.index.isin(train_part.index) & ~train_df.index.isin(val_part.index)]

    # Merge together the final train set
    train_final = pd.concat([train_part, remainder], axis=0)
    df.loc[train_final.index, "split"] = "train"
    df.loc[val_part.index, "split"] = "val"

    df.loc[val_part.index, "split"] = "val"

    image_paths = {}
    image_texts = {}
    image_splits = {"train": [], "val": [], "query": [], "gallery": []}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        img_rel_path = row["image_name"]
        item_id = str(row["item_id"])
        split = row["split"]
        img_full_path = os.path.join(IMG_FOLDER, img_rel_path)
        out_img_path = os.path.join(out_img_dir, split, img_rel_path)

        os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
        text_prompt = desc_map.get(item_id, "a photo of clothing item")

        try:
            Image.open(img_full_path).convert("RGB").save(out_img_path)
        except Exception as e:
            print(f"[Warning] Skipping image {img_full_path}: {e}")
            continue

        image_paths[img_rel_path] = {"item_id": item_id, "split": split}
        image_texts[img_rel_path] = text_prompt
        image_splits[split].append(img_rel_path)

    with open(os.path.join(out_meta_dir, "image_paths.json"), "w") as f:
        json.dump(image_paths, f, indent=2)
    with open(os.path.join(out_meta_dir, "image_texts.json"), "w") as f:
        json.dump(image_texts, f, indent=2)
    with open(os.path.join(out_meta_dir, "image_splits.json"), "w") as f:
        json.dump(image_splits, f, indent=2)

    print("\nDataset processed and saved:")
    print(f"  Images in:     {out_img_dir}")
    print(f"  Metadata in:   {out_meta_dir}")
    print("\n Split counts:\n", pd.Series({k: len(v) for k, v in image_splits.items()}))

# ========== ENTRY POINT ========== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare In-shop Fashion dataset for visual retrieval.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="images_full",
        help="Output directory under data/processed"
    )
    args = parser.parse_args()
    out_img_path = os.path.join(BASE_PROCESSED, args.out_dir)
    process_dataset(out_img_dir=out_img_path)
