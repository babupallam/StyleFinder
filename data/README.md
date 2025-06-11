
```markdown
#  data/ Directory Overview

This directory contains all raw and processed assets used for training, evaluating, and extracting features with CLIP-based models in the In-Shop Clothes Retrieval task.


---

##  Metadata Files

Located in `processed/metadata/`:

- `image_paths.json`  
  Maps each image to its item ID and split (e.g., train/val/query/gallery).

- `image_texts.json`  
  Stores the prompt (e.g., `"a red denim jacket"`) for each image.

- `image_splits.json`  
  Lists which image belongs to which split.

---

##  Scripts

Located in `raw/`:

- `prepare_inshop_fashion_dataset.py`  
  Preprocesses original DeepFashion images and annotations into train/val/query/gallery splits. Also generates the metadata files.

- `extract_clip_features.py`  
  Extracts features from OpenAI CLIP models (ViT-B/16 or RN50) and saves them to `.pt` files.

- `extract_finetuned_features.py`  
  Extracts features using your fine-tuned CLIP joint encoders.

---

##  Feature Outputs

- Saved in `processed/features_clip_vitb16/`, `features_clip_rn50/`, etc.
- Format: `{split}.pt` containing:
  - `"features"`: `[N x D]` tensor (image embeddings)
  - `"image_paths"`: list of corresponding filenames

---

##  Model Checkpoints

- Saved under `clip_joint_YYYYMMDD_*_BEST/`
- Each folder corresponds to a fine-tuned model checkpoint (Stage 3).
- Used during inference or continued training.

---

##  Tip

Run this to prepare the dataset:

```bash
python raw/prepare_inshop_fashion_dataset.py --out_dir images_full
````

And this to extract features with your fine-tuned model:

```bash
python raw/extract_finetuned_features.py \
  --img_dir data/processed/splits \
  --meta_dir data/processed/metadata \
  --ckpt path/to/checkpoint.pth \
  --out_dir data/processed/features_clip_vitb16 \
  --batch_size 64
```

---
