
#  CLIP-FH  Model Training Pipeline

This directory contains all training stages for the CLIP-FH pipeline, which fine-tunes CLIP for visual similarity and retrieval in fashion datasets (e.g., In-Shop).

---

##  Directory Structure

```

model/
 stage1\_zeroshot/              # Stage 1  Zero-shot evaluation (no training)
    logs/
    evaluate\_clip\_zeroshot.py
    utils.py
    README.md

 stage2\_img\_encoder\_finetune/  # Stage 2  Fine-tune CLIP image encoder
    checkpoints/              # Saved weights after fine-tuning
    archived/                 # Older experiments
    logs/
    train\_clip\_img\_encoder\_finetune.py
    eval\_clip\_img\_encoder.py
    dataset.py
    loss.py
    README.md

 stage3\_joint\_img\_text\_finetune/  # Stage 3  Joint fine-tuning (image + text)
    checkpoints/                 # Jointly trained model weights
    logs/                        # Preprocessing and filtering tools
    train\_clip\_joint\_encoders.py
    train\_clip\_joint\_encoders\_v1.py
    train\_clip\_joint\_encoders\_v2.py
    train\_clip\_joint\_encoders\_v3.py
    train\_clip\_joint\_encoders\_v4.py
    eval\_clip\_joint.py
    loss.py
    README.md


---

##  Stage 1: Zero-shot Evaluation (`stage1_zeroshot/`)

- Goal: Evaluate pre-trained CLIP (ViT-B/16 or RN50) without training
- Usage:
  ```bash
  python evaluate_clip_zeroshot.py --model ViT-B/16 --image_dir ... --metadata ...
````

* Output:

  * Rank-1, Rank-5, mAP metrics
  * Similarity logs and predictions

---

##  Stage 2: Image Encoder Fine-Tuning (`stage2_img_encoder_finetune/`)

* Goal: Fine-tune CLIPs image encoder using frozen text encoder
* Core script: `train_clip_img_encoder_finetune.py`
* Evaluation: `eval_clip_img_encoder.py`

###  Key Components

* `loss.py`  Combined loss (Triplet + CrossEntropy + Center Loss)
* `dataset.py`  Loads image/prompt pairs for training
* `checkpoints/`  Best model weights per run
* `logs/`  Training logs per experiment

###  Example command:

```bash
python train_clip_img_encoder_finetune.py \
  --image_dir data/processed/splits \
  --metadata_dir data/processed/metadata \
  --model ViT-B/16 \
  --epochs 20 --lr 5e-5 --loss triplet+center \
  --save_path checkpoints/
```

---

##  Stage 3: Joint Image + Text Fine-Tuning (`stage3_joint_img_text_finetune/`)

* Goal: Train both CLIP encoders (image and text) with structured prompts
* Key script: `train_clip_joint_encoders.py`
* Versions `v1`  `v4`: experimental variants (prompt strategies, learning rate, loss)

###  Preprocessing Logs

Located in `logs/`:

* `01_generate_prompts_from_descriptions.py`  Parses raw descriptions into prompts
* `02_analyze_prompt_consistency.py`  Checks if prompts are stable per item ID
* `03_filter_valid_identities.py`  Filters identities with <2 images (for SupCon)
* `04_generate_structured_prompts_for_v4.py`  Generates cleaned, token-safe prompts

###  Training Example:

```bash
python train_clip_joint_encoders_v4.py \
  --image_dir data/processed/splits \
  --prompt_json metadata/image_prompts_cleaned_for_v4.json \
  --model ViT-B/16 \
  --p 16 --k 2 --epochs 50 --loss supcon \
  --save_path checkpoints/ \
  --log_dir logs/
```

###  Evaluation

```bash
python eval_clip_joint.py \
  --ckpt checkpoints/clip_joint_YYYYMMDD_xxxxxx_BEST.pth \
  --image_dir data/processed/splits \
  --meta_dir data/processed/metadata \
  --model_arch ViT-B/16
```

---

##  Notes

* All training stages save logs and checkpoints separately
* Fine-tuned models from Stage 2 can be reused as warm-starts for Stage 3
* Evaluation is done using query-gallery retrieval over 10 splits

---

##  Summary Table

| Stage | Purpose                          | Encoder(s) Tuned       | Key Loss         |
| ----- | -------------------------------- | ---------------------- | ---------------- |
| 1     | Zero-shot baseline               | None (pretrained CLIP) | N/A              |
| 2     | Image encoder fine-tuning        | Visual only            | Triplet + Center |
| 3     | Joint fine-tuning (image + text) | Both image & text      | SupCon           |

---

##  Best Practices

* Run scripts from project root or set PYTHONPATH
* Use cleaned prompts (max 77 tokens) for v4 training
* Keep at least 2 images per identity for SupCon batches
* Use CosineAnnealingLR with warmup for stability

---
