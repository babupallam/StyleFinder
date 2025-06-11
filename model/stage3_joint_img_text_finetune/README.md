
#  Stage 3  Joint Image + Text Fine-Tuning

This folder contains **all experiments that train the CLIP image _and_ text encoders together** using prompt sentences.  
Each sub-version (v1  v4) represents an incremental improvement:

| Ver | Loss            | Sampler | Prompt Prep | LR scheduler      | Notes                           |
|-----|-----------------|---------|-------------|-------------------|---------------------------------|
| v1  | InfoNCE         | default | raw prompts | constant          | baseline joint training         |
| v2  | SupCon (+ PK)  | P  K   | *clean* prompts w/ 2 imgs/ID | constant          | fixes SupCon instability        |
| v3  | SupCon (+ PK)  | P  K   | v2 prompts  | constant (LR)    | **continuous** training (resume)|
| v4  | SupCon (+ PK)  | P  K   | 77-token-safe prompts | CosineAnnealing + warm-up | long-run, token-safe            |

---

##  Folder Layout

```

stage3\_joint\_img\_text\_finetune/
 checkpoints/                    # All BEST / FINAL .pth files
 logs/                           # Pre-processing & training logs
    01\_generate\_prompts\_from\_descriptions.py
    02\_analyze\_prompt\_consistency.py
    03\_filter\_valid\_identities.py
    04\_generate\_structured\_prompts\_for\_v4.py
     train\_log\_*.txt , eval\_log\_*.txt
 loss.py                         # SupCon + InfoNCE implementations
 eval\_clip\_joint.py              # Common evaluation script
 train\_clip\_joint\_encoders.py    # v4 (current main)
 train\_clip\_joint\_encoders\_v1.py # v1 baseline
 train\_clip\_joint\_encoders\_v2.py # v2 + PK SupCon
 train\_clip\_joint\_encoders\_v3.py # v3 resume-able
 train\_clip\_joint\_encoders\_v4.py # v4 (wrapper of main)

````

---

##  Pre-processing Utilities

Run **once** before any SupCon experiment:

```bash
# 0 Generate initial prompts
python logs/01_generate_prompts_from_descriptions.py

# 1 Verify prompt consistency per item-ID
python logs/02_analyze_prompt_consistency.py

# 2 Keep only identities with 2 images (needed for SupCon)
python logs/03_filter_valid_identities.py

# 3 (v4) Strip first sentence to respect 77-token limit
python logs/04_generate_structured_prompts_for_v4.py
````

Outputs:

* `image_prompts_per_identity.json`      (v1 raw)
* `image_prompts_clean.json`             (v2/v3)
* `image_prompts_cleaned_for_v4.json`    (v4 token-safe)
* `image_paths_clean.json`               (filtered paths)

---

##  Version 1  Baseline (InfoNCE)

### ViT-B/16 Training

```bash
python model/stage3_joint_img_text_finetune/train_clip_joint_encoders_v1.py \
  --image_dir data/processed/splits \
  --image_meta data/processed/metadata/image_paths.json \
  --prompt_json data/processed/metadata/image_prompts_per_identity.json \
  --model ViT-B/16 --batch_size 32 --epochs 10 --lr 5e-5 \
  --loss infonce --temperature 0.07 \
  --save_path model/stage3_joint_img_text_finetune/checkpoints/clip_joint \
  --log_dir   model/stage3_joint_img_text_finetune/logs \
  --val_split val --early_stop_metric val_loss --early_stop_patience 3
```

* **Best model**: `clip_joint_20250421_004034_BEST.pth`
* **Train log**: `logs/train_log_20250421_004034.txt`
* **Eval log**: `logs/clip_joint_20250421_004034_BEST/eval_log_*_021255.txt`
* **Features**: `data/processed/clip_joint_20250421_004034_BEST/`

### RN50 Training

(same flags, `--model RN50`)

---

##  Why SupCon Needed PK (Version 2)

* SupCon requires 2 positive samples per class in a batch.
* Implemented **PKSampler** (`P` identities  `K` images each) to guarantee this.
* Run `02_analyze_prompt_consistency.py` + `03_filter_valid_identities.py` beforehand.

---

##  Version 2  SupCon + PKSampler

Training example (ViT-B/16):

```bash
python model/stage3_joint_img_text_finetune/train_clip_joint_encoders_v2.py \
  --image_dir data/processed/splits \
  --image_meta data/processed/metadata/image_paths_clean.json \
  --prompt_json data/processed/metadata/image_prompts_clean.json \
  --model ViT-B/16 --p 16 --k 2 --epochs 20 --lr 5e-5 \
  --loss supcon --temperature 0.07 \
  --save_path model/stage3_joint_img_text_finetune/checkpoints/clip_joint \
  --log_dir   model/stage3_joint_img_text_finetune/logs \
  --val_split val --early_stop_metric val_loss --early_stop_patience 3
```

* **Best model**: `clip_joint_20250421_041851_BEST.pth`
* Features path: `data/processed/clip_joint_20250421_041851_BEST/`

---

##  Version 3  Continuous Training (Resume)

Resume from best v2 checkpoint with higher LR:

```bash
python model/stage3_joint_img_text_finetune/train_clip_joint_encoders_v2.py \
  --image_dir data/processed/splits \
  --image_meta data/processed/metadata/image_paths_clean.json \
  --prompt_json data/processed/metadata/image_prompts_clean.json \
  --model ViT-B/16 --p 16 --k 2 --epochs 20 --lr 1e-4 \
  --loss supcon --temperature 0.07 \
  --resume_ckpt model/stage3_joint_img_text_finetune/checkpoints/clip_joint_20250421_041851_BEST.pth \
  --save_path model/stage3_joint_img_text_finetune/checkpoints/clip_joint \
  --log_dir   model/stage3_joint_img_text_finetune/logs \
  --val_split val --early_stop_metric val_loss --early_stop_patience 3
```

Best checkpoint  `clip_joint_20250421_061620_BEST.pth`

---

##  Version 4  Token-Safe Prompts + Cosine LR

### Prompt Preparation

```bash
python logs/04_generate_structured_prompts_for_v4.py
```

### Training (ViT-B/16)

```bash
python model/stage3_joint_img_text_finetune/train_clip_joint_encoders.py \
  --image_dir data/processed/splits \
  --image_meta data/processed/metadata/image_paths_clean.json \
  --prompt_json data/processed/metadata/image_prompts_cleaned_for_v4.json \
  --model ViT-B/16 --p 16 --k 2 --epochs 50 --lr 5e-5 \
  --loss supcon --temperature 0.05 \
  --save_path model/stage3_joint_img_text_finetune/checkpoints/clip_joint \
  --log_dir   model/stage3_joint_img_text_finetune/logs \
  --val_split val --early_stop_metric val_loss --early_stop_patience 10
```

* CosineAnnealingLR + warm-up enabled inside script.
* **Best model**: `clip_joint_20250421_155346_BEST.pth`
* Features: `data/processed/clip_joint_20250421_155346_BEST/`

### Training (RN50)

Same command but `--model RN50`, temperature 0.07.
Best ckpt  `clip_joint_20250421_190645_BEST.pth`.

---

##  Common Evaluation Command

```bash
python model/stage3_joint_img_text_finetune/eval_clip_joint.py \
  --ckpt <checkpoint.pth> \
  --meta_dir data/processed/metadata \
  --image_dir data/processed/splits \
  --model_arch <ViT-B/16|RN50> \
  --store_path model/stage3_joint_img_text_finetune/logs
```

---

##  Key Takeaways

| Version | Improvement                    | Metric Impact\* |
| ------- | ------------------------------ | --------------- |
| v1      | Baseline InfoNCE               |                |
| v2      | SupCon + PKSampler             |  Rank-1, mAP   |
| v3      | Resume fine-tuning (higher LR) |  further gain  |
| v4      | Token-safe prompts + cosine LR | best so far     |

\* see individual `eval_log_*.txt` for exact numbers.

---

##  Tips

* **SupCon** requires **PK** batches: ensure `K  2`.
* Keep prompts  77 tokens; v4 script enforces this.
* Use `--resume_ckpt` for continual training.
* Logs print skipped batches, grad norms & cosine similarity for quick debugging.

---

