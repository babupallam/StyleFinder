
# 🧠 CLIP Fashion Search Project

This project implements AI-powered visual search for fashion retail using OpenAI's CLIP and fine-tuned variants. It includes zero-shot evaluation, image encoder fine-tuning (Stage 2), and joint image-text fine-tuning (Stage 3).

---
## 🔧 1. Environment Setup & Server Launch

### ✅ Step 1: Activate Virtual Environment

Activate your Python virtual environment (if not already active):

```powershell
. .\.venv\Scripts\Activate.ps1
````

> If `.venv` does not exist, create it with:
>
> ```bash
> python -m venv .venv
> ```

Install project dependencies:

```bash
pip install -r requirements.txt
```

---

### 🚀 Step 2: Launch the Server (Backend + Frontend)

Use the provided runner script to start the server environment.

```bash
python run_env.py
```

This script will:

* Automatically activate your `.venv`
* Start the FastAPI backend using Uvicorn on `http://127.0.0.1:8000`
* Optionally open or guide you to start the React frontend (on `http://localhost:3000`)

> Ensure `run_env.py` contains logic to:
>
> * Start the backend (`uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000`)
> * Change into the frontend directory and run `npm start` (optional)

---

After this, you should be able to:

* Test APIs at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Use the full visual search interface at: [http://localhost:3000](http://localhost:3000)

```
---

## 📦 2. Dataset Preparation

Prepare the In-Shop fashion dataset and generate metadata.

```bash
python data/prepare_inshop_fashion_dataset.py --out_dir splits
```

This creates:

* Train / Val / Query / Gallery splits
* `image_texts.json`, `image_paths.json`, `image_splits.json`

---

## 📤 3. Feature Extraction

### ✅ Using Official CLIP Models

```bash
python data/extract_clip_features.py \
  --img_dir data/processed/splits \
  --out_dir data/processed/features_clip_rn50 \
  --model RN50

python data/extract_clip_features.py \
  --img_dir data/processed/splits \
  --out_dir data/processed/features_clip_vitb16 \
  --model ViT-B/16
```

### 🔁 Using Fine-Tuned Models

```bash
python data/extract_finetuned_features.py \
  --img_dir data/processed/splits \
  --out_dir data/processed/features_finetuned \
  --ckpt checkpoints/clip_finetuned.pth \
  --batch_size 64
```

---

## 🧠 4. Stage 2 – Fine-Tuning Image Encoder (Text Frozen)

### 🏋️ Training

```bash
python train/stage2_img_encoder_finetune/train_clip_img_encoder_finetune.py \
  --image_dir data/processed/splits/train/img \
  --text_json data/processed/metadata/image_texts.json \
  --max_samples 100% \
  --model ViT-B/16 \
  --batch_size 32 \
  --epochs 2 \
  --lr 5e-5 \
  --temperature 0.07 \
  --loss supcon \
  --seed 42 \
  --save_path train/stage2_img_encoder_finetune/checkpoints/vitb16_subset \
  --log_dir train/stage2_img_encoder_finetune/logs/vitb16_subset
```

### ✅ Evaluation (Coming Soon)

```bash
python eval/stage2_img_encoder_finetune/evaluate_img_encoder_finetune.py \
  --image_dir data/processed/splits \
  --metadata data/processed/metadata/image_paths.json \
  --checkpoint train/stage2_img_encoder_finetune/checkpoints/vitb16_subset_ep2_bs32_lr5e-05.pth \
  --model ViT-B/16 \
  --log_dir eval/stage2_img_encoder_finetune/logs \
  --split_tag vitb16_supcon
```

---

## 🔁 5. Stage 3 – Joint Fine-Tuning (Image + Text Encoder)

### 🏋️ Training

```bash
python train/stage3_joint_img_text_finetune/train_clip_joint_encoders.py \
  --image_dir data/processed/splits \
  --image_meta data/processed/metadata/image_paths.json \
  --prompt_json data/processed/metadata/image_prompts_per_identity.json \
  --model ViT-B/16 \
  --batch_size 32 \
  --epochs 2 \
  --lr 5e-5 \
  --loss supcon \
  --save_path train/stage3_joint_img_text_finetune/checkpoints/vitb16_joint \
  --log_dir train/stage3_joint_img_text_finetune/logs/vitb16_joint_logs
```

### ✅ Evaluation

```bash
python eval/stage3_joint_img_text_finetune/evaluate_joint_clip_encoders.py \
  --image_dir data/processed/splits \
  --prompt_json data/processed/metadata/image_prompts_per_identity.json \
  --metadata data/processed/metadata/image_paths.json \
  --checkpoint train/stage3_joint_img_text_finetune/checkpoints/vitb16_joint_ep3_bs32_lr5e-05.pth \
  --model ViT-B/16 \
  --log_dir eval/stage3_joint_img_text_finetune/logs \
  --split_tag vitb16_joint_eval
```

---

## 📌 Notes

* Ensure all features, checkpoints, and metadata paths are correctly specified.
* Prompt-based training (Stage 3) uses grouped prompts per item ID.
* Adjust --epochs, --batch\_size, or --loss function to experiment.

---
