import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _p(relative_path: str) -> str:
    return os.path.normpath(os.path.join(BASE_DIR, relative_path))

MODEL_CONFIGS = {
    "ViT-B/16": {
        "baseline": {
            "clip_arch": "ViT-B/16",
            "checkpoint": None,
            "use_finetuned": False,
            "gallery_feats": _p("../../data/processed/features_clip_vitb16/gallery.pt"),
            "description": {
        "Stage": "1 – Baseline (zero-shot, no fine-tune)",
        "Script": "N/A (pre-trained OpenAI weights)",
        "Epochs": "0",
        "Batch Size": "N/A",
        "Sampler": "N/A",
        "Learning Rate": "N/A",
        "Scheduler": "N/A",
        "Loss": "N/A",
        "Temperature": "N/A",
        "Early Stop": "N/A",
        "Best Metrics": "N/A",
        "Checkpoint Date": "N/A",
        "Remarks": "Control model – original CLIP ViT-B/16 weights used out-of-the-box."
    }
        },
        "finetuned_stage2": {
            "clip_arch": "ViT-B/16",
            "checkpoint": _p("../../model/stage2_img_encoder_finetune/checkpoints/vitb16_subset_ep_10_20250419_175901.pth"),
            "use_finetuned": True,
            "gallery_feats": _p("../../data/processed/vitb16_subset_ep_10_20250419_175901/gallery.pt"),
            "description": {
                "Stage": "2 – Image-encoder fine-tune (text frozen)",
                "Script": "train_clipreid_stages.py → train_clipreid_image_stage()",
                "Epochs": "≤50 (early-stop on Rank-1)",
                "Batch Size": "identity-balanced ReID batches",
                "Sampler": "ReID PK (multi-view per ID)",
                "Learning Rate": "5e-5",
                "Scheduler": "CosineAnnealingLR + warm-up",
                "Loss": "BNNeck + ArcFace + Triplet + Center (λ = 0.0001)",
                "Temperature": "N/A",
                "Early Stop": "Rank-1 patience 10",
                "Best Metrics": "Rank-1 15.2 %, mAP 28.8 % (dorsal-R split)",
                "Checkpoint Date": "2025-04-19",
                "Remarks": "Text encoder frozen; projection normalisation & ArcFace-confidence logging enabled."
            }
        },
        "stage3_joint_v1": {
            "clip_arch": "ViT-B/16",
            "checkpoint": _p("../../model/stage3_joint_img_text_finetune/checkpoints/clip_joint_20250421_004034_BEST.pth"),
            "use_finetuned": True,
            "gallery_feats": _p("../../data/processed/clip_joint_20250421_004034_BEST/gallery.pt"),
            "description": {
        "Stage": "3 v1 – First joint fine-tune",
        "Script": "train_clip_joint_encoders_v1.py",
        "Epochs": "10",
        "Batch Size": "32",
        "Sampler": "Random (1 image/ID)",
        "Learning Rate": "5e-5",
        "Scheduler": "None",
        "Loss": "InfoNCE",
        "Temperature": "0.07",
        "Early Stop": "val_loss patience 3",
        "Best Metrics": "N/A",
        "Checkpoint Date": "2025-04-21",
        "Remarks": "SupCon skipped – batches lacked positive pairs for each identity."
    }
        },
        "stage3_joint_v2": {
            "clip_arch": "ViT-B/16",
            "checkpoint": _p("../../model/stage3_joint_img_text_finetune/checkpoints/clip_joint_20250421_041851_BEST.pth"),
            "use_finetuned": True,
            "gallery_feats": _p("../../data/processed/clip_joint_20250421_041851_BEST/gallery.pt"),
             "description": {
        "Stage": "3 v2 – Joint fine-tune with SupCon + PK sampler",
        "Script": "train_clip_joint_encoders_v2.py",
        "Epochs": "20",
        "Batch Size": "32 (P = 16, K = 2)",
        "Sampler": "PK (16 IDs × 2 images)",
        "Learning Rate": "5e-5",
        "Scheduler": "None",
        "Loss": "SupCon",
        "Temperature": "0.07",
        "Early Stop": "val_loss patience 3",
        "Best Metrics": "N/A",
        "Checkpoint Date": "2025-04-21",
        "Remarks": "Dataset cleaned to guarantee ≥2 images/ID; first stable SupCon run."
    }
        },
        "stage3_joint_v3": {
            "clip_arch": "ViT-B/16",
            "checkpoint": _p("../../model/stage3_joint_img_text_finetune/checkpoints/clip_joint_20250421_061620_BEST.pth"),
            "use_finetuned": True,
            "gallery_feats": _p("../../data/processed/clip_joint_20250421_061620_BEST/gallery.pt"),
            "description": {
        "Stage": "3 v3 – Resumed joint fine-tune (higher LR + CE head)",
        "Script": "train_clip_joint_encoders_v2.py (--resume_ckpt)",
        "Epochs": "additional 10 (total ≈ 30)",
        "Batch Size": "32 (P = 16, K = 2)",
        "Sampler": "PK",
        "Learning Rate": "1e-4",
        "Scheduler": "None",
        "Loss": "SupCon + CrossEntropy",
        "Temperature": "0.07",
        "Early Stop": "val_loss patience 3",
        "Best Metrics": "Top-1 82.4 % accuracy",
        "Checkpoint Date": "2025-04-19",
        "Remarks": "Higher LR accelerates convergence; monitor over-fit when transferring."
    }
        },
        "stage3_joint_v4": {
            "clip_arch": "ViT-B/16",
            "checkpoint": _p("../../model/stage3_joint_img_text_finetune/checkpoints/clip_joint_20250421_155346_BEST.pth"),
            "use_finetuned": True,
            "gallery_feats": _p("../../data/processed/clip_joint_20250421_155346_BEST/gallery.pt"),
             "description": {
        "Stage": "3 v4 – Joint fine-tune with cosine LR + prompt cleanup",
        "Script": "train_clip_joint_encoders.py",
        "Epochs": "50",
        "Batch Size": "32 (P = 16, K = 2)",
        "Sampler": "PK",
        "Learning Rate": "5e-5",
        "Scheduler": "Warm-up → CosineAnnealingLR",
        "Loss": "SupCon",
        "Temperature": "0.05",
        "Early Stop": "val_loss patience 10",
        "Best Metrics": "N/A",
        "Checkpoint Date": "2025-04-21",
        "Remarks": "Prompts trimmed to ≤ 77 tokens; cosine schedule tends to improve final mAP."
    }
        }
    },

    "RN50": {
        "baseline": {
            "clip_arch": "RN50",
            "checkpoint": None,
            "use_finetuned": False,
            "gallery_feats": _p("../../data/processed/features_clip_rn50/gallery.pt"),
            "description": {
                "Stage": "1 – Baseline (zero-shot, no fine-tune)",
                "Script": "N/A (pre-trained OpenAI weights)",
                "Epochs": "0",
                "Batch Size": "N/A",
                "Sampler": "N/A",
                "Learning Rate": "N/A",
                "Scheduler": "N/A",
                "Loss": "N/A",
                "Temperature": "N/A",
                "Early Stop": "N/A",
                "Best Metrics": "N/A",
                "Checkpoint Date": "N/A",
                "Remarks": "Control model – original CLIP RN50 weights used out-of-the-box."
            }
        },
        "finetuned_stage2": {
            "clip_arch": "RN50",
            "checkpoint": _p("../../model/stage2_img_encoder_finetune/checkpoints/rn50_subset_ep_10_20250420_222641.pth"),
            "use_finetuned": True,
            "gallery_feats": _p("../../data/processed/rn50_subset_ep_10_20250420_222641/gallery.pt"),
            "description": {
        "Stage": "2 – Image-encoder fine-tune (text frozen)",
        "Script": "train_clipreid_stages.py → train_clipreid_image_stage()",
        "Epochs": "≤50 (early-stop on Rank-1)",
        "Batch Size": "identity-balanced ReID batches",
        "Sampler": "ReID PK",
        "Learning Rate": "5e-4",
        "Scheduler": "CosineAnnealingLR + warm-up",
        "Loss": "BNNeck + ArcFace + Triplet + Center (λ = 0.0001)",
        "Temperature": "N/A",
        "Early Stop": "Rank-1 patience 10",
        "Best Metrics": "N/A",
        "Checkpoint Date": "2025-04-20",
        "Remarks": "Higher LR fits RN50’s deeper head; same loss stack as ViT variant."
    }
        },
        "stage3_joint_v1": {
            "clip_arch": "RN50",
            "checkpoint": _p("../../model/stage3_joint_img_text_finetune/checkpoints/clip_joint_20250421_020030_BEST.pth"),
            "use_finetuned": True,
            "gallery_feats": _p("../../data/processed/clip_joint_20250421_020030_BEST/gallery.pt"),
             "description": {
        "Stage": "3 v1 – First joint fine-tune",
        "Script": "train_clip_joint_encoders_v1.py",
        "Epochs": "10",
        "Batch Size": "32",
        "Sampler": "Random",
        "Learning Rate": "5e-5",
        "Scheduler": "None",
        "Loss": "InfoNCE",
        "Temperature": "0.07",
        "Early Stop": "val_loss patience 3",
        "Best Metrics": "N/A",
        "Checkpoint Date": "2025-04-21",
        "Remarks": "SupCon skipped for same reason as ViT-B/16 v1."
    }
        },
        "stage3_joint_v2": {
            "clip_arch": "RN50",
            "checkpoint": _p("../../model/stage3_joint_img_text_finetune/checkpoints/clip_joint_20250421_053013_BEST.pth"),
            "use_finetuned": True,
            "gallery_feats": _p("../../data/processed/clip_joint_20250421_053013_BEST/gallery.pt"),
            "description": {
        "Stage": "3 v2 – Joint fine-tune with SupCon + PK sampler",
        "Script": "train_clip_joint_encoders_v2.py",
        "Epochs": "20",
        "Batch Size": "32 (P = 16, K = 2)",
        "Sampler": "PK",
        "Learning Rate": "5e-4",
        "Scheduler": "None",
        "Loss": "SupCon",
        "Temperature": "0.07",
        "Early Stop": "val_loss patience 3",
        "Best Metrics": "N/A",
        "Checkpoint Date": "2025-04-21",
        "Remarks": "Higher LR for RN50; same PK logic as ViT variant."
    }
        },
        "stage3_joint_v3": {
            "clip_arch": "RN50",
            "checkpoint": _p("../../model/stage3_joint_img_text_finetune/checkpoints/clip_joint_20250421_062948_BEST.pth"),
            "use_finetuned": True,
            "gallery_feats": _p("../../data/processed/clip_joint_20250421_062948_BEST/gallery.pt"),
            "description": {
                "Stage": "3 v3 – Resumed joint fine-tune",
                "Script": "train_clip_joint_encoders_v2.py (--resume_ckpt)",
                "Epochs": "additional 10",
                "Batch Size": "32 (P = 16, K = 2)",
                "Sampler": "PK",
                "Learning Rate": "1e-5",
                "Scheduler": "None",
                "Loss": "SupCon + CrossEntropy",
                "Temperature": "0.05",
                "Early Stop": "val_loss patience 5",
                "Best Metrics": "N/A",
                "Checkpoint Date": "2025-04-21",
                "Remarks": "Lower LR and cooler temperature stabilise RN50 continuation."
            }
        },
         "stage3_joint_v4": {
             "clip_arch": "RN50",
             "checkpoint": _p("../../model/stage3_joint_img_text_finetune/checkpoints/clip_joint_20250421_190645_BEST.pth"),  # Not available yet
             "use_finetuned": True,
             "gallery_feats": _p("../../data/processed/clip_joint_20250421_190645_BEST/gallery.pt"),
             "description": {
                 "Stage": "3 v4 – Joint fine-tune with cosine LR + prompt cleanup",
                 "Script": "train_clip_joint_encoders.py",
                 "Epochs": "50",
                 "Batch Size": "32 (P = 16, K = 2)",
                 "Sampler": "PK",
                 "Learning Rate": "5e-5",
                 "Scheduler": "Warm-up → CosineAnnealingLR",
                 "Loss": "SupCon",
                 "Temperature": "0.07",
                 "Early Stop": "val_loss patience 10",
                 "Best Metrics": "N/A",
                 "Checkpoint Date": "2025-04-21",
                 "Remarks": "Prompt trimming respects CLIP’s 77-token cap; cosine schedule smooths training dynamics."
             }
         },
    }
}
