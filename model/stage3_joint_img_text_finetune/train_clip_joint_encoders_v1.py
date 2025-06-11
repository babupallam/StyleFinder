import os
import sys
import json
import argparse
from tqdm import tqdm
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

import clip  # OpenAI CLIP

# Allow importing loss.py from the same directory
sys.path.append(os.path.dirname(__file__))
from loss import SupConLoss, InfoNCELoss



class IdentityPromptDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, image_meta_path, identity_prompts_path, transform, max_samples=None, seed=42,split="train"):
        with open(image_meta_path, "r") as f:
            image_meta = json.load(f)

        with open(identity_prompts_path, "r") as f:
            id_prompts = json.load(f)
        self.split = split
        self.samples = []
        for rel_path, meta in image_meta.items():
            if meta["split"] != self.split:
                continue
            item_id = meta["item_id"]
            if item_id not in id_prompts:
                continue

            path = os.path.join(image_dir, self.split, "img", rel_path[4:] if rel_path.startswith("img/") else rel_path)
            if os.path.exists(path):
                self.samples.append((path, id_prompts[item_id]))

        # Optional subsampling
        if max_samples:
            if isinstance(max_samples, str) and max_samples.endswith('%'):
                pct = float(max_samples.strip('%')) / 100.0
                n = int(len(self.samples) * pct)
            else:
                n = int(max_samples)
            torch.manual_seed(seed)
            self.samples = self.samples[:n]

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, text = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), text


def get_loss_function(name, temperature=0.07):
    if name == "supcon":
        return SupConLoss(temperature)
    elif name == "infonce":
        return InfoNCELoss(temperature)
    else:
        raise ValueError(f"Unsupported loss function: {name}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_loss = float("inf")
    best_ckpt_path = ""
    patience_counter = 0

    print(f"\nStage 3: Joint Training  CLIP {args.model}")
    print(f"Images: {args.image_dir}")
    print(f"Metadata: {args.image_meta}")
    print(f"Prompts: {args.prompt_json}")

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"train_log_{timestamp}.txt")

    model, preprocess = clip.load(args.model, device=device)
    model.float() # for GPU
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    # Dataset
    dataset = IdentityPromptDataset(
        image_dir=args.image_dir,
        image_meta_path=args.image_meta,
        identity_prompts_path=args.prompt_json,
        transform=preprocess,
        max_samples=args.max_samples,
        seed=args.seed
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    val_dataset = IdentityPromptDataset(
        image_dir=args.image_dir,
        image_meta_path=args.image_meta,
        identity_prompts_path=args.prompt_json,
        transform=preprocess,
        split=args.val_split,
        seed=args.seed
    )
    print(f"Train samples: {len(dataset)} | Val samples: {len(val_dataset)}")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)


    print(f"Dataset: {len(dataset)} | BS: {args.batch_size} | Epochs: {args.epochs} | LR: {args.lr} | Loss: {args.loss}")
    loss_fn = get_loss_function(args.loss, args.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def validate():
        model.eval()
        losses = []
        skipped_batches = 0
        with torch.no_grad():

            for images, texts in val_loader:
                images = images.to(device)
                texts = [t[:300] for t in texts]
                text_tokens = clip.tokenize(texts, truncate=True).to(device)

                i_feats = model.encode_image(images)
                t_feats = model.encode_text(text_tokens)
                i_feats = nn.functional.normalize(i_feats, dim=1)
                t_feats = nn.functional.normalize(t_feats, dim=1)

                feats = torch.cat([i_feats, t_feats], dim=0)
                labels = torch.arange(len(images)).repeat(2).to(device)
                loss = loss_fn(feats, labels)
                if not torch.isfinite(loss):
                    print(f" Skipping batch due to non-finite loss (loss={loss.item()})")
                    skipped_batches += 1
                    continue


                losses.append(loss.item())

        model.train()
        return np.mean(losses), skipped_batches

    with open(log_file, "w") as log:
        log.write(f"Stage 3 Joint Training Log  {args.model}\n")
        log.write(f"Start: {timestamp}\n")
        log.write(f"Images: {args.image_dir}\n")
        log.write(f"Metadata: {args.image_meta}\n")
        log.write(f"Prompts: {args.prompt_json}\n\n")

        best_loss = float("inf")
        best_ckpt_path = ""

        for epoch in range(args.epochs):
            epoch_losses = []
            sim_scores = []
            grad_norms = []

            skipped_batches = 0
            for images, texts in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
                '''
                try:
                    # Show unique item_id count in current batch
                    item_ids = set([text.strip().lower() for text in texts])
                    if len(item_ids) == len(texts):
                        print("Batch Warning: All item_ids are unique  no positive pairs for SupCon")
                    else:
                        
                        print(f"Batch OK: {len(item_ids)} unique prompts in batch of {len(texts)}")

                    # Show a few sample prompts
                    print("Sample Prompts:")
                    for i in range(min(3, len(texts))):
                        print(f"  [{i}] {texts[i][:80]}...")

                except Exception as e:
                    print(f"Error in batch debug: {e}")
                '''
                images = images.to(device)

                # Manual truncate
                MAX_TOKENS = 77
                texts = [text[:MAX_TOKENS * 4] for text in texts]  # Roughly safe cutoff (~77 tokens)
                text_tokens = clip.tokenize(texts, truncate=True).to(device)  # if using newer clip

                image_feats = model.encode_image(images)
                text_feats = model.encode_text(text_tokens)

                image_feats = nn.functional.normalize(image_feats, dim=1)
                text_feats = nn.functional.normalize(text_feats, dim=1)

                sims = torch.sum(image_feats * text_feats, dim=1).detach().cpu().numpy()
                sim_scores.extend(sims)

                features = torch.cat([image_feats, text_feats], dim=0)
                labels = torch.arange(len(images)).repeat(2).to(device)

                loss = loss_fn(features, labels)
                optimizer.zero_grad()
                loss.backward()

                grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
                if grad_norm > 1e4:
                    print(f"Large grad norm: {grad_norm:.2f}")

                grad_norms.append(grad_norm)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                optimizer.step()
                epoch_losses.append(loss.item())



            # Epoch stats
            avg_loss = np.mean(epoch_losses)
            sim_avg = np.mean(sim_scores)
            grad_avg = np.mean(grad_norms)

            print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | CosSim: {sim_avg:.4f} | GradNorm: {grad_avg:.4f}")
            log.write(f"Epoch {epoch + 1}/{args.epochs}\n")
            log.write(f"Loss Avg: {avg_loss:.4f} | CosSim: {sim_avg:.4f} | GradNorm: {grad_avg:.4f}\n\n")

            val_loss, skipped_val = validate()
            print(f"Validation complete  Skipped {skipped_val} batches with non-finite loss")
            log.write(f"Validation Loss: {val_loss:.4f}\n")
            print(f"Validation Loss: {val_loss:.4f}\n")

            # Determine which metric to monitor
            monitor = val_loss if args.early_stop_metric == "val_loss" else avg_loss

            # Early stopping logic
            if monitor < best_loss:
                best_loss = monitor
                patience_counter = 0
                best_ckpt_path = f"{args.save_path}_{timestamp}_BEST.pth"
                os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "train_args": vars(args),
                }, best_ckpt_path)
                log.write(f"New best model saved to: {best_ckpt_path} (Loss: {best_loss:.4f})\n\n")
            else:
                patience_counter += 1
                log.write(f"No improvement. Patience {patience_counter}/{args.early_stop_patience}\n\n")
                if patience_counter >= args.early_stop_patience:
                    log.write("Early stopping triggered.\n")
                    print("Early stopping triggered.")
                    break
        print(f"Epoch {epoch + 1} summary: {skipped_batches} batches skipped due to invalid loss")

        # Final model save
        final_ckpt_path = f"{args.save_path}_{timestamp}_FINAL.pth"
        os.makedirs(os.path.dirname(final_ckpt_path), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_name": args.model,
            "train_args": vars(args),
        }, final_ckpt_path)
        log.write(f"Final model saved to: {final_ckpt_path}\n")
        log.write("Training complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--image_meta", type=str, required=True)
    parser.add_argument("--prompt_json", type=str, required=True)
    parser.add_argument("--max_samples", type=str, default=None)
    parser.add_argument("--model", type=str, default="ViT-B/16")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--loss", type=str, default="supcon", choices=["supcon", "infonce"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="model/stage3_joint_img_text_finetune/checkpoints/clip_joint")
    parser.add_argument("--log_dir", type=str, default="model/stage3_joint_img_text_finetune/logs")
    parser.add_argument("--val_split", type=str, default="val", help="Split name for validation set")
    parser.add_argument("--early_stop_metric", type=str, default="val_loss", choices=["val_loss", "train_loss"])
    parser.add_argument("--early_stop_patience", type=int, default=3, help="Stop after N epochs without improvement")

    args = parser.parse_args()

    train(args)
