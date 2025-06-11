import os, json, argparse, datetime
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm
from PIL import Image
from datetime import datetime

from dataset import CLIPImageTextDataset
from loss import SupConLoss, InfoNCELoss, CrossEntropyLoss, TripletLoss


def get_loss_function(name, temperature=0.07, margin=0.3):
    if name == "supcon":
        return SupConLoss(temperature)
    elif name == "infonce":
        return InfoNCELoss(temperature)
    elif name == "crossentropy":
        return CrossEntropyLoss()
    elif name == "triplet":
        return TripletLoss(margin)
    else:
        raise ValueError(f"Unsupported loss function: {name}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n Starting Training with CLIP model: {args.model}")
    print(f" Image directory: {args.image_dir}")

    # Create log directory and file
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    log_file = os.path.join(args.log_dir, f"train_log_{timestamp}.txt")
    print(f" Logging to: {log_file}\n")

    # Load model
    model, preprocess = clip.load(args.model, device=device)
    model.float()

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze image encoder
    model.visual.train()
    for param in model.visual.parameters():
        param.requires_grad = True

    # Dataset
    dataset = CLIPImageTextDataset(
        image_dir=args.image_dir,
        image_paths_json=args.image_paths_json,
        transform=preprocess,
        split="train",
        seed=args.seed
    )


    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    print(f"Dataset size: {len(dataset)} | Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs} | Learning Rate: {args.lr} | Loss: {args.loss}")

    loss_fn = get_loss_function(args.loss, args.temperature, args.margin)
    optimizer = torch.optim.AdamW(model.visual.parameters(), lr=args.lr)

    with open(log_file, "w") as log:
        log.write(f"Training Log  {args.model}\n")
        log.write(f"Start Time: {timestamp}\n")
        log.write(f"Image Dir: {args.image_dir}\n")
        log.write(f"Loss Function: {args.loss}\n")
        log.write(f"Learning Rate: {args.lr}\n")
        log.write(f"Batch Size: {args.batch_size}\n")
        log.write(f"Epochs: {args.epochs}\n\n")

        # Training loop
        model.train()
        for epoch in range(args.epochs):
            epoch_losses = []
            sim_scores = []
            grad_norms = []

            for images, texts in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
                images = images.to(device)
                text_tokens = clip.tokenize(texts).to(device)

                with torch.no_grad():
                    text_feats = model.encode_text(text_tokens)  # frozen

                image_feats = model.encode_image(images)
                image_feats = nn.functional.normalize(image_feats, dim=1)
                text_feats = nn.functional.normalize(text_feats, dim=1)

                # Cosine similarity (for stats)
                sims = torch.sum(image_feats * text_feats, dim=1).detach().cpu().numpy()
                sim_scores.extend(sims)

                features = torch.cat([image_feats, text_feats], dim=0)
                labels = torch.arange(len(images)).repeat(2).to(device)

                loss = loss_fn(features, labels)

                optimizer.zero_grad()
                loss.backward()

                # Gradient norm tracking
                grad_norm = sum(p.grad.norm().item() for p in model.visual.parameters() if p.grad is not None)
                grad_norms.append(grad_norm)

                optimizer.step()
                epoch_losses.append(loss.item())

            # Epoch metrics
            avg_loss = np.mean(epoch_losses)
            loss_min = np.min(epoch_losses)
            loss_max = np.max(epoch_losses)
            loss_std = np.std(epoch_losses)
            sim_avg = np.mean(sim_scores)
            sim_std = np.std(sim_scores)
            grad_avg = np.mean(grad_norms)

            # Print + log
            print(f" Epoch {epoch+1} | Loss: {avg_loss:.4f} | CosSim: {sim_avg:.4f} | GradNorm: {grad_avg:.4f}")
            log.write(f"Epoch {epoch+1}/{args.epochs}\n")
            log.write(f"Loss   | Avg: {avg_loss:.4f} | Min: {loss_min:.4f} | Max: {loss_max:.4f} | Std: {loss_std:.4f}\n")
            log.write(f"CosSim | Avg: {sim_avg:.4f} | Std: {sim_std:.4f}\n")
            log.write(f"Grad   | Avg Norm: {grad_avg:.4f}\n\n")


        log.write(" Training completed.\n")

        # === Final full checkpoint ===
        if args.save_path:
            final_ckpt_path = f"{args.save_path}_ep_{epoch+1}_{timestamp}.pth"
            torch.save({
                "model": model,
                "args": vars(args),
                "timestamp": timestamp
            }, final_ckpt_path)
            print(f" Final full model saved to: {final_ckpt_path}")
            log.write(f"\nFinal full model saved: {final_ckpt_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--image_paths_json", type=str, required=True, help="Path to image_paths.json")
    parser.add_argument("--model", type=str, default="ViT-B/16")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--loss", type=str, default="supcon", choices=["supcon", "infonce", "crossentropy", "triplet"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="model/stage2_img_encoder_finetune/checkpoints/clip_imgencoder")
    parser.add_argument("--log_dir", type=str, default="model/stage2_img_encoder_finetune/logs")
    parser.add_argument("--meta_dir", type=str, default="data/processed/metadata", help="Path to metadata (for query-gallery evaluation)")  # <-- NEW
    args = parser.parse_args()

    train(args)
