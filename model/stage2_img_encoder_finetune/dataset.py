import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
class CLIPImageTextDataset(Dataset):
    def __init__(self, image_dir, image_paths_json, transform, split="train", seed=42):
        self.image_dir = image_dir
        self.transform = transform
        self.seed = seed

        with open(image_paths_json, 'r') as f:
            data = json.load(f)

        self.samples = []
        for rel_path, meta in data.items():
            if meta["split"] != split:
                continue
            full_path = os.path.join(image_dir, rel_path[4:] if rel_path.startswith("img/") else rel_path)
            if os.path.exists(full_path):
                self.samples.append((rel_path, "a photo of a fashion item"))

        random.seed(seed)
        random.shuffle(self.samples)

        print(f"Loaded {len(self.samples)} image-text pairs from split='{split}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, text = self.samples[idx]
        image_path = os.path.join(self.image_dir, rel_path[4:] if rel_path.startswith("img/") else rel_path)
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), text
