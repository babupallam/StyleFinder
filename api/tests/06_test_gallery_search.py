import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# === Updated Configuration ===
GALLERY_FEATS = "data/processed/clip_features/cropped/gallery.pt"
QUERY_FEATS = "data/processed/clip_features/cropped/query.pt"
TOP_K = 5


def load_features(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file missing: {path}")
    data = torch.load(path)
    if isinstance(data, dict) and "features" in data:
        return data["features"]
    return data


def search_top_k(query_feats, gallery_feats, top_k=5):
    query_feats = query_feats / query_feats.norm(dim=1, keepdim=True)
    gallery_feats = gallery_feats / gallery_feats.norm(dim=1, keepdim=True)
    scores = cosine_similarity(query_feats, gallery_feats)
    top_k_indices = np.argsort(-scores, axis=1)[:, :top_k]
    return top_k_indices, scores


def test_gallery_retrieval():
    query = load_features(QUERY_FEATS)
    gallery = load_features(GALLERY_FEATS)

    print(f"Loaded query features: {query.shape}")
    print(f"Loaded gallery features: {gallery.shape}")

    top_k_indices, scores = search_top_k(query, gallery, top_k=TOP_K)

    print("\nTop-k indices for first 3 queries:")
    for i in range(min(3, len(top_k_indices))):
        print(f"Query {i}: {top_k_indices[i]} | Scores: {scores[i][top_k_indices[i]]}")

    assert top_k_indices.shape == (len(query), TOP_K), "Top-k output shape mismatch"
    print("\nGallery search test passed.")


def main():
    assert os.path.exists(GALLERY_FEATS), "Gallery features missing"
    assert os.path.exists(QUERY_FEATS), "Query features missing"
    test_gallery_retrieval()


if __name__ == "__main__":
    main()
