import torch
import numpy as np
from sklearn.metrics import average_precision_score

def cosine_similarity(query_feats, gallery_feats):
    # Normalize for cosine similarity
    query_feats = torch.nn.functional.normalize(query_feats, dim=1)
    gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1)
    return query_feats @ gallery_feats.T  # [Nq x Ng]

def compute_map(similarity, query_ids, gallery_ids):
    similarity_np = similarity.cpu().numpy()
    ap_scores = []
    for i in range(similarity_np.shape[0]):
        sims = similarity_np[i]
        sorted_indices = np.argsort(sims)[::-1]
        # Gather gallery ids as a list (strings)
        sorted_gallery_ids = [gallery_ids[j] for j in sorted_indices]
        # Create binary vector: 1 if match, 0 otherwise
        true_matches = [1 if sorted_gallery_ids[j] == query_ids[i] else 0 for j in range(len(sorted_gallery_ids))]
        # If no true match found, AP = 0
        if sum(true_matches) == 0:
            ap = 0.0
        else:
            ap = average_precision_score(true_matches, sims[sorted_indices])
        ap_scores.append(ap)
    return np.mean(ap_scores)

def compute_rank_k(similarity, query_ids, gallery_ids, k_list=[1, 5, 10]):
    similarity_np = similarity.cpu().numpy()
    ranks = {k: 0 for k in k_list}
    num_queries = similarity_np.shape[0]
    for i in range(num_queries):
        sims = similarity_np[i]
        sorted_indices = np.argsort(sims)[::-1]
        sorted_gallery_ids = [gallery_ids[j] for j in sorted_indices]
        for k in k_list:
            if query_ids[i] in sorted_gallery_ids[:k]:
                ranks[k] += 1
    # Normalize by number of queries
    for k in ranks:
        ranks[k] /= num_queries
    return ranks
