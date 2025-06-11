# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Correct implementation of Supervised Contrastive Loss:
    Handles arbitrary labels and multiple positives per class.
    Reference: https://arxiv.org/abs/2004.11362
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-8

    def forward(self, features, labels):
        """
        Args:
            features: (2N, D) tensor with normalized image+text features.
            labels: (2N,) tensor of item_id-based labels.
        """
        device = features.device
        features = F.normalize(features, dim=1)

        # Cosine similarity matrix
        sim_matrix = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()  # for numerical stability

        # Mask: same class and not self
        labels = labels.contiguous().view(-1, 1)
        match_mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.ones_like(match_mask).fill_diagonal_(0)

        positives_mask = match_mask * logits_mask

        # Compute log-softmax
        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)

        # Average over positive pairs
        mean_log_prob_pos = (positives_mask * log_prob).sum(1) / (positives_mask.sum(1) + self.eps)

        # Final loss
        loss = -mean_log_prob_pos.mean()
        return loss

class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (e.g., CLIP-style loss)
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels=None):
        device = features.device
        batch_size = features.shape[0] // 2
        image_feats, text_feats = features[:batch_size], features[batch_size:]

        # Compute logits
        logits_per_image = torch.matmul(image_feats, text_feats.T) / self.temperature
        logits_per_text = logits_per_image.T

        targets = torch.arange(batch_size, device=device)

        loss_i2t = F.cross_entropy(logits_per_image, targets)
        loss_t2i = F.cross_entropy(logits_per_text, targets)

        return (loss_i2t + loss_t2i) / 2
