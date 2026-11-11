"""Training-label weights; these are not user/item input features."""
import numpy as np
import torch


def user_week_weights(users, anchors):
    keys=np.column_stack([users,anchors])
    groups,inverse,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
    raw=1./counts[inverse]
    weights=(raw/raw.mean()).astype(np.float32)
    return weights,inverse,counts,groups


def row_partition_losses(scores, probabilities, known_mask):
    """Same exact-positive / importance-negative partition as the E2 recipe."""
    corrected=(scores[:,1:]-torch.log((scores.shape[1]-1)*probabilities)).masked_fill(known_mask,-torch.inf)
    return torch.logsumexp(torch.cat([scores[:,:1],corrected],1),1)-scores[:,0]


def fixed_weight_mean(losses, weights):
    # Global train normalization is already in weights. No random batch denominator.
    return (losses*weights).mean()
