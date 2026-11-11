"""CCL from SimpleX, applied to existing normalized tower representations.

Negative sampling remains the frozen past-only proposal. No softmax temperature
or logQ correction belongs in this criterion. Average only valid sampled negatives.
"""
import torch

def cosine_contrastive(cosine,mask,margin,negative_weight):
    assert 0<=margin<1 and negative_weight>0
    valid=~mask;negative=torch.relu(cosine[:,1:]-margin).masked_fill(mask,0.)
    denominator=valid.sum(1).clamp_min(1)
    positive=1-cosine[:,0]
    negative_mean=negative.sum(1)/denominator
    return positive+negative_weight*negative_mean,positive,negative_mean
