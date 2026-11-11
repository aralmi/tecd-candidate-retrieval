"""Shared tower components used in the recorded experiments.

Item IDs in tensors are local IDs plus one; zero is padding.
The content buffer is external to checkpoints, as in the original recipe.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
STRENGTH = np.array([0., .1, 1., 2., 3.], dtype=np.float32)

class TwoTower(nn.Module):
    """Общая товарная часть; pooling или causal attention для истории."""

    def __init__(self, vectors, use_content=True, encoder="mean", content_tensor=None):
        super().__init__()
        self.use_content, self.encoder_name = use_content, encoder
        if content_tensor is None:
            content_tensor = (
                torch.from_numpy(np.vstack([np.zeros((1, 300), np.float32), vectors]))
                if use_content
                else torch.zeros((1, 300))
            )
        self.register_buffer("content", content_tensor, persistent=False)
        self.register_buffer("id_seen", torch.zeros(len(vectors) + 1, dtype=torch.bool))
        self.ids = nn.Embedding(len(vectors) + 1, 32, padding_idx=0)
        self.item_mlp = nn.Sequential(
            nn.Linear(333, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64)
        )
        self.actions = nn.Embedding(5, 64, padding_idx=0)
        self.user_mlp = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        if encoder == "attention":
            self.positions = nn.Embedding(50, 64)
            layer = nn.TransformerEncoderLayer(
                64,
                2,
                dim_feedforward=128,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )
            self.attention = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)

    def item(self, ids):
        content = (
            self.content[ids]
            if self.use_content
            else torch.zeros((*ids.shape, 300), device=ids.device)
        )
        present = (content.square().sum(-1, keepdim=True) > 0).float()
        known = ids if self.training else torch.where(self.id_seen[ids], ids, 0)
        value = torch.cat([self.ids(known), content, present], dim=-1)
        return F.normalize(self.item_mlp(value), dim=-1)

    def user(self, history, actions):
        valid = history != 0
        tokens = self.item(history) + self.actions(actions)
        if self.encoder_name == "mean":
            value = (tokens * valid.unsqueeze(-1)).sum(1) / valid.sum(
                1, keepdim=True
            ).clamp_min(1)
            return F.normalize(self.user_mlp(value), dim=-1)
        safe_valid = valid.clone()
        safe_valid[:, 0] = (
            True  # полностью пустая строка не создаёт all-masked attention
        )
        tokens = tokens + self.positions(
            torch.arange(history.shape[1], device=history.device)
        )
        causal = torch.triu(
            torch.ones(
                history.shape[1],
                history.shape[1],
                device=history.device,
                dtype=torch.bool,
            ),
            1,
        )
        encoded = self.attention(tokens, mask=causal, src_key_padding_mask=~safe_valid)
        last = (valid.sum(1) - 1).clamp_min(0)
        value = encoded[torch.arange(len(history), device=history.device), last]
        return F.normalize(self.user_mlp(value), dim=-1)

    def forward(self, h, actions, candidates):
        return (self.user(h, actions).unsqueeze(1) * self.item(candidates)).sum(
            -1
        ) / 0.1

class WeightedTower(TwoTower):
    """Bind each action to its item by a fixed nonnegative action weight."""

    def user(self, history, actions):
        weights = torch.as_tensor(STRENGTH, device=history.device)[actions]
        weights = weights * (history != 0)
        tokens = self.item(history)
        value = (tokens * weights.unsqueeze(-1)).sum(1) / weights.sum(
            1, keepdim=True
        ).clamp_min(1e-12)
        return F.normalize(self.user_mlp(value), dim=-1)

class ResidualTower(WeightedTower):
    """One fixed identity skip; identical parameters/init to WeightedTower."""

    def user(self, history, actions):
        weights = torch.as_tensor(STRENGTH,device=history.device)[actions]*(history!=0)
        profile = (self.item(history)*weights.unsqueeze(-1)).sum(1)/weights.sum(1,keepdim=True).clamp_min(1e-12)
        return F.normalize(profile+self.user_mlp(profile),dim=-1)
