"""Gap-aware prefix construction and a matched compact sequence encoder.

No validation labels are used here. Catalog birth days are deliberately conservative:
an item must have been observed on a complete day before the prefix cutoff day.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import DAY_US, STRENGTH
from .experiments import ResidualTower


def prefix_arrays(ts, items, actions, target_ts, length=100, history_days=60):
    """Right-padded, as-of prefixes; collapsing a run never consults its future end."""
    cutoff = np.asarray(target_ts, np.int64) - DAY_US
    ends = np.searchsorted(ts, cutoff, side='left')
    run_ends = np.flatnonzero(np.r_[((items[1:] != items[:-1]) |
                                    (actions[1:] != actions[:-1])), True]) if len(ts) else np.array([], int)
    h = np.zeros((len(cutoff), length), np.int32)
    a = np.zeros((len(cutoff), length), np.uint8)
    earliest = np.zeros(len(cutoff), np.int64)
    latest = np.zeros(len(cutoff), np.int64)
    count_before_tail = np.zeros(len(cutoff), np.int32)
    for row, (end, boundary) in enumerate(zip(ends, cutoff)):
        if not end:
            continue
        # Previous completed runs plus the final event known at this cutoff.
        runs = np.r_[run_ends[:np.searchsorted(run_ends, end-1)], end-1]
        runs = runs[ts[runs] >= boundary-history_days*DAY_US]
        count_before_tail[row] = len(runs)
        selected = runs[-length:]
        n = len(selected)
        if n:
            h[row, :n] = items[selected]+1; a[row, :n] = actions[selected]
            earliest[row], latest[row] = ts[selected[0]], ts[selected[-1]]
    return h, a, cutoff, earliest, latest, count_before_tail


class PrefixTower(ResidualTower):
    """Same item tower/action weights/residual output; pooling vs causal attention."""
    def __init__(self, vectors, encoder='pool', length=100):
        super().__init__(vectors, True, 'mean')
        self.item_mlp[2].p = 0.
        self.prefix_encoder, self.max_length = encoder, length
        if encoder == 'sequence':
            self.positions = nn.Embedding(length, 64)
            layer = nn.TransformerEncoderLayer(64, 2, 128, dropout=.1,
                       activation='gelu', batch_first=True, norm_first=True)
            self.attention = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
            self.output_norm = nn.LayerNorm(64)
        elif encoder != 'pool':
            raise ValueError(encoder)

    def encode_positions(self, history, actions):
        valid = history.ne(0)
        weights = torch.as_tensor(STRENGTH, device=history.device)[actions]*valid
        tokens = self.item(history)*weights.unsqueeze(-1)
        if self.prefix_encoder == 'pool':
            return tokens.sum(1)/weights.sum(1, keepdim=True).clamp_min(1e-12)
        safe = valid.clone(); safe[:, 0] = True
        # Item retrieval vectors have unit norm, whereas default positional
        # embeddings have norm~sqrt(d). Balance them before causal attention.
        tokens = tokens*(64**.5) + self.positions(torch.arange(history.shape[1], device=history.device))
        causal = torch.triu(torch.ones(history.shape[1], history.shape[1],
                            dtype=torch.bool, device=history.device), diagonal=1)
        encoded = self.attention(tokens, mask=causal, src_key_padding_mask=~safe)
        return self.output_norm(encoded)

    def user(self, history, actions):
        value = self.encode_positions(history, actions)
        if self.prefix_encoder == 'sequence':
            last = history.ne(0).sum(1).sub(1).clamp_min(0)
            value = value[torch.arange(len(history), device=history.device), last]
        return F.normalize(value+self.user_mlp(value), dim=-1)


def save_last(path, model, optimizer, rng, progress):
    """Atomic resumable checkpoint, including both CPU/MPS RNG streams."""
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                 numpy_rng_json=json.dumps(rng.bit_generator.state),
                 torch_rng=torch.get_rng_state(), progress=progress,
                 mps_rng=torch.mps.get_rng_state() if torch.backends.mps.is_available() else None)
    path = Path(path); temporary = path.with_suffix('.tmp')
    torch.save(state, temporary); temporary.replace(path)


def load_last(path, model, optimizer, rng):
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
    rng.bit_generator.state = json.loads(state['numpy_rng_json'])
    torch.set_rng_state(state['torch_rng'])
    if state['mps_rng'] is not None:
        torch.mps.set_rng_state(state['mps_rng'])
    return state['progress']
