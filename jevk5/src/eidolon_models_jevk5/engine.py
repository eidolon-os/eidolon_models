"""Official prompt/readout with bounded last-block low-rank adaptation."""
import json
from pathlib import Path

import torch
from torch import nn

from .data import payload,digest
from .vendor.jevk5.runtime import JevK5


class LowRankLinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=5 ** .5)

    def forward(self, x):
        delta = (x.float() @ self.a.T) @ self.b.T
        return self.base(x) + (delta * self.scale).to(x.dtype)


class Engine:
    def __init__(self, source, device='mps', adapter=None):
        self.device = device
        self.base_sha256 = digest(Path(source) / 'model.safetensors')
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable in this execution context')
        torch.set_num_threads(4)
        dtype = torch.float16 if device != 'cpu' else torch.float32
        # Direct MPS device-map loading segfaulted in the comparison; stage on CPU.
        self.runtime = JevK5(str(source), device='cpu', dtype=dtype, graphs=False)
        self.runtime.model.to(device)
        self.runtime.device = device
        self.runtime.slot_weight = self.runtime.model.lm_head.weight[self.runtime.slots].detach().contiguous()
        self.runtime.model.requires_grad_(False)
        self.adapted = False
        self.targets = []
        if adapter:
            cfg = json.loads((Path(adapter) / 'adapter.json').read_text())
            if cfg.get('base_sha256') != self.base_sha256:
                raise ValueError('adapter base weights mismatch')
            self.add_adapter(cfg['rank'], cfg['alpha'])
            if cfg['targets'] != self.targets:
                raise ValueError('adapter architecture mismatch')
            from safetensors.torch import load_file
            state = load_file(str(Path(adapter) / 'adapter.safetensors'))
            params = dict(self.runtime.model.named_parameters())
            if set(state) != {n for n,p in params.items() if p.requires_grad}:
                raise ValueError('adapter parameter mismatch')
            with torch.no_grad():
                for name, tensor in state.items():
                    params[name].copy_(tensor)

    def add_adapter(self, rank=8, alpha=16):
        if self.adapted:
            raise ValueError('adapter already installed')
        layers = self.runtime.model.model.layers
        idx = len(layers) - 1
        attention = layers[idx].self_attn
        for name in ('q_proj', 'v_proj', 'o_proj'):
            module = getattr(attention, name)
            setattr(attention, name, LowRankLinear(module, rank, alpha))
            self.targets.append(f'model.layers.{idx}.self_attn.{name}')
        self.rank, self.alpha, self.adapted = rank, alpha, True

    def encode(self, row):
        state, q, options = payload(row)
        ids = self.runtime.encode(state, q['instructions'], [v for _,v in options])
        if len(ids) > 4096 or self.runtime.tok.unk_token_id in ids:
            raise ValueError('lossy or over-budget input')
        return ids, [k for k,_ in options]

    def logits(self, ids, count):
        tensor = torch.tensor([ids], device=self.device)
        hidden = self.runtime.model.model(input_ids=tensor, use_cache=False).last_hidden_state[0,-1]
        return (hidden @ self.runtime.slot_weight[:count].T).float() / self.runtime.temperature

    def predict(self, row):
        ids, names = self.encode(row)
        with torch.inference_mode():
            probs = self.logits(ids, len(names)).softmax(-1).cpu().tolist()
        return dict(zip(names, probs)), len(ids)

    def save_adapter(self, path, extra=None):
        from safetensors.torch import save_file
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        state = {n:p.detach().cpu().contiguous() for n,p in self.runtime.model.named_parameters() if p.requires_grad}
        save_file(state, str(path / 'adapter.safetensors'))
        (path / 'adapter.json').write_text(json.dumps(dict(rank=self.rank, alpha=self.alpha,base_sha256=self.base_sha256,
            targets=self.targets, parameters=sum(p.numel() for p in state.values()), **(extra or {})), indent=2))


def acceptable_set_loss(logits, gold_indices):
    return torch.logsumexp(logits, dim=-1) - torch.logsumexp(logits[gold_indices], dim=-1)
