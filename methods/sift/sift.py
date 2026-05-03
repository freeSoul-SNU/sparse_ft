import os
import hashlib
import torch
import torch.nn as nn
import numpy as np

IDX_CACHE_DIR = os.environ.get(
    "SIFT_IDX_CACHE_DIR",
    os.path.join(
        os.environ.get("RAPA_HOME", "/data/nksol0405/LLM/rapa"),
        "checkpoints",
        ".sift_idx_cache",
    ),
)


def _cache_key(model_name, sparse_rate, sparse_modules, seed=42, dataset_name=""):
    raw = f"{model_name}_{sparse_rate}_{sorted(sparse_modules)}_{seed}_{dataset_name}"
    return hashlib.md5(raw.encode()).hexdigest()


class SIFT():
    """DeepSpeed-compatible SIFT. Frozen model params + trainable sparse_params."""

    def __init__(self, model, sparse_rate, sparse_module, exception=[], grad_acc=1,
                 gradient_checkpointing=False, model_name="", seed=42, dataset_name="") -> None:
        self.model = model
        self.total_num = 0
        self.sparse_rate = sparse_rate
        self.sparse_module = sparse_module
        self.exception = exception
        self.sparse_mapping = dict()  # name -> sparse_param
        self.sparse_indices = dict()  # name -> flat_idx (torch.LongTensor)

        # Cache path (.pt for fast load)
        self._cache_path = None
        if model_name:
            os.makedirs(IDX_CACHE_DIR, exist_ok=True)
            key = _cache_key(model_name, sparse_rate, sparse_module, seed, dataset_name)
            self._cache_path = os.path.join(IDX_CACHE_DIR, f"{key}.pt")

        self.register_sparse_param(seed=seed)

    def register_sparse_param(self, seed=42):
        # Try loading cached indices (torch.load is fast)
        cached = None
        if self._cache_path and os.path.exists(self._cache_path):
            try:
                cached = torch.load(self._cache_path, weights_only=True)
                print(f"[SIFT] Loaded cached indices from {self._cache_path} ({len(cached)} entries)")
            except Exception as e:
                print(f"[SIFT] Cache load failed: {e}")
                cached = None

        idx_to_save = {}

        for n, p in self.model.named_parameters():
            self.total_num += p.numel()
            if any(m in n for m in self.sparse_module):
                p.requires_grad = False
                train_num = min(int(self.sparse_rate * p.numel()) + 1, p.numel())

                cache_key = n.replace(".", "_")
                if cached is not None and cache_key in cached:
                    # FAST PATH: use cached indices, skip generation entirely
                    flat_idx = cached[cache_key]
                else:
                    # SLOW PATH (first run only): generate random indices
                    flat_idx = torch.randint(0, p.numel(), (train_num,), dtype=torch.long)
                    idx_to_save[cache_key] = flat_idx

                self.sparse_indices[n] = flat_idx
                sparse_param = nn.Parameter(
                    torch.zeros(len(flat_idx), dtype=p.dtype), requires_grad=True
                )
                self.sparse_mapping[n] = sparse_param

            elif self.exception and any(item in n for item in self.exception):
                p.requires_grad = True
            else:
                p.requires_grad = False

        # Save cache if we generated new indices
        if idx_to_save:
            all_idx = {}
            # Merge with any loaded cache
            if cached:
                all_idx.update(cached)
            all_idx.update(idx_to_save)
            if self._cache_path:
                torch.save(all_idx, self._cache_path)
                print(f"[SIFT] Saved index cache ({len(all_idx)} entries) to {self._cache_path}")

    def get_trainable_num(self):
        return sum(sp.numel() for sp in self.sparse_mapping.values())

    def print_trainable_parameters(self):
        trainable = self.get_trainable_num()
        print(f"trainable params: {trainable:,d} || all params: {self.total_num:,d} || "
              f"trainable%: {100 * trainable / self.total_num:.2f}")
