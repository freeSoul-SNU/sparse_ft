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
                 gradient_checkpointing=False, model_name="", seed=42, dataset_name="",
                 sparse_indices=None) -> None:
        self.model = model
        self.total_num = 0
        self.sparse_rate = sparse_rate
        self.sparse_module = sparse_module
        self.exception = exception
        self.sparse_mapping = dict()  # name -> sparse_param
        self.sparse_indices = dict()  # name -> flat_idx (torch.LongTensor)
        self.provided_sparse_indices = sparse_indices or {}

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
                    if n in self.provided_sparse_indices:
                        flat_idx = self.provided_sparse_indices[n].to(dtype=torch.long, device="cpu")
                    elif p.grad is not None:
                        flat_idx = torch.topk(
                            p.grad.detach().abs().reshape(-1).float().cpu(),
                            k=train_num,
                            largest=True,
                            sorted=False,
                        ).indices
                    else:
                        raise ValueError(
                            "SIFT requires gradient-based sparse indices. Run a calibration "
                            "backward pass first or pass sparse_indices; random selection is disabled."
                        )
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


class HookSIFT:
    """Author-style SIFT hooks with optimizer-visible sparse parameters.

    This keeps the original module weights in place, reads dense gradients in a
    parameter hook, copies only selected gradient entries into sparse parameters,
    and merges sparse optimizer updates back into the original weights after
    each optimizer step.
    """

    def __init__(
        self,
        model,
        sparse_rate,
        sparse_module,
        exception=None,
        grad_acc=1,
        gradient_checkpointing=False,
        sparse_indices=None,
        zero_dense_grad=True,
    ) -> None:
        self.model = model
        self.total_num = 0
        self.sparse_rate = sparse_rate
        self.sparse_module = sparse_module
        self.exception = exception or []
        self.grad_acc = grad_acc
        self.gradient_checkpointing = gradient_checkpointing
        self.provided_sparse_indices = sparse_indices or {}
        self.zero_dense_grad = zero_dense_grad

        self.sparse_mapping = {}
        self.sparse_indices = {}
        self.param_mapping = {}
        self.named_trainable_parameters_list = []
        self.named_parameters_in_optimizer_list = []
        self.sparse_attr_names = []
        self.hook_handles = []
        self.if_get_idx = {}

        self.register_sparse_param()

    @staticmethod
    def _safe_attr_name(name):
        return name.replace(".", "_") + "_sparse"

    def _initial_indices(self, name, param, train_num):
        if name in self.provided_sparse_indices:
            return self.provided_sparse_indices[name].detach().to(dtype=torch.long, device="cpu")
        # Same spirit as the author's initial random index: it is replaced by
        # gradient top-k on the first backward pass when calibration was not
        # provided.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(hashlib.md5(name.encode()).hexdigest()[:8], 16))
        return torch.randperm(param.numel(), generator=generator)[:train_num].to(dtype=torch.long)

    def register_sparse_param(self):
        for n, p in list(self.model.named_parameters()):
            self.total_num += p.numel()
            is_sparse_target = p.ndim == 2 and any(m in n for m in self.sparse_module)
            if is_sparse_target:
                p.requires_grad = True
                train_num = min(int(self.sparse_rate * p.numel()) + 1, p.numel())
                flat_idx = self._initial_indices(n, p, train_num)
                sparse_param = nn.Parameter(p.new_zeros(train_num), requires_grad=True)
                sparse_param.train_num = train_num

                attr_name = self._safe_attr_name(n)
                setattr(self.model, attr_name, sparse_param)
                self.sparse_attr_names.append(attr_name)

                self.sparse_mapping[n] = sparse_param
                self.sparse_indices[n] = flat_idx
                self.param_mapping[n] = p
                self.if_get_idx[n] = n in self.provided_sparse_indices
                self.named_trainable_parameters_list.append((n, p))
                self.named_parameters_in_optimizer_list.append((n + "_sparse", sparse_param))
                self.hook_handles.append(p.register_hook(self._make_sparse_grad_hook(n, p, sparse_param)))
                if hasattr(p, "register_post_accumulate_grad_hook"):
                    self.hook_handles.append(p.register_post_accumulate_grad_hook(self._make_clear_grad_hook()))
            elif self.exception and any(item in n for item in self.exception):
                p.requires_grad = True
                self.named_trainable_parameters_list.append((n, p))
                self.named_parameters_in_optimizer_list.append((n, p))
            elif self.gradient_checkpointing and n == next(self.model.named_parameters())[0]:
                p.requires_grad = True
            else:
                p.requires_grad = False

    def _make_clear_grad_hook(self):
        def hook(param):
            param.grad = None

        return hook

    def _make_sparse_grad_hook(self, name, param, sparse_param):
        def hook(grad):
            if grad is None:
                return None
            with torch.no_grad():
                if not self.if_get_idx[name]:
                    flat_idx = torch.topk(
                        grad.detach().abs().reshape(-1).float().cpu(),
                        k=sparse_param.train_num,
                        largest=True,
                        sorted=False,
                    ).indices.to(dtype=torch.long)
                    self.sparse_indices[name] = flat_idx
                    self.if_get_idx[name] = True
                idx = self.sparse_indices[name].to(device=grad.device, dtype=torch.long)
                sparse_grad = grad.reshape(-1).index_select(0, idx).detach().to(dtype=sparse_param.dtype)
                if sparse_param.grad is None:
                    sparse_param.grad = sparse_grad.clone()
                else:
                    sparse_param.grad.add_(sparse_grad)
            if self.zero_dense_grad:
                return torch.zeros_like(grad)
            return grad

        return hook

    def named_trainable_parameters(self):
        return iter(self.named_trainable_parameters_list)

    def trainable_parameters(self):
        return iter(p for _, p in self.named_trainable_parameters_list)

    def named_parameters_in_optimizer(self):
        return iter(self.named_parameters_in_optimizer_list)

    def parameters_in_optimizer(self):
        return iter(p for _, p in self.named_parameters_in_optimizer_list)

    def get_trainable_num(self):
        return sum(p.numel() for p in self.parameters_in_optimizer())

    def print_trainable_parameters(self):
        trainable = self.get_trainable_num()
        print(
            f"trainable params: {trainable:,d} || all params: {self.total_num:,d} || "
            f"trainable%: {100 * trainable / self.total_num:.2f}"
        )

    def merge_sparse_updates(self):
        with torch.no_grad():
            for name, sparse_param in self.sparse_mapping.items():
                param = self.param_mapping[name]
                if sparse_param.numel() == 0:
                    continue
                idx = self.sparse_indices[name].to(device=param.device, dtype=torch.long)
                param.data.reshape(-1).index_add_(0, idx, sparse_param.data.to(dtype=param.dtype, device=param.device))
                sparse_param.data.zero_()
                sparse_param.grad = None
                param.grad = None

    def remove_sparse_parameters(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        for attr_name in self.sparse_attr_names:
            if hasattr(self.model, attr_name):
                delattr(self.model, attr_name)
        self.sparse_attr_names = []
