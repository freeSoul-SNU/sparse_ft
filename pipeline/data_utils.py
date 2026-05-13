"""Dataset helpers matching the LMFlow decoder fine-tuning label contract."""
import glob
import json
import os

from torch.utils.data import Dataset as TorchDataset
from transformers import DataCollatorForSeq2Seq


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _json_candidates(dataset_path):
    if os.path.isfile(dataset_path):
        return [dataset_path]
    return sorted(glob.glob(os.path.join(dataset_path, "*.json")))


def load_lmflow_instances(dataset_path):
    """Load LMFlow-style JSON datasets from either a file or dataset directory."""
    instances = []
    dataset_type = None
    for path in _json_candidates(dataset_path):
        with open(path) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            dataset_type = dataset_type or raw.get("type")
            instances.extend(raw.get("instances", []))
        elif isinstance(raw, list):
            instances.extend(raw)
    return dataset_type, instances


class LMFlowTextDataset(TorchDataset):
    """Text-only causal-LM dataset compatible with LMFlow's HFDecoderModel.tokenize.

    For text-only MMLU data, LMFlow trains on the whole text field, includes an
    attention mask, and masks padded labels with -100.  Dynamic padding is an
    extension used by the original run script; fixed padding reproduces LMFlow's
    default ``disable_group_texts=True`` path.
    """

    def __init__(self, instances, tokenizer, max_len, dynamic_padding=True):
        self.data = []
        self.dynamic_padding = dynamic_padding
        pad_token_id = tokenizer.pad_token_id
        for item in instances:
            text = item.get("text", "")
            enc = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_len,
                padding=False,
            )
            input_ids = list(enc["input_ids"])
            attention_mask = list(enc["attention_mask"])
            labels = input_ids.copy()

            if not dynamic_padding:
                pad_len = max_len - len(input_ids)
                if pad_len > 0:
                    input_ids.extend([pad_token_id] * pad_len)
                    attention_mask.extend([0] * pad_len)
                    labels.extend([-100] * pad_len)
                else:
                    input_ids = input_ids[:max_len]
                    attention_mask = attention_mask[:max_len]
                    labels = labels[:max_len]

            self.data.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def build_lmflow_text_dataset(dataset_path, tokenizer, max_seq_length, dynamic_padding=None):
    """Return a dataset and collator using the same label/padding rules as LMFlow."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if dynamic_padding is None:
        dynamic_padding = _env_bool("RAPA_USE_DYNAMIC_PADDING", True)

    dataset_type, instances = load_lmflow_instances(dataset_path)
    if dataset_type not in {None, "text_only"}:
        raise ValueError(
            f"Only LMFlow text_only datasets are supported here; got type={dataset_type!r}"
        )
    dataset = LMFlowTextDataset(
        instances,
        tokenizer,
        max_len=max_seq_length,
        dynamic_padding=dynamic_padding,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True if dynamic_padding else "max_length",
        max_length=None if dynamic_padding else max_seq_length,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    return dataset, collator, dynamic_padding
