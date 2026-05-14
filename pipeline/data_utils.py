"""Dataset helpers for sparse fine-tuning runs."""
import glob
import json
import logging
import os

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import DataCollatorForSeq2Seq

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100


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
        self.truncated_examples = 0
        self.first_1000_max_input_len = 0
        pad_token_id = tokenizer.pad_token_id
        for idx, item in enumerate(instances):
            text = item.get("text", "")
            untruncated = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )["input_ids"]
            if len(untruncated) > max_len:
                self.truncated_examples += 1
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
            if idx < 1000:
                self.first_1000_max_input_len = max(
                    self.first_1000_max_input_len,
                    len(input_ids),
                )

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


class RapaInstructionDataset(TorchDataset):
    """RAPA instruction-tuning label contract with response-only loss.

    The reference instruction-tuning collator builds ``source = bos + input`` and
    ``full = bos + input + output + eos``.  It masks source tokens when
    ``train_on_source`` is false.  This dataset keeps that loss contract while
    letting sparse-ft set the full sequence cap explicitly with max_seq_length.
    """

    def __init__(
        self,
        instances,
        tokenizer,
        full_max_len,
        source_max_len=768,
        train_on_source=False,
    ):
        self.data = []
        self.truncated_examples = 0
        self.first_1000_max_input_len = 0
        self.train_on_source = train_on_source
        self.full_max_len = full_max_len
        self.source_max_len = source_max_len

        bos = tokenizer.bos_token or ""
        eos = tokenizer.eos_token or ""
        for idx, item in enumerate(instances):
            source_text, output_text = _instruction_pair(item)
            source = f"{bos}{source_text}"
            full = f"{bos}{source_text}{output_text}{eos}"

            tokenized_source = tokenizer(
                source,
                max_length=source_max_len,
                truncation=True,
                add_special_tokens=False,
            )["input_ids"]
            untruncated_full = tokenizer(
                full,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]
            if len(untruncated_full) > full_max_len:
                self.truncated_examples += 1
            tokenized_full = tokenizer(
                full,
                max_length=full_max_len,
                truncation=True,
                add_special_tokens=False,
            )["input_ids"]

            if idx < 1000:
                self.first_1000_max_input_len = max(
                    self.first_1000_max_input_len,
                    len(tokenized_full),
                )

            labels = list(tokenized_full)
            if not train_on_source:
                source_len = min(len(tokenized_source), len(labels))
                labels = [IGNORE_INDEX] * source_len + labels[source_len:]

            self.data.append(
                {
                    "input_ids": tokenized_full,
                    "attention_mask": [1] * len(tokenized_full),
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class LoggingDataCollator:
    def __init__(self, base_collator, dataset, mode, max_seq_length, source_masking):
        self.base_collator = base_collator
        self.dataset = dataset
        self.mode = mode
        self.max_seq_length = max_seq_length
        self.source_masking = source_masking
        self._logged = False

    def __call__(self, features):
        batch = self.base_collator(features)
        if not self._logged:
            input_ids = batch.get("input_ids")
            labels = batch.get("labels")
            batch_max = int(input_ids.shape[-1]) if input_ids is not None else 0
            masked = int((labels == IGNORE_INDEX).sum().item()) if labels is not None else 0
            total_labels = int(labels.numel()) if labels is not None else 0
            logger.info(
                "[data] first_batch mode=%s max_seq_length=%s actual_batch_max_length=%s "
                "source_masking=%s masked_labels=%s/%s",
                self.mode,
                self.max_seq_length,
                batch_max,
                self.source_masking,
                masked,
                total_labels,
            )
            self._logged = True
        return batch


class RapaInstructionCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids = [
            torch.tensor(feature["input_ids"], dtype=torch.long)
            for feature in features
        ]
        attention_mask = [
            torch.tensor(feature["attention_mask"], dtype=torch.long)
            for feature in features
        ]
        labels = [
            torch.tensor(feature["labels"], dtype=torch.long)
            for feature in features
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _instruction_pair(item):
    if "input" in item or "output" in item:
        return item.get("input", ""), item.get("output", "")
    return "", item.get("text", "")


def build_lmflow_text_dataset(dataset_path, tokenizer, max_seq_length, dynamic_padding=None):
    """Return a dataset and collator for sparse-ft training."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if dynamic_padding is None:
        dynamic_padding = _env_bool("RAPA_USE_DYNAMIC_PADDING", True)

    dataset_type, instances = load_lmflow_instances(dataset_path)
    if dataset_type not in {None, "text_only"}:
        raise ValueError(
            f"Only LMFlow text_only datasets are supported here; got type={dataset_type!r}"
        )
    data_format = os.environ.get("RAPA_DATA_FORMAT", "text").lower()
    train_on_source = _env_bool("RAPA_TRAIN_ON_SOURCE", False)
    source_max_len = int(os.environ.get("RAPA_INSTRUCT_SOURCE_MAX_LEN", "768"))
    target_max_len = int(os.environ.get("RAPA_INSTRUCT_TARGET_MAX_LEN", "256"))

    if data_format in {"instruction", "rapa", "oasst1"}:
        dataset = RapaInstructionDataset(
            instances,
            tokenizer,
            full_max_len=max_seq_length,
            source_max_len=source_max_len,
            train_on_source=train_on_source,
        )
        collator = LoggingDataCollator(
            RapaInstructionCollator(tokenizer),
            dataset=dataset,
            mode="rapa_instruction",
            max_seq_length=max_seq_length,
            source_masking=not train_on_source,
        )
        logger.info(
            "[data] mode=rapa_instruction max_seq_length=%s source_max_len=%s "
            "target_max_len_reference=%s train_on_source=%s label_policy=%s samples=%s "
            "first_1000_max_input_len=%s truncated_examples=%s",
            max_seq_length,
            source_max_len,
            target_max_len,
            train_on_source,
            "response_only" if not train_on_source else "full_text",
            len(dataset),
            dataset.first_1000_max_input_len,
            dataset.truncated_examples,
        )
        return dataset, collator, dynamic_padding

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
    collator = LoggingDataCollator(
        collator,
        dataset=dataset,
        mode="lmflow_text",
        max_seq_length=max_seq_length,
        source_masking=False,
    )
    logger.info(
        "[data] mode=lmflow_text max_seq_length=%s dynamic_padding=%s "
        "label_policy=full_text samples=%s first_1000_max_input_len=%s truncated_examples=%s",
        max_seq_length,
        dynamic_padding,
        len(dataset),
        dataset.first_1000_max_input_len,
        dataset.truncated_examples,
    )
    return dataset, collator, dynamic_padding
