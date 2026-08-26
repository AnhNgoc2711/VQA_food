"""Vietnamese VQA dataset cho Hướng A (PhoBERT vocab + ViT input)."""
from __future__ import annotations
import json
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

import config

try:
    from underthesea import word_tokenize as _vn_tokenize
except ImportError:
    _vn_tokenize = None


def vn_segment(text: str) -> str:
    """Word-segment tiếng Việt cho PhoBERT (multi-word ghép bằng '_')."""
    if _vn_tokenize is None:
        return text
    return _vn_tokenize(text, format="text")


def detok_vn(tokenizer, ids) -> str:
    """Decode + bỏ dấu '_' của PhoBERT."""
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return text.replace("_", " ").strip()


def load_split(name: str) -> list[dict]:
    path = config.SPLITS_DIR / f"{name}_pairs.json"
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_image_path(image_path_in_json: str):
    """
    Xử lý các khác biệt giữa data path trong JSON và filesystem
    """
    rel = image_path_in_json.replace("\\", "/")
    candidates = [
        config.IMAGES_ROOT / rel,                       
    ]
    if rel.startswith("data/"):
        candidates.append(config.IMAGES_ROOT / rel[5:])
    elif not rel.startswith("data/"):
        candidates.append(config.IMAGES_ROOT / "data" / rel)
    # Fallback: relative tới project ROOT
    candidates.append(config.ROOT / rel)
    if rel.startswith("data/"):
        candidates.append(config.ROOT / rel[5:])

    for p in candidates:
        if p.exists():
            return p
    # Trả về candidate đầu tiên (sẽ fail rõ ràng) để debug dễ hơn
    return candidates[0]


def make_image_transform(train: bool = False):
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE + 16, config.IMG_SIZE + 16)),
            transforms.RandomCrop(config.IMG_SIZE),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class VQADataset(Dataset):
    """Trả về dict cho Hướng A (đã tokenize bằng PhoBERT)."""

    def __init__(self, split: str, tokenizer,
                 max_q_len: int = config.MAX_Q_LEN,
                 max_a_len: int = config.MAX_A_LEN,
                 transform=None):
        self.rows = load_split(split)
        self.tok = tokenizer
        self.max_q = max_q_len
        self.max_a = max_a_len
        self.tf = transform or make_image_transform(train=False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = resolve_image_path(row["image_path"])
        image = Image.open(img_path).convert("RGB")
        pixel = self.tf(image)

        q = self.tok(vn_segment(row["question"]),
                     padding="max_length", truncation=True,
                     max_length=self.max_q, return_tensors="pt")
        a = self.tok(vn_segment(row["answer"]),
                     padding="max_length", truncation=True,
                     max_length=self.max_a, return_tensors="pt")

        return {
            "pixel_values": pixel,
            "q_input_ids": q["input_ids"][0],
            "q_attn_mask": q["attention_mask"][0],
            "a_input_ids": a["input_ids"][0],
            "a_attn_mask": a["attention_mask"][0],
            "image_path": row["image_path"],
            "question": row["question"],
            "answer": row["answer"],
        }


def collate_fn(batch):
    out = {}
    for k in ("pixel_values", "q_input_ids", "q_attn_mask",
              "a_input_ids", "a_attn_mask"):
        out[k] = torch.stack([b[k] for b in batch])
    for k in ("image_path", "question", "answer"):
        out[k] = [b[k] for b in batch]
    return out


class VQARawDataset(Dataset):
    """Chỉ trả image PIL + question + answer — dùng cho Qwen-VL (B1/B2/DPO)."""

    def __init__(self, split: str):
        self.rows = load_split(split)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = resolve_image_path(row["image_path"])
        image = Image.open(img_path).convert("RGB")
        return {
            "image": image,
            "image_path": row["image_path"],
            "question": row["question"],
            "answer": row["answer"],
        }
