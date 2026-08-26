"""Gộp toàn bộ model code: Hướng A (ViT+PhoBERT+Cross-Attn + LSTM/TF) và Hướng B (Qwen-VL)."""
from __future__ import annotations
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from transformers import AutoModel, AutoProcessor

# transformers ≥ 5.x đổi tên Vision2Seq → ImageTextToText. Hỗ trợ cả 2.
try:
    from transformers import AutoModelForImageTextToText as _VLMAuto
except ImportError:
    from transformers import AutoModelForVision2Seq as _VLMAuto

from peft import LoraConfig, get_peft_model, PeftModel

import config


# ============================== HƯỚNG A ==============================

class ViTEncoder(nn.Module):
    """ViT-base/16. Output (B, 197, 768)."""

    def __init__(self, name: str = config.VIT_NAME, freeze: bool = True):
        super().__init__()
        self.vit = timm.create_model(name, pretrained=True, num_classes=0)
        self.freeze = freeze
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, pixel_values):
        if self.freeze:
            with torch.no_grad():
                return self.vit.forward_features(pixel_values)
        return self.vit.forward_features(pixel_values)


class PhoBertEncoder(nn.Module):
    """PhoBERT-base. Output (B, T, 768)."""

    def __init__(self, name: str = config.PHOBERT_NAME, freeze: bool = False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(name)
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class CrossAttentionFusion(nn.Module):
    """Text query × Image key/value."""

    def __init__(self, d=768, heads=8, layers=2, dropout=0.1):
        super().__init__()
        layer = nn.TransformerDecoderLayer(d, heads, d * 4, dropout,
                                           batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=layers)

    def forward(self, text_feats, img_feats, text_mask):
        return self.dec(tgt=text_feats, memory=img_feats,
                        tgt_key_padding_mask=~text_mask.bool())


class LSTMDecoder(nn.Module):
    """LSTM với Bahdanau attention."""

    def __init__(self, vocab_size, d=768, hidden=512, n_layers=2, dropout=0.3,
                 pad_id=1, bos_id=0, eos_id=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d, padding_idx=pad_id)
        self.lstm = nn.LSTM(d + d, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.attn_q = nn.Linear(hidden, d)
        self.attn_k = nn.Linear(d, d)
        self.proj = nn.Linear(hidden + d, hidden)
        self.out = nn.Linear(hidden, vocab_size)
        self.h_init = nn.Linear(d, hidden)
        self.c_init = nn.Linear(d, hidden)
        self.n_layers, self.hidden, self.d = n_layers, hidden, d
        self.pad_id, self.bos_id, self.eos_id = pad_id, bos_id, eos_id

    def _init_state(self, memory, mask):
        m = mask.unsqueeze(-1).float()
        pooled = (memory * m).sum(1) / m.sum(1).clamp_min(1)
        h = torch.tanh(self.h_init(pooled)).unsqueeze(0).repeat(self.n_layers, 1, 1)
        c = torch.tanh(self.c_init(pooled)).unsqueeze(0).repeat(self.n_layers, 1, 1)
        return h, c, pooled

    def _attend(self, h_top, memory, mask):
        q = self.attn_q(h_top).unsqueeze(1)
        k = self.attn_k(memory)
        scores = (q * k).sum(-1) / math.sqrt(self.d)
        scores = scores.masked_fill(~mask.bool(), -1e4)
        w = F.softmax(scores, -1).unsqueeze(-1)
        return (w * memory).sum(1)

    def forward(self, memory, mem_mask, target_ids):
        T = target_ids.size(1)
        emb = self.embed(target_ids)
        h, c, ctx = self._init_state(memory, mem_mask)
        outs = []
        for t in range(T):
            inp = torch.cat([emb[:, t], ctx], dim=-1).unsqueeze(1)
            o, (h, c) = self.lstm(inp, (h, c))
            o = o.squeeze(1)
            ctx = self._attend(o, memory, mem_mask)
            outs.append(self.proj(torch.cat([o, ctx], dim=-1)))
        return self.out(torch.stack(outs, dim=1))

    @torch.no_grad()
    def generate(self, memory, mem_mask, max_len=config.MAX_A_LEN):
        B, dev = memory.size(0), memory.device
        h, c, ctx = self._init_state(memory, mem_mask)
        ids = torch.full((B, 1), self.bos_id, dtype=torch.long, device=dev)
        finished = torch.zeros(B, dtype=torch.bool, device=dev)
        for _ in range(max_len - 1):
            emb = self.embed(ids[:, -1])
            inp = torch.cat([emb, ctx], dim=-1).unsqueeze(1)
            o, (h, c) = self.lstm(inp, (h, c))
            o = o.squeeze(1)
            ctx = self._attend(o, memory, mem_mask)
            logits = self.out(self.proj(torch.cat([o, ctx], dim=-1)))
            nxt = logits.argmax(-1, keepdim=True)
            nxt = torch.where(finished.unsqueeze(-1),
                              torch.full_like(nxt, self.pad_id), nxt)
            ids = torch.cat([ids, nxt], dim=1)
            finished = finished | (nxt.squeeze(-1) == self.eos_id)
            if finished.all():
                break
        return ids


class _PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TFDecoder(nn.Module):
    """Transformer decoder cross-attn sang fused features."""

    def __init__(self, vocab_size, d=768, n_heads=8, n_layers=4, ff=2048,
                 dropout=0.1, pad_id=1, bos_id=0, eos_id=2, max_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d, padding_idx=pad_id)
        self.pos = _PositionalEncoding(d, max_len)
        layer = nn.TransformerDecoderLayer(d, n_heads, ff, dropout,
                                           batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d, vocab_size)
        self.d, self.pad_id, self.bos_id, self.eos_id = d, pad_id, bos_id, eos_id

    @staticmethod
    def _causal_mask(T, dev):
        return torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), diagonal=1)

    def forward(self, memory, mem_mask, target_ids):
        x = self.pos(self.embed(target_ids) * math.sqrt(self.d))
        h = self.dec(
            tgt=x, memory=memory,
            tgt_mask=self._causal_mask(x.size(1), x.device),
            tgt_key_padding_mask=(target_ids == self.pad_id),
            memory_key_padding_mask=~mem_mask.bool(),
        )
        return self.out(h)

    @torch.no_grad()
    def generate(self, memory, mem_mask, max_len=config.MAX_A_LEN):
        B, dev = memory.size(0), memory.device
        ids = torch.full((B, 1), self.bos_id, dtype=torch.long, device=dev)
        finished = torch.zeros(B, dtype=torch.bool, device=dev)
        for _ in range(max_len - 1):
            logits = self.forward(memory, mem_mask, ids)[:, -1]
            nxt = logits.argmax(-1, keepdim=True)
            nxt = torch.where(finished.unsqueeze(-1),
                              torch.full_like(nxt, self.pad_id), nxt)
            ids = torch.cat([ids, nxt], dim=1)
            finished = finished | (nxt.squeeze(-1) == self.eos_id)
            if finished.all():
                break
        return ids


class VQAModelA(nn.Module):
    """ViT + PhoBERT + Cross-Attn + decoder (lstm | tf)."""

    def __init__(self, decoder_type: str, vocab_size: int, cfg: dict,
                 pad_id=1, bos_id=0, eos_id=2):
        super().__init__()
        assert decoder_type in {"lstm", "tf"}
        self.decoder_type = decoder_type
        self.vit = ViTEncoder(freeze=True)
        self.txt = PhoBertEncoder(freeze=False)
        self.fuse = CrossAttentionFusion(cfg["d_model"], cfg["fusion_heads"],
                                         cfg["fusion_layers"], 0.1)
        if decoder_type == "lstm":
            self.dec = LSTMDecoder(vocab_size, cfg["d_model"], cfg["hidden"],
                                   cfg["n_layers"], cfg["dropout"],
                                   pad_id, bos_id, eos_id)
        else:
            self.dec = TFDecoder(vocab_size, cfg["d_model"], cfg["n_heads"],
                                 cfg["n_layers"], cfg["ff"], cfg["dropout"],
                                 pad_id, bos_id, eos_id)

    def encode(self, batch):
        img = self.vit(batch["pixel_values"])
        txt = self.txt(batch["q_input_ids"], batch["q_attn_mask"])
        fused = self.fuse(txt, img, batch["q_attn_mask"])
        return fused, batch["q_attn_mask"]

    def forward(self, batch):
        memory, mask = self.encode(batch)
        return self.dec(memory, mask, batch["a_input_ids"][:, :-1])

    @torch.no_grad()
    def generate(self, batch, max_len=config.MAX_A_LEN):
        memory, mask = self.encode(batch)
        return self.dec.generate(memory, mask, max_len=max_len)

    def trainable_param_groups(self, lr_enc, lr_dec):
        enc = list(self.txt.parameters())
        rest = list(self.fuse.parameters()) + list(self.dec.parameters())
        return [
            {"params": [p for p in enc if p.requires_grad], "lr": lr_enc},
            {"params": [p for p in rest if p.requires_grad], "lr": lr_dec},
        ]


# ============================== HƯỚNG B (Qwen-VL) ==============================

def load_qwenvl(adapter_path: str | Path | None = None,
                dtype=torch.bfloat16, device_map: str = "auto",
                for_training: bool = False):
    """Load Qwen-VL base; optionally merge LoRA adapter."""
    processor = AutoProcessor.from_pretrained(config.B["model_name"])
    model = _VLMAuto.from_pretrained(
        config.B["model_name"], torch_dtype=dtype, device_map=device_map,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    if not for_training:
        model.eval()
    return model, processor


def attach_lora(model, cfg: dict):
    lcfg = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lcfg)


def build_user_messages(question: str):
    return [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": config.PROMPT_VI.format(question=question)},
    ]}]


def build_full_messages(question: str, answer: str):
    return [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": config.PROMPT_VI.format(question=question)},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]


@torch.inference_mode()
def generate_answer(model, processor, image: Image.Image, question: str,
                    max_new: int = None) -> str:
    if max_new is None:
        max_new = config.B["max_new_tokens"]
    text = processor.apply_chat_template(
        build_user_messages(question), tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image],
                       padding=True, return_tensors="pt").to(model.device)
    in_len = inputs["input_ids"].size(1)
    gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return processor.batch_decode(gen[:, in_len:], skip_special_tokens=True)[0].strip()


@torch.inference_mode()
def generate_answer_sampled(model, processor, image: Image.Image, question: str,
                            temperature: float = 0.7, top_p: float = 0.9,
                            max_new: int = None) -> str:
    """Sampling — dùng khi build preference data cho DPO."""
    if max_new is None:
        max_new = config.B["max_new_tokens"]
    text = processor.apply_chat_template(
        build_user_messages(question), tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image],
                       padding=True, return_tensors="pt").to(model.device)
    in_len = inputs["input_ids"].size(1)
    gen = model.generate(**inputs, max_new_tokens=max_new,
                         do_sample=True, temperature=temperature, top_p=top_p)
    return processor.batch_decode(gen[:, in_len:], skip_special_tokens=True)[0].strip()
