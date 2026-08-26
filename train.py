"""Training cho A1, A2, B2-SFT, B2-DPO. Chọn qua --mode."""
from __future__ import annotations
import os as _os

_os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import config
from dataset import VQADataset, VQARawDataset, collate_fn, detok_vn, vn_segment, resolve_image_path


# ------------------------------ Common utils ------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%H:%M:%S")
    fh = logging.FileHandler(config.LOG_DIR / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    return f"{m:d}m{s:02d}s"


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ============================== A1 / A2 ==============================

def _quick_bleu(preds: list[str], golds: list[str]) -> float:
    import sacrebleu
    return sacrebleu.corpus_bleu(preds, [golds]).score / 100.0


def _evaluate_a(model, loader, tokenizer, device_, max_len: int,
                loss_fn=None) -> dict:
    """Eval model A trên val: trả về val_loss + BLEU + EM + token_acc."""
    model.eval()
    preds, golds = [], []
    total_loss, total_tokens = 0.0, 0
    correct_tokens = 0
    pad_id = tokenizer.pad_token_id

    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: (v.to(device_) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}

            # 1. Tính val_loss + token-level accuracy (teacher forcing)
            if loss_fn is not None:
                logits = model(batch_dev)                        # (B, T-1, V)
                target = batch_dev["a_input_ids"][:, 1:]
                loss = loss_fn(logits.reshape(-1, logits.size(-1)),
                               target.reshape(-1))
                mask = (target != pad_id)
                n_tok = mask.sum().item()
                total_loss += loss.item() * n_tok
                total_tokens += n_tok
                pred_tok = logits.argmax(-1)
                correct_tokens += ((pred_tok == target) & mask).sum().item()

            # 2. Generate để tính BLEU + EM
            ids = model.generate(batch_dev, max_len=max_len)
            for i in range(ids.size(0)):
                preds.append(detok_vn(tokenizer, ids[i].tolist()))
                golds.append(batch["answer"][i])

    bleu = _quick_bleu(preds, golds)
    em = sum(p.lower().strip() == g.lower().strip() for p, g in zip(preds, golds)) / len(preds)
    out = {"BLEU": bleu, "EM": em, "n": len(preds)}
    if loss_fn is not None and total_tokens > 0:
        out["loss"] = total_loss / total_tokens
        out["token_acc"] = correct_tokens / total_tokens
    return out


def train_a(decoder_type: str, cfg: dict):
    name = "A1" if decoder_type == "lstm" else "A2"
    logger = setup_logger(name)
    set_seed(config.SEED)
    dev = device()

    logger.info(f"=== Training {name} ({decoder_type}) on {dev} ===")
    logger.info(f"Config: {cfg}")

    # tokenizer
    tok = AutoTokenizer.from_pretrained(config.PHOBERT_NAME)
    pad_id = tok.pad_token_id
    bos_id = tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id

    # data
    from dataset import make_image_transform
    train_ds = VQADataset("train", tok, transform=make_image_transform(train=True))
    val_ds = VQADataset("val", tok, transform=make_image_transform(train=False))
    train_loader = DataLoader(train_ds, batch_size=cfg["bs"], shuffle=True,
                              num_workers=cfg["num_workers"], collate_fn=collate_fn,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["bs"], shuffle=False,
                            num_workers=cfg["num_workers"], collate_fn=collate_fn,
                            pin_memory=True)
    logger.info(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")

    # model
    from models import VQAModelA
    model = VQAModelA(decoder_type, vocab_size=tok.vocab_size, cfg=cfg,
                      pad_id=pad_id, bos_id=bos_id, eos_id=eos_id).to(dev)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {n_train/1e6:.2f}M")

    # optimizer (2 lr groups)
    groups = model.trainable_param_groups(cfg["lr_enc"], cfg["lr_dec"])
    optim = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])

    total_steps = len(train_loader) * cfg["epochs"]
    warmup = int(total_steps * cfg["warmup_ratio"])
    sched = get_linear_schedule_with_warmup(optim, warmup, total_steps)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id,
                                  label_smoothing=cfg["label_smoothing"])

    # logging
    writer = SummaryWriter(config.RUNS_DIR / name)
    ckpt_dir = config.CKPT_PATHS[name]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_amp = (dev == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_bleu = -1.0
    best_path = ckpt_dir / f"best_{name}.pt"
    global_step = 0
    t_start = time.time()

    # Early stopping + LR plateau state
    epochs_no_improve = 0
    plateau_counter = 0
    cur_lr_scale = 1.0

    with logging_redirect_tqdm():
        outer = tqdm(range(cfg["epochs"]), desc=f"{name}", position=0,
                     ncols=110, dynamic_ncols=True)
        for epoch in outer:
            model.train()
            inner = tqdm(train_loader, desc=f"  ep{epoch+1:02d}",
                         position=1, leave=False, ncols=110, dynamic_ncols=True)
            running = 0.0
            for step, batch in enumerate(inner):
                batch_dev = {k: (v.to(dev) if torch.is_tensor(v) else v)
                             for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                    logits = model(batch_dev)              # (B, T-1, V)
                    target = batch_dev["a_input_ids"][:, 1:]
                    loss = loss_fn(logits.reshape(-1, logits.size(-1)),
                                   target.reshape(-1))

                optim.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg["grad_clip"],
                )
                scaler.step(optim)
                scaler.update()
                sched.step()

                running = 0.9 * running + 0.1 * loss.item() if running > 0 else loss.item()
                inner.set_postfix(loss=f"{running:.3f}",
                                  lr=f"{sched.get_last_lr()[1]:.1e}")
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr_dec", sched.get_last_lr()[1], global_step)
                writer.add_scalar("train/lr_enc", sched.get_last_lr()[0], global_step)
                global_step += 1

                if (step + 1) % cfg["log_every"] == 0:
                    logger.info(f"ep{epoch+1} step{step+1}/{len(train_loader)} loss={running:.3f}")

            # validation
            metrics = _evaluate_a(model, val_loader, tok, dev,
                                  max_len=config.MAX_A_LEN, loss_fn=loss_fn)
            writer.add_scalar("val/BLEU", metrics["BLEU"], epoch)
            writer.add_scalar("val/EM", metrics["EM"], epoch)
            if "loss" in metrics:
                writer.add_scalar("val/loss", metrics["loss"], epoch)
                writer.add_scalar("val/token_acc", metrics["token_acc"], epoch)
            elapsed = time.time() - t_start
            extras = (f" loss={metrics['loss']:.4f} tok_acc={metrics['token_acc']:.4f}"
                      if "loss" in metrics else "")
            logger.info(f"[VAL] ep{epoch+1} BLEU={metrics['BLEU']:.4f} "
                        f"EM={metrics['EM']:.4f}{extras} "
                        f"elapsed={fmt_time(elapsed)}")

            improved = metrics["BLEU"] > best_bleu + cfg.get("early_stop_min_delta", 0.0)
            if improved:
                best_bleu = metrics["BLEU"]
                torch.save({
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "decoder_type": decoder_type,
                    "vocab_size": tok.vocab_size,
                    "epoch": epoch + 1,
                    "BLEU": best_bleu,
                }, best_path)
                epochs_no_improve = 0
                plateau_counter = 0
                logger.info(f"[BEST] saved {best_path.name} at epoch {epoch+1} BLEU={best_bleu:.4f}")
            else:
                epochs_no_improve += 1
                plateau_counter += 1
                logger.info(f"[NO IMPROVE] {epochs_no_improve}/{cfg['early_stop_patience']} "
                            f"epochs since best BLEU={best_bleu:.4f}")

            # LR plateau: giảm LR khi val đứng vài epoch
            if plateau_counter >= cfg.get("lr_plateau_patience", 999):
                cur_lr_scale *= cfg.get("lr_plateau_factor", 0.5)
                for g in optim.param_groups:
                    g["lr"] *= cfg.get("lr_plateau_factor", 0.5)
                logger.info(f"[LR PLATEAU] giảm LR x{cfg['lr_plateau_factor']} "
                            f"-> scale tổng = {cur_lr_scale:.3f}")
                plateau_counter = 0

            postfix = {
                "BLEU": f"{metrics['BLEU']:.3f}",
                "best": f"{best_bleu:.3f}",
                "EM": f"{metrics['EM']:.3f}",
                "patience": f"{epochs_no_improve}/{cfg['early_stop_patience']}",
            }
            if "loss" in metrics:
                postfix["val_loss"] = f"{metrics['loss']:.3f}"
            outer.set_postfix(**postfix)

            # Early stopping
            if epochs_no_improve >= cfg.get("early_stop_patience", 999):
                logger.info(f"[EARLY STOP] {epochs_no_improve} epoch không cải thiện, "
                            f"dừng ở epoch {epoch+1}/{cfg['epochs']}")
                break

    writer.close()
    total = time.time() - t_start
    logger.info(f"=== Done {name}. Best val BLEU={best_bleu:.4f}. Total time={fmt_time(total)} ===")
    logger.info(f"Best ckpt: {best_path}")


# ============================== B2 SFT ==============================

class _QwenSFTDataset(torch.utils.data.Dataset):
    """Dataset cho Qwen-VL SFT — trả raw image + chuỗi prompt + chuỗi đầy đủ.
    """

    def __init__(self, split: str, processor, image_max_size: int = 448,
                 subset_size: int | None = None):
        self.raw = VQARawDataset(split)
        self.proc = processor
        self.image_max_size = image_max_size

        # Subsample
        n_full = len(self.raw)
        if subset_size and subset_size < n_full:
            rng = random.Random(42)
            self.indices = rng.sample(range(n_full), subset_size)
        else:
            self.indices = list(range(n_full))

    def __len__(self):
        return len(self.indices)

    def _resize_image(self, img):
        """Resize ảnh để cạnh dài nhất ≤ image_max_size — kiểm soát num_patches."""
        w, h = img.size
        max_side = max(w, h)
        if max_side > self.image_max_size:
            scale = self.image_max_size / max_side
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size)
        return img

    def __getitem__(self, idx):
        from models import build_user_messages, build_full_messages
        real_idx = self.indices[idx]
        row = self.raw[real_idx]
        img = self._resize_image(row["image"])      # FIX OOM: resize trước
        q, a = row["question"], row["answer"]

        prompt_msgs = build_user_messages(q)
        full_msgs = build_full_messages(q, a)
        prompt_text = self.proc.apply_chat_template(prompt_msgs, tokenize=False,
                                                    add_generation_prompt=True)
        full_text = self.proc.apply_chat_template(full_msgs, tokenize=False,
                                                  add_generation_prompt=False)

        full = self.proc(text=[full_text], images=[img],
                         padding=True, return_tensors="pt")
        prompt = self.proc(text=[prompt_text], images=[img],
                           padding=True, return_tensors="pt")

        input_ids = full["input_ids"][0]
        attn = full["attention_mask"][0]
        prompt_len = prompt["input_ids"].size(1)
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        item = {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
        }
        # Qwen2-VL: vision tensors có shape variable theo từng ảnh.
        if "pixel_values" in full:
            item["pixel_values"] = full["pixel_values"]
        if "image_grid_thw" in full:
            item["image_grid_thw"] = full["image_grid_thw"]
        if "mm_token_type_ids" in full:
            item["mm_token_type_ids"] = full["mm_token_type_ids"][0]   # (T,)
        return item


def _qwen_sft_collator(batch):
    """Pad input_ids/labels/attention_mask, stack vision tensors (đã giống nhau về shape)."""
    keys = batch[0].keys()
    out = {}
    pad_id = 0 
    max_len = max(b["input_ids"].size(0) for b in batch)
    for b in batch:
        n = max_len - b["input_ids"].size(0)
        if n > 0:
            b["input_ids"] = torch.cat([b["input_ids"], torch.full((n,), pad_id)])
            b["attention_mask"] = torch.cat([b["attention_mask"], torch.zeros(n, dtype=torch.long)])
            b["labels"] = torch.cat([b["labels"], torch.full((n,), -100)])
    out["input_ids"] = torch.stack([b["input_ids"] for b in batch])
    out["attention_mask"] = torch.stack([b["attention_mask"] for b in batch])
    out["labels"] = torch.stack([b["labels"] for b in batch])
    if "pixel_values" in keys:
        out["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])
    if "image_grid_thw" in keys:
        out["image_grid_thw"] = torch.stack([b["image_grid_thw"] for b in batch])
    return out


def train_b2_sft(cfg: dict):
    """Fine-tune Qwen-VL với LoRA bằng custom loop (giữ tqdm + tensorboard)."""
    name = "B2"
    logger = setup_logger("B2_sft")
    set_seed(config.SEED)
    dev = device()
    logger.info(f"=== Training B2 (Qwen-VL LoRA SFT) on {dev} ===")
    logger.info(f"Config: {cfg}")

    # Giảm fragmentation VRAM
    import os as _os
    _os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    from models import load_qwenvl, attach_lora
    model, processor = load_qwenvl(adapter_path=None, for_training=True)

    # Giới hạn kích thước ảnh để giảm patch count → giảm VRAM activation
    if cfg.get("image_max_pixels"):
        try:
            processor.image_processor.max_pixels = cfg["image_max_pixels"]
            logger.info(f"image_processor.max_pixels = {cfg['image_max_pixels']}")
        except Exception as e:
            logger.info(f"Không set được max_pixels: {e}")

    model = attach_lora(model, cfg)

    # Gradient checkpointing — đổi thời gian lấy VRAM
    if cfg.get("grad_checkpointing", False):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
            logger.info("Gradient checkpointing ENABLED")
        except Exception as e:
            logger.info(f"Không bật được gradient checkpointing: {e}")

    model.print_trainable_parameters()
    model.train()

    # Tính image_max_size từ image_max_pixels
    img_max_pixels = cfg.get("image_max_pixels", 448 * 448)
    img_max_side = int(img_max_pixels ** 0.5)
    subset_size = cfg.get("train_subset", None)
    logger.info(f"image_max_side = {img_max_side} | train_subset = {subset_size}")
    ds = _QwenSFTDataset("train", processor,
                         image_max_size=img_max_side,
                         subset_size=subset_size)
    logger.info(f"Dataset size (after subset): {len(ds)}")

    pad_id = processor.tokenizer.pad_token_id or 0

    def collate(batch):
        # 1. Pad input_ids / attention_mask / labels / mm_token_type_ids về cùng max_len
        max_len = max(b["input_ids"].size(0) for b in batch)
        for b in batch:
            n = max_len - b["input_ids"].size(0)
            if n > 0:
                b["input_ids"] = torch.cat(
                    [b["input_ids"], torch.full((n,), pad_id, dtype=torch.long)])
                b["attention_mask"] = torch.cat(
                    [b["attention_mask"], torch.zeros(n, dtype=torch.long)])
                b["labels"] = torch.cat(
                    [b["labels"], torch.full((n,), -100, dtype=torch.long)])
                if "mm_token_type_ids" in b:
                    b["mm_token_type_ids"] = torch.cat(
                        [b["mm_token_type_ids"],
                         torch.zeros(n, dtype=b["mm_token_type_ids"].dtype)])

        out = {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }
        # 2. mm_token_type_ids: (B, T) — stack giống input_ids
        if "mm_token_type_ids" in batch[0]:
            out["mm_token_type_ids"] = torch.stack(
                [b["mm_token_type_ids"] for b in batch])

        # 3. Qwen2-VL vision tensors — shape variable theo từng ảnh.
        if "pixel_values" in batch[0]:
            out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
        if "image_grid_thw" in batch[0]:
            out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
        return out

    loader = DataLoader(ds, batch_size=cfg["bs"], shuffle=True,
                        num_workers=0, collate_fn=collate)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    total_steps = (len(loader) // cfg["grad_accum"]) * cfg["epochs"]
    warmup = int(total_steps * cfg["warmup_ratio"])
    sched = get_linear_schedule_with_warmup(optim, warmup, total_steps)

    writer = SummaryWriter(config.RUNS_DIR / "B2_sft")
    ckpt_dir = config.CKPT_PATHS["B2"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    global_step = 0

    with logging_redirect_tqdm():
        for epoch in range(cfg["epochs"]):
            outer = tqdm(loader, desc=f"B2 ep{epoch+1}/{cfg['epochs']}",
                         ncols=110, dynamic_ncols=True)
            running = 0.0
            optim.zero_grad(set_to_none=True)
            for step, batch in enumerate(outer):
                batch = {k: v.to(dev) for k, v in batch.items()}
                out = model(**batch)
                loss = out.loss / cfg["grad_accum"]
                loss.backward()

                if (step + 1) % cfg["grad_accum"] == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    optim.step()
                    sched.step()
                    optim.zero_grad(set_to_none=True)
                    global_step += 1

                running = 0.9 * running + 0.1 * out.loss.item() if running > 0 else out.loss.item()
                outer.set_postfix(loss=f"{running:.3f}",
                                  lr=f"{sched.get_last_lr()[0]:.1e}")
                writer.add_scalar("train/loss", out.loss.item(), global_step)

                if (step + 1) % cfg.get("log_every", 200) == 0:
                    logger.info(f"ep{epoch+1} step{step+1}/{len(loader)} loss={running:.3f}")

    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)
    writer.close()
    logger.info(f"=== Done B2 SFT. Adapter saved to {ckpt_dir}. "
                f"Total time={fmt_time(time.time()-t_start)} ===")


# ============================== B2 DPO ==============================

def _compute_logps(model, input_ids, attention_mask, labels, **vision):
    """Tính sum log-prob của các token có label != -100 (token assistant)."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, **vision)
    logits = out.logits[:, :-1, :]                     # (B, T-1, V)
    targets = labels[:, 1:]                             # (B, T-1)
    mask = (targets != -100)
    targets_safe = targets.clone()
    targets_safe[~mask] = 0                             # tránh index -100
    logp = torch.log_softmax(logits.float(), dim=-1)
    chosen_logp = logp.gather(-1, targets_safe.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    chosen_logp = chosen_logp * mask.float()
    return chosen_logp.sum(dim=-1)                      # (B,)


def train_b2_dpo(cfg: dict):
    """DPO trên adapter B2 với preference data — MANUAL LOOP (không dùng DPOTrainer).

    Mỗi step:
      1. Tính log_p_policy_chosen, log_p_policy_rejected (LoRA adapter active)
      2. Tính log_p_ref_chosen, log_p_ref_rejected (LoRA adapter disabled)
      3. DPO loss = -log σ(β × ((log_p_policy_chosen - log_p_ref_chosen)
                              - (log_p_policy_rejected - log_p_ref_rejected)))
    """
    name = "B2_DPO"
    logger = setup_logger("B2_dpo")
    set_seed(config.SEED)
    dev = device()
    logger.info(f"=== Training B2-DPO (MANUAL LOOP) on {dev} ===")
    logger.info(f"Config: {cfg}")

    from models import load_qwenvl, build_user_messages, build_full_messages
    from PIL import Image

    pref_path = config.OUT_DIR / "preferences.jsonl"
    if not pref_path.exists():
        raise FileNotFoundError(f"Chạy `python rl.py --task build` trước để có {pref_path}")

    rows = [json.loads(l) for l in pref_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    logger.info(f"Loaded {len(rows)} preference pairs")

    # Load policy (có adapter B2 SFT, sẽ train tiếp)
    policy, processor = load_qwenvl(adapter_path=config.CKPT_PATHS["B2"], for_training=True)
    policy.gradient_checkpointing_disable()
    if hasattr(policy, "base_model") and hasattr(policy.base_model, "gradient_checkpointing_disable"):
        policy.base_model.gradient_checkpointing_disable()
    policy.config.use_cache = False
    if hasattr(policy, "enable_input_require_grads"):
        policy.enable_input_require_grads()

    # FORCE UNFREEZE LoRA — PeftModel.from_pretrained() mặc định load ở mode eval
    # → tất cả params bị freeze. Phải unfreeze thủ công các param có tên chứa "lora".
    n_unfrozen = 0
    for n, p in policy.named_parameters():
        if "lora_" in n.lower():
            p.requires_grad = True
            n_unfrozen += 1
    logger.info(f"Force unfrozen {n_unfrozen} LoRA tensors")

    # Đếm trainable params sau khi unfreeze
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {n_train/1e6:.2f}M")
    if n_train == 0:
        raise RuntimeError(
            "Vẫn không có tham số nào trainable! Kiểm tra checkpoints/B2_lora/ có "
            "phải LoRA adapter không (phải có file adapter_model.safetensors)."
        )
    policy.train()

    # Resize image trước khi đưa vào processor (tiết kiệm VRAM)
    img_max_side = 336

    def _resize(img):
        w, h = img.size
        m = max(w, h)
        if m > img_max_side:
            s = img_max_side / m
            img = img.resize((int(w * s), int(h * s)))
        return img

    def _build_inputs(question: str, answer: str, image: Image.Image):
        """Build batch B=1 với labels mask cho phần answer."""
        prompt_text = processor.apply_chat_template(
            build_user_messages(question), tokenize=False, add_generation_prompt=True)
        full_text = processor.apply_chat_template(
            build_full_messages(question, answer), tokenize=False, add_generation_prompt=False)
        full = processor(text=[full_text], images=[image],
                         padding=True, return_tensors="pt")
        prompt = processor(text=[prompt_text], images=[image],
                           padding=True, return_tensors="pt")
        labels = full["input_ids"].clone()
        labels[:, : prompt["input_ids"].size(1)] = -100
        out = {
            "input_ids": full["input_ids"].to(dev),
            "attention_mask": full["attention_mask"].to(dev),
            "labels": labels.to(dev),
        }
        for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            if k in full:
                out[k] = full[k].to(dev)
        return out

    optim = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                              lr=cfg["lr"], weight_decay=0.01)
    total_steps = (len(rows) // cfg["grad_accum"]) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(optim, int(total_steps * 0.03), total_steps)

    writer = SummaryWriter(config.RUNS_DIR / "B2_dpo")
    ckpt_dir = config.CKPT_PATHS["B2_DPO"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    beta = cfg["beta"]
    t_start = time.time()
    global_step = 0

    with logging_redirect_tqdm():
        for epoch in range(cfg["epochs"]):
            random.Random(config.SEED + epoch).shuffle(rows)
            pbar = tqdm(rows, desc=f"DPO ep{epoch+1}/{cfg['epochs']}", ncols=110)
            running_loss, running_acc = 0.0, 0.0
            optim.zero_grad(set_to_none=True)
            for step, r in enumerate(pbar):
                try:
                    img_path = resolve_image_path(r["image_path"])
                    img = _resize(Image.open(img_path).convert("RGB"))
                except Exception as e:
                    logger.info(f"skip {r['image_path']}: {e}")
                    continue

                # Build batch cho chosen + rejected
                bc = _build_inputs(r["question"], r["chosen"], img)
                br = _build_inputs(r["question"], r["rejected"], img)

                # Tách vision kwargs
                vis_c = {k: v for k, v in bc.items()
                         if k in ("pixel_values", "image_grid_thw", "mm_token_type_ids")}
                vis_r = {k: v for k, v in br.items()
                         if k in ("pixel_values", "image_grid_thw", "mm_token_type_ids")}

                # Policy log-probs (adapter active)
                logp_pi_c = _compute_logps(policy, bc["input_ids"], bc["attention_mask"],
                                           bc["labels"], **vis_c)
                logp_pi_r = _compute_logps(policy, br["input_ids"], br["attention_mask"],
                                           br["labels"], **vis_r)

                # Reference log-probs (adapter disabled — base model gốc)
                with torch.no_grad():
                    with policy.disable_adapter():
                        logp_ref_c = _compute_logps(policy, bc["input_ids"],
                                                    bc["attention_mask"], bc["labels"], **vis_c)
                        logp_ref_r = _compute_logps(policy, br["input_ids"],
                                                    br["attention_mask"], br["labels"], **vis_r)

                # DPO loss
                pi_logratios = logp_pi_c - logp_pi_r
                ref_logratios = logp_ref_c - logp_ref_r
                logits = beta * (pi_logratios - ref_logratios)
                loss = -torch.nn.functional.logsigmoid(logits).mean()
                acc = (logits > 0).float().mean().item()

                (loss / cfg["grad_accum"]).backward()

                if (step + 1) % cfg["grad_accum"] == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in policy.parameters() if p.requires_grad], 1.0)
                    optim.step()
                    sched.step()
                    optim.zero_grad(set_to_none=True)
                    global_step += 1

                running_loss = 0.9 * running_loss + 0.1 * loss.item() if running_loss > 0 else loss.item()
                running_acc = 0.9 * running_acc + 0.1 * acc if running_acc > 0 else acc
                pbar.set_postfix(
                    loss=f"{running_loss:.4f}",
                    acc=f"{running_acc:.3f}",
                    margin=f"{(pi_logratios - ref_logratios).mean().item():+.3f}",
                )
                writer.add_scalar("dpo/loss", loss.item(), global_step)
                writer.add_scalar("dpo/accuracy", acc, global_step)
                writer.add_scalar("dpo/margin",
                                  (pi_logratios - ref_logratios).mean().item(), global_step)

                # Giải phóng VRAM
                del bc, br, vis_c, vis_r, logp_pi_c, logp_pi_r, logp_ref_c, logp_ref_r
                torch.cuda.empty_cache()

                if (step + 1) % 20 == 0:
                    logger.info(f"ep{epoch+1} step{step+1}/{len(rows)} "
                                f"loss={running_loss:.4f} acc={running_acc:.3f}")

    policy.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)
    writer.close()
    logger.info(f"=== Done DPO. Adapter saved to {ckpt_dir}. "
                f"Total time={fmt_time(time.time()-t_start)} ===")


# ============================== Dispatcher ==============================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["a1", "a2", "b2_sft", "b2_dpo"])
    args = p.parse_args()

    if args.mode == "a1":
        train_a("lstm", config.A1)
    elif args.mode == "a2":
        train_a("tf", config.A2)
    elif args.mode == "b2_sft":
        train_b2_sft(config.B2_SFT)
    elif args.mode == "b2_dpo":
        train_b2_dpo(config.DPO_CFG)


if __name__ == "__main__":
    main()
