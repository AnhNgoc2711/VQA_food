"""Generate predictions cho 4 cấu hình. Output JSON list[dict]."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import config
from dataset import VQADataset, VQARawDataset, collate_fn, detok_vn


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def trim_to_10_words(text: str) -> str:
    words = text.split()
    return " ".join(words[:10])


# ------------------------------ A1 / A2 ------------------------------

def infer_a(ckpt_path: str, split: str, out_path: str):
    dev = device()
    tok = AutoTokenizer.from_pretrained(config.PHOBERT_NAME)
    pad_id = tok.pad_token_id
    bos_id = tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id

    state = torch.load(ckpt_path, map_location=dev)
    cfg = state["cfg"]
    decoder_type = state["decoder_type"]

    from models import VQAModelA
    model = VQAModelA(decoder_type, vocab_size=tok.vocab_size, cfg=cfg,
                      pad_id=pad_id, bos_id=bos_id, eos_id=eos_id).to(dev)
    model.load_state_dict(state["model"])
    model.eval()

    ds = VQADataset(split, tok)
    loader = DataLoader(ds, batch_size=cfg["bs"], shuffle=False,
                        num_workers=cfg["num_workers"], collate_fn=collate_fn)

    rows = []
    pbar = tqdm(loader, desc=f"infer {decoder_type}", ncols=110)
    with torch.no_grad():
        for batch in pbar:
            batch_dev = {k: (v.to(dev) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
            ids = model.generate(batch_dev, max_len=config.MAX_A_LEN)
            for i in range(ids.size(0)):
                pred = trim_to_10_words(detok_vn(tok, ids[i].tolist()))
                rows.append({
                    "image_path": batch["image_path"][i],
                    "question": batch["question"][i],
                    "gold": batch["answer"][i],
                    "pred": pred,
                })
    Path(out_path).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} predictions → {out_path}")


# ------------------------------ B1 / B2 / B2-DPO ------------------------------

def infer_b(adapter_path: str | None, split: str, out_path: str):
    """B1: adapter_path=None. B2 / B2-DPO: truyền thư mục adapter."""
    import os as _os
    _os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import gc

    dev = device()
    from models import load_qwenvl, generate_answer
    model, processor = load_qwenvl(adapter_path=adapter_path)

    # Resize ảnh trước inference để tránh OOM trên T4 (giới hạn 336px cạnh dài nhất)
    IMG_MAX_SIDE = 336

    def _resize(img):
        w, h = img.size
        m = max(w, h)
        if m > IMG_MAX_SIDE:
            s = IMG_MAX_SIDE / m
            img = img.resize((int(w * s), int(h * s)))
        return img

    ds = VQARawDataset(split)
    rows = []
    pbar = tqdm(range(len(ds)), desc="infer Qwen-VL", ncols=110)
    n_oom = 0
    for i in pbar:
        row = ds[i]
        try:
            img = _resize(row["image"])
            ans = generate_answer(model, processor, img, row["question"])
        except torch.cuda.OutOfMemoryError:
            n_oom += 1
            torch.cuda.empty_cache()
            gc.collect()
            ans = ""        # bỏ trống — sẽ EM=0 cho sample này
        rows.append({
            "image_path": row["image_path"],
            "question": row["question"],
            "gold": row["answer"],
            "pred": trim_to_10_words(ans),
        })
        # Giải phóng VRAM mỗi sample
        if i % 50 == 0:
            torch.cuda.empty_cache()
    if n_oom:
        print(f"⚠ Bỏ qua {n_oom} sample vì OOM (pred = empty)")
    Path(out_path).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} predictions → {out_path}")


# ------------------------------ Dispatcher ------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["a1", "a2", "b1", "b2", "b2_dpo"])
    p.add_argument("--ckpt", default=None,
                   help="Path .pt cho A1/A2 (mặc định: checkpoints/{A1|A2}/best.pt)")
    p.add_argument("--adapter", default=None,
                   help="Path adapter LoRA cho B2/B2-DPO")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.mode in ("a1", "a2"):
        cfg_name = args.mode.upper()
        ckpt = args.ckpt or str(config.CKPT_PATHS[cfg_name] / f"best_{cfg_name}.pt")
        out = args.out or str(config.PRED_PATHS[cfg_name])
        infer_a(ckpt, args.split, out)
    elif args.mode == "b1":
        out = args.out or str(config.PRED_PATHS["B1"])
        infer_b(adapter_path=None, split=args.split, out_path=out)
    elif args.mode == "b2":
        adapter = args.adapter or str(config.CKPT_PATHS["B2"])
        out = args.out or str(config.PRED_PATHS["B2"])
        infer_b(adapter_path=adapter, split=args.split, out_path=out)
    elif args.mode == "b2_dpo":
        adapter = args.adapter or str(config.CKPT_PATHS["B2_DPO"])
        out = args.out or str(config.PRED_PATHS["B2_DPO"])
        infer_b(adapter_path=adapter, split=args.split, out_path=out)


if __name__ == "__main__":
    main()
