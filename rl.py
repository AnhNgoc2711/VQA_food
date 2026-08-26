"""DPO support: build preference data từ B2 + human evaluation app."""
from __future__ import annotations
import argparse
import gc
import json
import os as _os
import random
import re
import string
import unicodedata
from pathlib import Path

# Phải set TRƯỚC import torch 
_os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from tqdm.auto import tqdm

import config
from dataset import load_split, resolve_image_path

# Giới hạn kích thước file ảnh (byte).→ OOM khi forward Qwen-VL.
MAX_IMAGE_BYTES = 300 * 1024     # 300 KB
IMAGE_RESIZE_MAX_SIDE = 336      # cạnh dài nhất sau resize


# ------------------------------ Inline metric helpers ------------------------------

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def _normalize_text(t: str) -> str:
    t = unicodedata.normalize("NFC", t).lower().strip()
    t = _PUNCT_RE.sub(" ", t)
    return " ".join(t.split())


def _bertscore_batch(preds: list[str], golds: list[str]) -> list[float]:
    """BERTScore (xlm-roberta-base) batch — load model 1 lần, score nhiều cặp."""
    from bert_score import score
    P, R, F = score(preds, golds, model_type="xlm-roberta-base",
                    lang="vi", verbose=False)
    return F.tolist()


# ------------------------------ Preference builder ------------------------------

def _score_batch(preds: list[str], golds: list[str]) -> list[float]:
    """Score = 0.5*EM + 0.5*BERTScore cho từng cặp pred/gold (batch)."""
    em = [1.0 if _normalize_text(p) == _normalize_text(g) else 0.0
          for p, g in zip(preds, golds)]
    bs = _bertscore_batch(preds, golds)
    return [0.5 * e + 0.5 * b for e, b in zip(em, bs)]


def build_preferences(adapter_path: str, n_samples: int = 150,
                      n_candidates: int = 2, min_delta: float = 0.1,
                      out_path: str = None):
    """Sinh cặp (chosen, rejected) từ B2 trên train set.

      A) GOLD vs SAMPLED: chosen=gold, rejected=B2 candidate khi B2 trả khác gold
      B) SELF DIVERSITY: nếu B2 sinh nhiều candidate khác nhau, chọn cặp (best, worst) khi Δ score đủ lớn.

    Pair sẽ chọn cách tốt nhất giữa A và B.
    """
    from models import load_qwenvl, generate_answer_sampled

    out_path = out_path or str(config.OUT_DIR / "preferences.jsonl")
    rows = load_split("train")
    random.Random(config.SEED).shuffle(rows)
    rows = rows[:n_samples]
    print(f"Building preferences from {len(rows)} train samples ...")

    model, processor = load_qwenvl(adapter_path=adapter_path)

    # Bước 1 — Sinh tất cả candidates trước (chỉ load Qwen-VL 1 lần)
    all_cands = []      # list[(row, list[str candidates])]
  
    temps = [0.7, 1.0, 1.2, 1.5][:n_candidates]   # Nhiệt độ rộng hơn để tăng diversity của candidates
    n_skip_size, n_skip_err = 0, 0

    def _resize_image(img):
        """Resize cạnh dài nhất xuống IMAGE_RESIZE_MAX_SIDE."""
        w, h = img.size
        m = max(w, h)
        if m > IMAGE_RESIZE_MAX_SIDE:
            scale = IMAGE_RESIZE_MAX_SIDE / m
            img = img.resize((int(w * scale), int(h * scale)))
        return img

    for r in tqdm(rows, desc="sampling B2", ncols=110):
        img_path = resolve_image_path(r["image_path"])

        # Lọc theo kích thước file để tránh OOM
        try:
            file_size = img_path.stat().st_size
        except Exception as e:
            print(f"skip (stat) {r['image_path']}: {e}")
            n_skip_err += 1
            continue
        if file_size > MAX_IMAGE_BYTES:
            n_skip_size += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img = _resize_image(img)
        except Exception as e:
            print(f"skip (load) {r['image_path']}: {e}")
            n_skip_err += 1
            continue

        try:
            cands = [generate_answer_sampled(model, processor, img,
                                              r["question"], temperature=t)
                     for t in temps]
        except torch.cuda.OutOfMemoryError as e:
            print(f"skip (OOM) {r['image_path']}: {e}")
            n_skip_err += 1
            torch.cuda.empty_cache()
            gc.collect()
            continue

        all_cands.append((r, cands))

        # Giải phóng bộ nhớ đệm sau mỗi sample để tránh fragmentation
        del img
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Sampled {len(all_cands)} | skipped size: {n_skip_size} | "
          f"skipped err/OOM: {n_skip_err}")

    # Bước 2 — Free Qwen-VL VRAM trước khi load BERTScore
    del model, processor
    torch.cuda.empty_cache()

    # Bước 3 — Batch BERTScore: chấm cả candidates VÀ gold (gold luôn = 1.0 với gold).
    #   Để dùng strategy A, ta chấm điểm từng candidate vs gold.
    print("Computing BERTScore for all candidates ...")
    flat_preds, flat_golds, owner_idx = [], [], []
    for i, (r, cands) in enumerate(all_cands):
        for c in cands:
            flat_preds.append(c)
            flat_golds.append(r["answer"])
            owner_idx.append(i)
    flat_scores = _score_batch(flat_preds, flat_golds)

    # Bước 4 — Gộp score về từng row, chọn chosen/rejected theo 2 chiến lược
    pairs = []
    n_strategy_a, n_strategy_b = 0, 0
    for i, (r, cands) in enumerate(all_cands):
        row_scores = [flat_scores[j] for j, oi in enumerate(owner_idx) if oi == i]
        scored = sorted(zip(cands, row_scores), key=lambda x: x[1], reverse=True)
        best_cand, sc_best = scored[0]
        worst_cand, sc_worst = scored[-1]

        # ============ STRATEGY B — diversity nội bộ ============
        # Nếu B2 sinh đa dạng đủ → dùng (best, worst) candidates
        if sc_best - sc_worst >= min_delta and best_cand.strip() != worst_cand.strip():
            pairs.append({
                "image_path": r["image_path"],
                "question": r["question"],
                "gold": r["answer"],
                "chosen": best_cand,
                "rejected": worst_cand,
                "score_chosen": sc_best,
                "score_rejected": sc_worst,
                "strategy": "self_diversity",
            })
            n_strategy_b += 1
            continue

        # ============ STRATEGY A — gold vs B2 sai ============
        # B2 hội tụ nhưng có thể CHƯA HOÀN TOÀN ĐÚNG. Lấy gold làm chosen,
        # lấy candidate KÉM NHẤT làm rejected — miễn là worst != gold.
        gold_norm = _normalize_text(r["answer"])
        worst_norm = _normalize_text(worst_cand)

        if worst_norm != gold_norm and sc_worst < 0.95:
            # gold = chosen (perfect, score gần 1.0)
            # rejected = worst candidate (B2 sai)
            pairs.append({
                "image_path": r["image_path"],
                "question": r["question"],
                "gold": r["answer"],
                "chosen": r["answer"],          # GOLD
                "rejected": worst_cand,
                "score_chosen": 1.0,
                "score_rejected": sc_worst,
                "strategy": "gold_vs_sampled",
            })
            n_strategy_a += 1
            continue

        # Cả 2 chiến lược đều không tạo được cặp → skip
        continue

    print(f"\nStrategy breakdown: gold_vs_sampled = {n_strategy_a} | "
          f"self_diversity = {n_strategy_b} | total = {len(pairs)}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} preference pairs → {out_path}")
    if len(pairs) < 100:
        print(f"WARN: chỉ có {len(pairs)} cặp (<100). Tăng n_samples hoặc giảm min_delta.")


# ------------------------------ Human evaluation app ------------------------------

def human_eval(pred_sft: str, pred_dpo: str, n: int = 30):
    """Mini Gradio app để annotate A/B/Tie giữa B2-SFT và B2-DPO."""
    import gradio as gr
    import csv

    sft = json.loads(Path(pred_sft).read_text(encoding="utf-8"))
    dpo = json.loads(Path(pred_dpo).read_text(encoding="utf-8"))

    # Chuẩn hoá key
    def _norm_key(image_path, question):
        ip = image_path.replace("\\", "/")
        if ip.startswith("data/"):
            ip = ip[5:]
        return (ip, question.strip())

    by_key_sft = {_norm_key(r["image_path"], r["question"]): r for r in sft}
    rows = []
    n_unmatched = 0
    for r in dpo:
        key = _norm_key(r["image_path"], r["question"])
        if key not in by_key_sft:
            n_unmatched += 1
            continue
        rows.append({
            "image_path": r["image_path"],
            "question": r["question"],
            "gold": r["gold"],
            "pred_sft": by_key_sft[key]["pred"],
            "pred_dpo": r["pred"],
        })
    print(f"Matched {len(rows)} / {len(dpo)} samples (unmatched: {n_unmatched})")
    if not rows:
        raise RuntimeError(
            "Không match được sample nào giữa B2 và B2_DPO predictions!\n"
            "Kiểm tra: outputs/B2_test.json và outputs/B2_DPO_test.json có cùng "
            "schema (image_path, question, gold, pred) không."
        )

    random.Random(config.SEED).shuffle(rows)
    rows = rows[:n]
    print(f"Sẽ chấm {len(rows)} sample")

    out_csv = config.OUT_DIR / "human_eval.csv"
    state = {"idx": 0, "results": []}

    def render():
        if state["idx"] >= len(rows):
            return None, "Done!", "", "", "", "Hoàn tất tất cả sample!"
        r = rows[state["idx"]]
        # Random order A/B để blind
        if random.random() < 0.5:
            a_label, a_text = "SFT", r["pred_sft"]
            b_label, b_text = "DPO", r["pred_dpo"]
        else:
            a_label, a_text = "DPO", r["pred_dpo"]
            b_label, b_text = "SFT", r["pred_sft"]
        r["_a_label"], r["_b_label"] = a_label, b_label

        # Resolve image path qua helper
        img_path = resolve_image_path(r["image_path"])
        img_str = str(img_path) if img_path.exists() else None
        if img_str is None:
            print(f"[WARN] Không tìm thấy ảnh: {r['image_path']}")

        return (
            img_str,
            r["question"],
            r["gold"],
            f"A: {a_text}",
            f"B: {b_text}",
            f"Sample {state['idx']+1}/{len(rows)}",
        )

    def vote(choice):
        r = rows[state["idx"]]
        if state["idx"] < len(rows):
            winner = "Tie" if choice == "Tie" else r[f"_{choice.lower()}_label"]
            state["results"].append({
                "image_path": r["image_path"], "question": r["question"],
                "winner": winner, "choice": choice,
            })
            state["idx"] += 1
            with out_csv.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=state["results"][0].keys())
                w.writeheader()
                w.writerows(state["results"])
        return render()

    with gr.Blocks(title="VQA Human Eval — SFT vs DPO") as app:
        gr.Markdown("# Đánh giá thủ công: B2-SFT vs B2-DPO\nChọn A, B, hoặc Tie.")
        progress = gr.Textbox(label="Progress", interactive=False)
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="filepath", label="Image", height=320)
                q = gr.Textbox(label="Câu hỏi", interactive=False)
                gold = gr.Textbox(label="Đáp án vàng", interactive=False)
            with gr.Column(scale=1):
                ans_a = gr.Textbox(label="Đáp án A", interactive=False)
                ans_b = gr.Textbox(label="Đáp án B", interactive=False)
                with gr.Row():
                    btn_a = gr.Button("A tốt hơn")
                    btn_t = gr.Button("Tie")
                    btn_b = gr.Button("B tốt hơn")

        outputs = [img, q, gold, ans_a, ans_b, progress]
        app.load(render, outputs=outputs)
        btn_a.click(lambda: vote("A"), outputs=outputs)
        btn_t.click(lambda: vote("Tie"), outputs=outputs)
        btn_b.click(lambda: vote("B"), outputs=outputs)

    # Cho phép Gradio serve ảnh từ các vị trí ngoài working directory
    allowed = [
        "/content/images",
        "/content",
        str(config.IMAGES_ROOT),
        str(config.ROOT),
    ]
    # Dedupe
    seen = set()
    allowed = [p for p in allowed if not (p in seen or seen.add(p))]
    print(f"Gradio allowed_paths: {allowed}")
    app.launch(share=True, allowed_paths=allowed)


# ------------------------------ Dispatcher ------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["build", "humans"])
    # build
    p.add_argument("--adapter", default=str(config.CKPT_PATHS["B2"]))
    p.add_argument("--n_samples", type=int, default=150)
    p.add_argument("--n_candidates", type=int, default=4)
    p.add_argument("--min_delta", type=float, default=0.1)
    p.add_argument("--out", default=None)
    # humans
    p.add_argument("--pred_sft", default=str(config.PRED_PATHS["B2"]))
    p.add_argument("--pred_dpo", default=str(config.PRED_PATHS["B2_DPO"]))
    p.add_argument("--n", type=int, default=30)
    args = p.parse_args()

    if args.task == "build":
        build_preferences(args.adapter, args.n_samples, args.n_candidates,
                          args.min_delta, args.out)
    else:
        human_eval(args.pred_sft, args.pred_dpo, args.n)


if __name__ == "__main__":
    main()
