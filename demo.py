"""Gradio demo: 1 trang — upload ảnh + nhập câu hỏi → in answer của CẢ 4 model."""
from __future__ import annotations

import time

import gradio as gr
import torch
from PIL import Image

import config


_CACHE = {}        # cache models theo tên: {"A1": (model, tok), ...}
_TOKENIZER = None  # PhoBERT tokenizer dùng chung A1, A2


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _trim_10(text: str) -> str:
    return " ".join(text.split()[:10])


# ------------------------------ Lazy loaders ------------------------------

def _load_a(name: str):
    if name in _CACHE:
        return _CACHE[name]
    from transformers import AutoTokenizer
    from models import VQAModelA

    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(config.PHOBERT_NAME)
    tok = _TOKENIZER
    pad = tok.pad_token_id
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id

    state = torch.load(config.CKPT_PATHS[name] / f"best_{name}.pt", map_location=device())
    model = VQAModelA(state["decoder_type"], vocab_size=tok.vocab_size,
                      cfg=state["cfg"], pad_id=pad, bos_id=bos, eos_id=eos)
    model.load_state_dict(state["model"])
    model.to(device()).eval()
    _CACHE[name] = (model, tok)
    return _CACHE[name]


def _load_b(name: str):
    if name in _CACHE:
        return _CACHE[name]
    from models import load_qwenvl
    adapter = None if name == "B1" else config.CKPT_PATHS["B2"]
    model, processor = load_qwenvl(adapter_path=adapter)
    _CACHE[name] = (model, processor)
    return _CACHE[name]


# ------------------------------ Inference ------------------------------

def _infer_a(name: str, image: Image.Image, question: str) -> str:
    from dataset import vn_segment, detok_vn, make_image_transform
    model, tok = _load_a(name)
    pixel = make_image_transform(train=False)(image).unsqueeze(0).to(device())
    enc = tok(vn_segment(question), padding="max_length", truncation=True,
              max_length=config.MAX_Q_LEN, return_tensors="pt")
    batch = {
        "pixel_values": pixel,
        "q_input_ids": enc["input_ids"].to(device()),
        "q_attn_mask": enc["attention_mask"].to(device()),
    }
    with torch.no_grad():
        ids = model.generate(batch, max_len=config.MAX_A_LEN)
    return _trim_10(detok_vn(tok, ids[0].tolist()))


def _infer_b(name: str, image: Image.Image, question: str) -> str:
    from models import generate_answer
    model, processor = _load_b(name)
    return _trim_10(generate_answer(model, processor, image, question))


def _safe_call(name: str, fn):
    """Bao bọc 1 inference: trả về (answer, latency_ms)."""
    t0 = time.time()
    try:
        ans = fn()
    except FileNotFoundError as e:
        return f"[Chưa có checkpoint cho {name}] {e}", 0.0
    except Exception as e:
        return f"[Lỗi {name}] {type(e).__name__}: {e}", 0.0
    return ans, (time.time() - t0) * 1000


# ------------------------------ Main ------------------------------

def answer_all(image: Image.Image, question: str):
    """Trả về answer của 4 model: A1, A2, B1, B2."""
    if image is None:
        msg = "Vui lòng upload ảnh."
        return msg, msg, msg, msg
    if not question or not question.strip():
        msg = "Vui lòng nhập câu hỏi."
        return msg, msg, msg, msg

    a1, t1 = _safe_call("A1", lambda: _infer_a("A1", image, question))
    a2, t2 = _safe_call("A2", lambda: _infer_a("A2", image, question))
    b1, t3 = _safe_call("B1", lambda: _infer_b("B1", image, question))
    b2, t4 = _safe_call("B2", lambda: _infer_b("B2", image, question))

    return (
        f"{a1}\n\n⏱ {t1:.0f} ms",
        f"{a2}\n\n⏱ {t2:.0f} ms",
        f"{b1}\n\n⏱ {t3:.0f} ms",
        f"{b2}\n\n⏱ {t4:.0f} ms",
    )


def build_app():
    with gr.Blocks(title="Vietnamese Food VQA — So sánh 4 model") as app:
        gr.Markdown(
            "# VQA — Ẩm thực Việt Nam\n"
            "Upload ảnh + nhập câu hỏi"
        )

        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="pil", label="Ảnh món ăn", height=380)
                q = gr.Textbox(label="Câu hỏi (tiếng Việt)",
                               placeholder="Ví dụ: Trong ảnh có gì?",
                               lines=2)
                btn = gr.Button("Trả lời", variant="primary")

            with gr.Column(scale=1):
                with gr.Row():
                    a1_box = gr.Textbox(label="A1 — ViT + PhoBERT + LSTM", lines=4)
                    a2_box = gr.Textbox(label="A2 — ViT + PhoBERT + Transformer", lines=4)
                with gr.Row():
                    b1_box = gr.Textbox(label="B1 — Qwen-VL zero-shot", lines=4)
                    b2_box = gr.Textbox(label="B2 — Qwen-VL fine-tuned", lines=4)

        outputs = [a1_box, a2_box, b1_box, b2_box]
        btn.click(answer_all, inputs=[img, q], outputs=outputs)
        q.submit(answer_all, inputs=[img, q], outputs=outputs)

    return app


if __name__ == "__main__":
    build_app().launch()
