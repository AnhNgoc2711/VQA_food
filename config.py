"""Tập trung mọi hyperparameter và đường dẫn."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLITS_DIR = ROOT / "splits"
IMAGES_ROOT = Path("/content")
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "outputs"
RUNS_DIR = OUT_DIR / "runs"
LOG_DIR = OUT_DIR / "logs"

for _d in (CKPT_DIR, OUT_DIR, RUNS_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

IMG_SIZE = 224
MAX_Q_LEN = 32
MAX_A_LEN = 16

PHOBERT_NAME = "vinai/phobert-base"
VIT_NAME = "vit_base_patch16_224"

# A1, A2 chia sẻ encoder + fusion + tokenizer; chỉ khác decoder
A_COMMON = dict(
    d_model=768,
    fusion_heads=8,
    fusion_layers=2,
    bs=32,
    epochs=20,                    
    lr_enc=2e-5,
    lr_dec=5e-4,
    weight_decay=0.01,
    label_smoothing=0.1,
    warmup_ratio=0.05,
    grad_clip=1.0,
    log_every=200,
    eval_every_epoch=1,
    num_workers=2,
    # Early stopping
    early_stop_patience=5,          # dừng nếu BLEU val không tăng sau N epoch liên tiếp
    early_stop_min_delta=0.001,     # cải thiện < 0.001 thì coi như không cải thiện
    # LR plateau
    lr_plateau_patience=2,          # giảm LR sau N epoch không cải thiện
    lr_plateau_factor=0.5,          # nhân LR với 0.5 khi plateau
)

A1 = {**A_COMMON, "decoder": "lstm", "hidden": 512, "n_layers": 2, "dropout": 0.3}
A2 = {**A_COMMON, "decoder": "tf", "n_heads": 8, "n_layers": 4, "ff": 2048,
      "dropout": 0.1, "lr_dec": 3e-4}

# Qwen-VL
B = dict(
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    max_new_tokens=20,
)

B2_SFT = dict(
    lora_r=8,                   
    lora_alpha=16,                
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],   # chỉ q_proj + v_proj (đủ cho LoRA cơ bản)
    lr=2e-4,            
    bs=1,
    grad_accum=8,           
    epochs=1,                  
    warmup_ratio=0.03,
    weight_decay=0.01,
    max_length=512,
    grad_checkpointing=False,      # tắt để nhanh hơn ~30% (đã có image resize đủ)
    image_max_pixels=336 * 336,
    train_subset=10000,             # CHỈ DÙNG 10000 samples (random, seed 42)
    log_every=500,               
)

DPO_CFG = dict(
    beta=0.1,
    lr=5e-6,
    bs=2,
    grad_accum=8,
    epochs=3,
    max_length=512,
    max_prompt_length=256,
)

LLM_JUDGE = dict(
    n_samples=200,
    provider="gemini",                 
    model="gemini-2.0-flash",              
    sleep_per_request=0.5,                 
)

PROMPT_VI = (
    "Câu hỏi: {question}\n"
    "Trả lời ngắn gọn bằng tiếng Việt, tối đa 10 từ."
)

# Path predictions cho mỗi cấu hình
PRED_PATHS = {
    "A1": OUT_DIR / "A1_test.json",
    "A2": OUT_DIR / "A2_test.json",
    "B1": OUT_DIR / "B1_test.json",
    "B2": OUT_DIR / "B2_test.json",
    "B2_DPO": OUT_DIR / "B2_DPO_test.json",
}

CKPT_PATHS = {
    "A1": CKPT_DIR / "A1",
    "A2": CKPT_DIR / "A2",
    "B2": CKPT_DIR / "B2_lora",
    "B2_DPO": CKPT_DIR / "B2_DPO_lora",
}
