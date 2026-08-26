# Vietnamese Food VQA

Hệ thống **Visual Question Answering (VQA) tiếng Việt** cho miền ẩm thực Việt Nam - so sánh hướng tự thiết kế (ViT + PhoBERT) với foundation VLM (Qwen2-VL), kèm fine-tune LoRA và Direct Preference Optimization (DPO).

---

## Architecture

```
Hướng A                             Hướng B
────────                            ────────
   Ảnh  ──► ViT ──────┐                Ảnh + câu hỏi
Câu hỏi ──► PhoBERT ──┤                ──► Qwen2-VL-2B
                Cross - Attn         · B1: zero-shot
                  ┌───┴───┐          · B2: LoRA SFT
                 LSTM     TF         · B2-DPO: preference + DPO
                 (A1)    (A2)
```


| Cấu hình   | Mô tả                                                                                 |
| ---------- | ------------------------------------------------------------------------------------- |
| **A1**     | `vit_base_patch16_224` + PhoBERT-base + Cross-Attention + **LSTM decoder**            |
| **A2**     | Cùng encoder/fusion/vocab/seed với A1 + **Transformer decoder** (cô lập biến so sánh) |
| **B1**     | Qwen2-VL-2B-Instruct **zero-shot**                                                    |
| **B2**     | Qwen2-VL-2B-Instruct **LoRA SFT**                                                     |
| **B2-DPO** | Tiếp tục từ B2 bằng **Direct Preference Optimization** (≥ 100 cặp preference)         |


**DPO (manual loop, không dùng `DPOTrainer`):**

1. Log-prob chosen/rejected với policy (LoRA bật)
2. `disable_adapter()` → log-prob reference (base)
3. Loss: `-log σ(β · [(log π_c − log π_ref_c) − (log π_r − log π_ref_r)])`

Preference data kết hợp 2 chiến lược: `gold_vs_sampled` + `self_diversity`.

---

## Results

Kết quả trên **test set** (tính trong `visualize.ipynb`):


| Config         | EM        | BLEU      | ROUGE-L   | BERTScore |
| -------------- | --------- | --------- | --------- | --------- |
| **A1**         | **0.447** | 0.632     | 0.845     | 0.957     |
| **A2**         | 0.441     | **0.638** | **0.848** | 0.957     |
| B1 (zero-shot) | 0.007     | 0.073     | 0.364     | 0.855     |
| B2 (LoRA SFT)  | 0.393     | 0.469     | 0.751     | 0.951     |
| **B2-DPO**     | 0.436     | 0.590     | 0.813     | **0.969** |


### Insights

1. **A1 ≈ A2** trên hầu hết metric — LSTM và Transformer decoder gần ngang khi cùng encoder/fusion.
2. **B1 zero-shot rất yếu** trên miền ẩm thực VN + ràng buộc trả lời ngắn → cần adaptation.
3. **B2 → B2-DPO** cải thiện rõ EM / BLEU / ROUGE-L / BERTScore; human eval nghiêng về DPO.
4. Model tự dựng (A) vẫn cạnh tranh tốt trên metric lexical/overlap so với VLM đã SFT.

---

## Dataset

| Split    | Số cặp QA   |
| -------- | ----------- |
| Train    | 30 357      |
| Val      | 3 790       |
| Test     | 3 793       |
| **Tổng** | **~37 940** |


---

## Link download data & checkpoints

Repo chỉ chứa **source + splits + predictions**.

| Tài nguyên | Nội dung gợi ý | Link |
|---|---|---|
| **Images** | Folder `images/` (~2.9k ảnh, ~1 GB) | https://drive.google.com/drive/folders/18gnethh_qo9SqluF38fzS6e_e3t4_w4y?usp=sharing |
| **Checkpoints** | `A1/`, `A2/`, `B2_lora/`, `B2_DPO_lora/` | https://drive.google.com/drive/folders/1ugbdBqc9NBztNkeCPyZ8zwv4S7GwumWb?usp=sharing |

---



## Repository structure

```
VQA_food/
├── config.py              # Hyperparameter + đường dẫn
├── dataset.py             # VQADataset (A) + VQARawDataset (B) + resolve_image_path
├── models.py              # Hướng A (ViT+PhoBERT+…) + Hướng B (Qwen-VL / LoRA)
├── train.py               # --mode a1 | a2 | b2_sft | b2_dpo
├── infer.py               # --mode a1 | a2 | b1 | b2 | b2_dpo
├── rl.py                  # Build preferences + Gradio human-eval
├── demo.py                # Gradio: A1 / A2 / B1 / B2
├── visualize.ipynb        # Metrics, biểu đồ, so sánh qualitative
├── keys.py                # API keys (local only — không commit)
├── requirements.txt
├── README.md
├── DEMO_1.mp4
│
├── splits/                # train / val / test pairs
├── images/                # Ảnh theo folder món
├── pre_pro_data/          # Script + JSON preprocess (tuỳ chọn)
├── checkpoints/           # A1, A2, B2_lora, B2_DPO_lora
└── outputs/               # Predictions, preferences, human_eval, logs, runs
```

---



## Installation



### Yêu cầu phần cứng


| Thành phần   | VRAM gợi ý         |
| ------------ | ------------------ |
| A1 / A2      | ~10–12 GB          |
| B2 LoRA SFT  | ~12–14 GB          |
| B2-DPO       | ~13–15 GB          |
| B1 zero-shot | T4 hoặc CPU (chậm) |


Đề xuất: Colab T4 / Kaggle GPU / RTX 3060+ tương đương.

### Cài đặt

```bash
cd VQA_food
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

**Colab / Kaggle (rút gọn):**

```bash
!pip install -q timm peft trl rouge-score google-genai gradio underthesea sacrebleu bert-score qwen-vl-utils
```



### Đường dẫn ảnh

1. Tải `images/` từ [Google Drive](#download-data--checkpoints-google-drive) vào thư mục project.
2. `dataset.py` resolve ảnh qua `config.IMAGES_ROOT` và fallback về `VQA_food/`.  
   - **Local:** để `images/` trong project (hoặc sửa `IMAGES_ROOT` trong `config.py`).  
   - **Colab:** mount Drive rồi chỉnh `IMAGES_ROOT` cho khớp path ảnh.

### API key (tuỳ chọn — LLM-as-a-judge)

Chỉ cần nếu bạn tự gắn Gemini judge. Tạo / sửa `keys.py`:

```python
GOOGLE_API_KEYS = [
    "YOUR_KEY_1",
    "YOUR_KEY_2",
]
```

Hoặc:

```bash
# Windows PowerShell
$env:GOOGLE_API_KEY = "key1,key2"
```

---



## Quickstart

> Chưa có `images/` / `checkpoints/`? Xem [Download data & checkpoints](#download-data--checkpoints-google-drive).

### 1. Kiểm tra data

```bash
python -c "import json; d=json.load(open('splits/train_pairs.json',encoding='utf-8')); print('train pairs:', len(d))"
```



### 2. Training

```bash
python train.py --mode a1          # → checkpoints/A1/best_A1.pt
python train.py --mode a2          # → checkpoints/A2/best_A2.pt
python train.py --mode b2_sft      # → checkpoints/B2_lora/
```

**DPO:**

```bash
python rl.py --task build --n_samples 200   # → outputs/preferences.jsonl
python train.py --mode b2_dpo               # → checkpoints/B2_DPO_lora/
```

Train có `tqdm` 2 tầng, log `outputs/logs/`, TensorBoard `outputs/runs/`.  
A1/A2: early stopping + LR plateau theo BLEU val.

### 3. Inference

```bash
python infer.py --mode a1       # → outputs/A1_test.json
python infer.py --mode a2
python infer.py --mode b1       # zero-shot, không cần ckpt
python infer.py --mode b2
python infer.py --mode b2_dpo
```

Tuỳ chọn: `--ckpt`, `--adapter`, `--split val`, `--out path.json`.

### 4. Đánh giá

```bash
jupyter notebook visualize.ipynb
```

Notebook gồm: bảng metrics, biểu đồ cột, sample trực quan, A1 vs A2, B1 vs B2, per-question-type / per-món, B2 vs B2-DPO.

### 5. Human eval & Demo

```bash
python rl.py --task humans --n 30   # → outputs/human_eval.csv
python demo.py
```

**TensorBoard:**

```bash
tensorboard --logdir outputs/runs --port 6006
```



### Pipeline end-to-end (tóm tắt)

```bash
python train.py --mode a1 && python train.py --mode a2 && python train.py --mode b2_sft
python infer.py --mode a1 && python infer.py --mode a2 && python infer.py --mode b1 && python infer.py --mode b2
python rl.py --task build --n_samples 200
python train.py --mode b2_dpo && python infer.py --mode b2_dpo
# mở visualize.ipynb → Run All
python rl.py --task humans --n 30
python demo.py
```

---



## Evaluation details


| Metric        | Ghi chú                              |
| ------------- | ------------------------------------ |
| **EM**        | Exact match sau normalize            |
| **BLEU**      | `sacrebleu` corpus BLEU              |
| **ROUGE-L**   | `rouge-score`                        |
| **BERTScore** | `xlm-roberta-base` (đa ngữ)          |
| **Human**     | Blind A/B (SFT vs DPO), shuffle nhãn |


