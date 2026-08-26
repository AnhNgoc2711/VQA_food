"""
Generate a new VQA dataset for `am_thuc` by paraphrasing questions and
building answers ONLY from existing fields (no hallucination).

Input format (per image):
  - keyword, image_id, image_path
  - image_analysis.main_objects
  - image_analysis.visual_details.colors
  - image_analysis.visual_details.materials
  - image_analysis.visual_details.composition

Output format (JSONL):
  - one line per (image, question, answer)

Rules from user:
  - 2 QA per question type, except dish identification => 1 QA
  - answers should be natural sentences
  - do NOT add new facts; only rephrase / truncate existing facts
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import ijson


BASE_DIR = Path(r"d:\..DL\final_DL\am_thuc_only")
SPLITS_DIR = BASE_DIR / "splits"
OUT_DIR = BASE_DIR / "new_pairs"

# Input split files (per-image records)
SPLIT_IN = {
    "train": SPLITS_DIR / "train_images_am_thuc.json",
    "val": SPLITS_DIR / "val_images_am_thuc.json",
    "test": SPLITS_DIR / "test_images_am_thuc.json",
}


MAX_ANSWER_WORDS = 10  # project constraint; safe default


def norm_space(s: str | None) -> str:
    # Some raw fields may be non-strings (e.g., list/null). Convert safely.
    if s is None:
        return ""
    if isinstance(s, list):
        s = " ".join(str(x) for x in s if x is not None)
    elif not isinstance(s, str):
        s = str(s)
    return re.sub(r"\s+", " ", s.strip())


def words(s: str) -> list[str]:
    return [w for w in norm_space(s).split(" ") if w]


def cap_first(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    return s[:1].upper() + s[1:]


def truncate_to_word_limit(text: str, max_words: int = MAX_ANSWER_WORDS) -> str:
    w = words(text)
    w = w[:max_words]
    # Avoid dangling function words after truncation
    while w and w[-1].lower() in {"và", "của", "như", "là", "ở", "trong", "các", "những", "một"}:
        w.pop()
    out = cap_first(" ".join(w)).rstrip(".").strip()
    return (out + ".") if out else ""


def shorten_phrase_keep_meaning(s: str) -> str:
    """
    Removes parenthetical notes and trailing clauses after commas.
    This only deletes information, never adds.
    """
    s = norm_space(s)
    if not s:
        return s
    s = re.split(r"[;]", s)[0].strip()
    s = re.split(r"\(", s)[0].strip()
    # keep comma-separated chunks but avoid very long strings
    return s


def join_items_natural(items: list[str], max_items: int = 3) -> str:
    items = [shorten_phrase_keep_meaning(x) for x in items]
    items = [norm_space(x) for x in items if norm_space(x)]
    if not items:
        return ""
    items = items[:max_items]
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} và {items[1]}"
    return ", ".join(items[:-1]) + f" và {items[-1]}"


def normalize_sentence(sentence: str) -> str:
    # Keep sentence natural; only enforce word limit and punctuation.
    return truncate_to_word_limit(sentence)


@dataclass(frozen=True)
class Templates:
    dish_q: tuple[str, ...] = (
        "Trong ảnh là món gì?",
        "Bạn thấy món gì trong ảnh?",
        "Món trong ảnh là gì?",
        "Đây là món gì?",
        "Trong ảnh là gì?",
    )
    objects_q: tuple[str, ...] = (
        "Trong ảnh có những gì?",
        "Bạn nhìn thấy những gì trong ảnh?",
        "Trong ảnh gồm những thành phần nào?",
        "Trong ảnh có những thành phần nào?",
    )
    materials_q: tuple[str, ...] = (
        "Các nguyên liệu trong ảnh là gì?",
        "Món ăn có những nguyên liệu nào?",
        "Trong ảnh có những nguyên vật liệu gì?",
    )
    colors_q: tuple[str, ...] = (
        "Trong ảnh có những màu nào?",
        "Ảnh có những màu đặc trưng nào?",
        "Bạn thấy những màu gì trong ảnh?",
    )
    composition_q: tuple[str, ...] = (
        "Ảnh được chụp từ góc nào?",
        "Góc chụp của bức ảnh là gì?",
        "Bức ảnh được chụp theo góc nào?",
    )
    yesno_obj_q: tuple[str, ...] = (
        "Trong ảnh có {obj} không?",
        "Bạn có thấy {obj} trong ảnh không?",
    )
    yesno_color_q: tuple[str, ...] = (
        "Trong ảnh có màu {color} không?",
        "Bạn có thấy màu {color} trong ảnh không?",
    )
    dish_a: tuple[str, ...] = (
        "Đây là {dish}.",
        "Trong ảnh là {dish}.",
        "Món trong ảnh là {dish}.",
    )
    objects_a: tuple[str, ...] = (
        "Trong ảnh có {items}.",
        "Bạn có thể thấy {items} trong ảnh.",
    )
    materials_a: tuple[str, ...] = (
        "Món này có {items}.",
        "Nguyên liệu gồm {items}.",
    )
    colors_a: tuple[str, ...] = (
        "Trong ảnh có các màu {items}.",
        "Các màu nổi bật là {items}.",
    )
    composition_a: tuple[str, ...] = (
        "Ảnh có {comp}.",
        "Góc chụp là {comp}.",
    )


T = Templates()


def base_color_token(color_phrase: str) -> str:
    """
    Extract a base color token (e.g., 'vàng', 'xanh', 'nâu') from a raw phrase.
    Only removes info; never invents.
    """
    c = norm_space(color_phrase).lower()
    # common Vietnamese color tokens
    for tok in ["xanh", "đỏ", "vàng", "nâu", "trắng", "đen", "cam", "hồng", "tím", "xám", "be", "kem", "bạc"]:
        if tok in c:
            return tok
    # fallback: first word
    return c.split(" ")[0] if c else ""


def parse_item_fields(item: dict) -> dict:
    ia = item.get("image_analysis", {}) or {}
    vd = ia.get("visual_details", {}) or {}
    return {
        "image_id": item.get("image_id"),
        "image_path": item.get("image_path"),
        "keyword": norm_space(item.get("keyword")),
        "main_objects": [norm_space(x) for x in (ia.get("main_objects") or []) if norm_space(x)],
        "colors_raw": [norm_space(x) for x in (vd.get("colors") or []) if norm_space(x)],
        "materials_raw": [norm_space(x) for x in (vd.get("materials") or []) if norm_space(x)],
        "composition_raw": norm_space(vd.get("composition")),
    }


def pick_first_nonempty(items: list[str]) -> str:
    for x in items:
        x = norm_space(x)
        if x:
            return x
    return ""


def make_pairs_for_image(item: dict) -> list[dict]:
    f = parse_item_fields(item)

    image_id = f["image_id"]
    image_path = f["image_path"]

    dish = f["keyword"] or pick_first_nonempty(f["main_objects"]) or "món ăn"
    main_objects = f["main_objects"]

    # Clean colors/materials: keep base tokens if raw is long
    colors_base = [base_color_token(c) for c in f["colors_raw"]]
    colors_base = [c for c in colors_base if c]
    # unique preserve order
    seen = set()
    colors_base_u = []
    for c in colors_base:
        if c not in seen:
            seen.add(c)
            colors_base_u.append(c)

    materials_short = [shorten_phrase_keep_meaning(m) for m in f["materials_raw"]]
    materials_short = [m for m in materials_short if m]
    seen = set()
    materials_u = []
    for m in materials_short:
        if m.lower() not in seen:
            seen.add(m.lower())
            materials_u.append(m)

    composition = f["composition_raw"]

    pairs: list[dict] = []
    qid = 1

    # 1) dish identification (1 QA)
    q = T.dish_q[qid % len(T.dish_q)]
    a_t = T.dish_a[qid % len(T.dish_a)]
    a = normalize_sentence(a_t.format(dish=dish))
    pairs.append(
        {
            "image_id": image_id,
            "image_path": image_path,
            "question_type": "name_dish",
            "question": q,
            "answer": a,
        }
    )
    qid += 1

    # 2) objects_list (2 QA)
    obj_phrase = join_items_natural(main_objects, max_items=3) or dish
    for idx in range(2):
        q = T.objects_q[(qid + idx) % len(T.objects_q)]
        a_t = T.objects_a[(qid + idx) % len(T.objects_a)]
        a = normalize_sentence(a_t.format(items=obj_phrase))
        pairs.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "question_type": "objects_list",
                "question": q,
                "answer": a,
            }
        )
        qid += 1

    # 3) materials_list (2 QA)
    mat_phrase = join_items_natural(materials_u, max_items=3)
    if mat_phrase:
        for idx in range(2):
            q = T.materials_q[(qid + idx) % len(T.materials_q)]
            a_t = T.materials_a[(qid + idx) % len(T.materials_a)]
            a = normalize_sentence(a_t.format(items=mat_phrase))
            pairs.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "question_type": "materials_list",
                    "question": q,
                    "answer": a,
                }
            )
            qid += 1

    # 4) colors_list (2 QA)
    col_phrase = join_items_natural(colors_base_u, max_items=3)
    if col_phrase:
        for idx in range(2):
            q = T.colors_q[(qid + idx) % len(T.colors_q)]
            a_t = T.colors_a[(qid + idx) % len(T.colors_a)]
            a = normalize_sentence(a_t.format(items=col_phrase))
            pairs.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "question_type": "colors_list",
                    "question": q,
                    "answer": a,
                }
            )
            qid += 1

    # 5) composition/camera angle (2 QA) - only if raw exists and mentions a shot/angle
    if composition:
        # Keep only the leading part to avoid long/subjective claims
        comp_short = composition.split(",")[0].strip()
        # If it doesn't mention angle/shot, keep but still from raw
        for idx in range(2):
            q = T.composition_q[(qid + idx) % len(T.composition_q)]
            a_t = T.composition_a[(qid + idx) % len(T.composition_a)]
            a = normalize_sentence(a_t.format(comp=comp_short))
            pairs.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "question_type": "composition",
                    "question": q,
                    "answer": a,
                }
            )
            qid += 1

    # 6) yes/no (2 QA) - only TRUE statements (avoid negatives that could be missing in annotations)
    # pick up to 1 object + 1 color if available
    if main_objects:
        obj = main_objects[0]
        q = T.yesno_obj_q[qid % len(T.yesno_obj_q)].format(obj=obj)
        a = truncate_to_word_limit(f"Trong ảnh có {obj}.")
        pairs.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "question_type": "yes_no_object",
                "question": q,
                "answer": a,
            }
        )
        qid += 1
        if len(main_objects) > 1:
            obj2 = main_objects[1]
            q = T.yesno_obj_q[qid % len(T.yesno_obj_q)].format(obj=obj2)
            a = truncate_to_word_limit(f"Trong ảnh có {obj2}.")
            pairs.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "question_type": "yes_no_object",
                    "question": q,
                    "answer": a,
                }
            )
            qid += 1

    if colors_base_u:
        c = colors_base_u[0]
        q = T.yesno_color_q[qid % len(T.yesno_color_q)].format(color=c)
        a = truncate_to_word_limit(f"Trong ảnh có màu {c}.")
        pairs.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "question_type": "yes_no_color",
                "question": q,
                "answer": a,
            }
        )
        qid += 1
        if len(colors_base_u) > 1:
            c2 = colors_base_u[1]
            q = T.yesno_color_q[qid % len(T.yesno_color_q)].format(color=c2)
            a = truncate_to_word_limit(f"Trong ảnh có màu {c2}.")
            pairs.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "question_type": "yes_no_color",
                    "question": q,
                    "answer": a,
                }
            )
            qid += 1

    return pairs


def iter_json_array(path: Path) -> Iterable[dict]:
    with path.open("rb") as f:
        yield from ijson.items(f, "item")


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split, in_path in SPLIT_IN.items():
        out_path = OUT_DIR / f"{split}_pairs_new.jsonl"

        def gen_rows():
            for item in iter_json_array(in_path):
                for pair in make_pairs_for_image(item):
                    pair["split"] = split
                    summary.setdefault("by_type", {}).setdefault(pair["question_type"], 0)
                    summary["by_type"][pair["question_type"]] += 1
                    yield pair

        n = write_jsonl(out_path, gen_rows())
        summary[split] = {"pairs": n, "out": str(out_path)}

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

