from __future__ import annotations

import json
import random
from pathlib import Path

import ijson

# Inputs
RAW_AM_THUC = Path(r"d:\..DL\final_DL\am_thuc_only\vietnamese_vqa_dataset_am_thuc.json")
GEN_SCRIPT = Path(r"d:\..DL\final_DL\am_thuc_only\generate_new_pairs_dataset.py")

# Outputs
SPLITS_801010_DIR = Path(r"d:\..DL\final_DL\am_thuc_only\splits_80_10_10")
NEW_PAIRS_801010_DIR = Path(r"d:\..DL\final_DL\am_thuc_only\new_pairs_80_10_10")

SEED = 42


def load_all_images(path: Path) -> list[dict]:
    # 2941 images: safe to load fully
    items: list[dict] = []
    with path.open("rb") as f:
        for item in ijson.items(f, "item"):
            items.append(item)
    return items


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    SPLITS_801010_DIR.mkdir(parents=True, exist_ok=True)
    NEW_PAIRS_801010_DIR.mkdir(parents=True, exist_ok=True)

    items = load_all_images(RAW_AM_THUC)
    rng = random.Random(SEED)
    rng.shuffle(items)

    n = len(items)
    n_train = int(round(n * 0.80))
    n_val = int(round(n * 0.10))
    # ensure exact partition
    n_test = n - n_train - n_val

    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]

    write_json(SPLITS_801010_DIR / "train_images_am_thuc_80.json", train)
    write_json(SPLITS_801010_DIR / "val_images_am_thuc_10.json", val)
    write_json(SPLITS_801010_DIR / "test_images_am_thuc_10.json", test)

    # Reuse generator by invoking it as a module-like script: we just import and call functions.
    # Import generator script in a way compatible with dataclasses on Windows/Py3.13
    import importlib.util
    import sys

    module_name = "am_thuc_gen_pairs"
    spec = importlib.util.spec_from_file_location(module_name, GEN_SCRIPT)
    assert spec and spec.loader
    gen = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = gen
    spec.loader.exec_module(gen)  # type: ignore

    # gen expects per-image JSON arrays; we'll generate JSONL pairs
    split_map = {
        "train": SPLITS_801010_DIR / "train_images_am_thuc_80.json",
        "val": SPLITS_801010_DIR / "val_images_am_thuc_10.json",
        "test": SPLITS_801010_DIR / "test_images_am_thuc_10.json",
    }

    summary = {"seed": SEED, "images": {"train": len(train), "val": len(val), "test": len(test)}}

    for split, in_path in split_map.items():
        out_jsonl = NEW_PAIRS_801010_DIR / f"{split}_pairs_new_80_10_10.jsonl"
        out_json = NEW_PAIRS_801010_DIR / f"{split}_pairs_new_80_10_10.json"

        rows = []
        for item in gen.iter_json_array(in_path):
            for pair in gen.make_pairs_for_image(item):
                pair["split"] = split
                rows.append(pair)

        out_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.setdefault("pairs", {})[split] = len(rows)

    write_json(NEW_PAIRS_801010_DIR / "summary_80_10_10.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

