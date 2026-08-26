from __future__ import annotations

import json
import sys
from pathlib import Path

import ijson

# Inputs
RAW_AM_THUC = Path(r"d:\..DL\final_DL\am_thuc_only\vietnamese_vqa_dataset_am_thuc.json")
GEN_SCRIPT = Path(r"d:\..DL\final_DL\am_thuc_only\generate_new_pairs_dataset.py")

# Output (per-image JSON array, giống format gốc nhưng questions mới)
OUT_PATH = Path(r"d:\..DL\final_DL\am_thuc_only\vietnamese_vqa_dataset_am_thuc_newQA.json")


def load_generator_module():
    import importlib.util

    module_name = "am_thuc_gen_pairs_fullmerge"
    spec = importlib.util.spec_from_file_location(module_name, GEN_SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def main() -> None:
    gen = load_generator_module()

    # Stream write JSON array to keep memory low
    n_images = 0
    n_questions = 0

    with RAW_AM_THUC.open("rb") as f, OUT_PATH.open("w", encoding="utf-8") as w:
        w.write("[\n")
        first = True
        for item in ijson.items(f, "item"):
            # Generate pairs for this image, then group back into questions list
            pairs = gen.make_pairs_for_image(item)
            questions = []
            for idx, p in enumerate(pairs, start=1):
                questions.append(
                    {
                        "question_id": idx,
                        "question": p["question"],
                        "answer": p["answer"],
                        "question_type": p["question_type"],
                    }
                )

            # Replace questions only; keep all other fields unchanged
            item["questions"] = questions

            if not first:
                w.write(",\n")
            first = False
            json.dump(item, w, ensure_ascii=False)

            n_images += 1
            n_questions += len(questions)

        w.write("\n]\n")

    print(json.dumps({"out": str(OUT_PATH), "images": n_images, "questions": n_questions}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

