"""
Stage 7 步骤1（通用版）：Token 跨度对齐
用法：python build_span_alignment_generic.py <model_path> <model_tag>

示例：
  python build_span_alignment_generic.py D:/models/Ministral-8B-Instruct-2410 ministral8b
  python build_span_alignment_generic.py D:/models/gemma-3-4b-it gemma3_4b
"""
import sys, json, re, csv
from pathlib import Path
from transformers import AutoTokenizer

if len(sys.argv) < 3:
    print("用法: python build_span_alignment_generic.py <model_path> <model_tag>")
    sys.exit(1)

MODEL_PATH = sys.argv[1]
MODEL_TAG  = sys.argv[2]   # e.g. "ministral8b", "gemma3_4b"

PACK_DIR = Path("d:/pps/paperSELFatten/ist_attention_v1_runs/06_stage6_full")
OUT_DIR  = Path(f"d:/pps/paperSELFatten/ist_attention_v1_runs/07_stage7_{MODEL_TAG}")
OUT_DIR.mkdir(exist_ok=True)

DIM_PATTERNS = [
    ("WHAT",      r"任务目标\s*\(What\)\s*:"),
    ("WHY",       r"执行原因\s*\(Why\)\s*:"),
    ("WHO",       r"执行角色\s*\(Who\)\s*:"),
    ("WHEN",      r"时间安排\s*\(When\)\s*:"),
    ("WHERE",     r"执行场所\s*\(Where\)\s*:"),
    ("HOW_TO_DO", r"执行方法\s*\(How to do\)\s*:"),
    ("HOW_MUCH",  r"量化要素\s*\(How much\)\s*:"),
    ("HOW_FEEL",  r"预期效果\s*\(How feel\)\s*:"),
]
TAIL_PAT     = r"请按照以上内容执行任务"
PRIVATE_DIMS = {"WHO", "HOW_MUCH"}
PUBLIC_DIMS  = {"HOW_TO_DO"}


def find_dim_spans(text: str) -> dict:
    anchors = []
    for dim, pat in DIM_PATTERNS:
        m = re.search(pat, text)
        if m:
            anchors.append((m.end(), dim))
    anchors.sort(key=lambda x: x[0])

    tail_m = re.search(TAIL_PAT, text)
    tail_start = tail_m.start() if tail_m else len(text)

    spans = {}
    for i, (start, dim) in enumerate(anchors):
        if i + 1 < len(anchors):
            next_start = anchors[i + 1][0]
            end = next_start
            for _, p in DIM_PATTERNS:
                m2 = re.search(p, text[max(0, next_start - 40):next_start + 5])
                if m2:
                    end = max(0, next_start - 40) + m2.start()
                    break
        else:
            end = tail_start
        spans[dim] = (start, end)
    return spans


def char_span_to_token_span(enc, char_start: int, char_end: int):
    offsets = enc["offset_mapping"]
    t_start = t_end = None
    for i, (cs, ce) in enumerate(offsets):
        if cs >= char_end:
            break
        if ce > char_start:
            if t_start is None:
                t_start = i
            t_end = i + 1
    return (0, 0) if t_start is None else (t_start, t_end)


def main():
    print(f"加载 tokenizer：{MODEL_PATH}")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    with open(PACK_DIR / "full_manifest.csv", encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))

    qwen_rows = [r for r in all_rows if "qwen" in r["model"].lower()]
    print(f"处理 Qwen3 prompt（共 {len(qwen_rows)} 条）— 与原始 36 个 prompt 相同，用新 tokenizer 对齐")

    out_path = OUT_DIR / "span_alignment_log.jsonl"
    written = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for row in qwen_rows:
            sid = row["sample_id"]
            prompt_path = PACK_DIR / "prompts" / f"{sid}__prompt.txt"
            prompt_text = prompt_path.read_text(encoding="utf-8")

            enc = tok(prompt_text, return_offsets_mapping=True, add_special_tokens=True)
            total_tokens = len(enc["input_ids"])

            char_spans = find_dim_spans(prompt_text)
            token_spans = {}
            for dim, (cs, ce) in char_spans.items():
                ts, te = char_span_to_token_span(enc, cs, ce)
                token_spans[dim] = {
                    "char": [cs, ce], "token": [ts, te],
                    "len_tok": te - ts,
                    "role": ("private" if dim in PRIVATE_DIMS
                             else "public" if dim in PUBLIC_DIMS else "other"),
                }

            record = {
                "sample_id": sid, "condition": row["condition"],
                "total_tokens": total_tokens,
                "dims": token_spans, "present_dims": list(token_spans.keys()),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            present = ", ".join(
                f"{d}[{v['token'][0]}:{v['token'][1]}]"
                for d, v in token_spans.items()
            )
            print(f"  {sid}  总tokens={total_tokens}  {present}")

    print(f"\n✅ 完成 → {out_path}  ({written} 条)")


if __name__ == "__main__":
    main()
