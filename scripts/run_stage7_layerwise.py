"""
Stage 7 逐层注意力提取（用于 heatmap）
用法：python run_stage7_layerwise.py <model_path> <model_tag>

输出：ist_attention_v1_runs/07_stage7_{tag}/stage7_layerwise_{tag}.csv
每行 = 一个 sample × 一个维度，列 = layer_0 … layer_N-1
"""
import sys, csv, json, gc
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if len(sys.argv) < 3:
    print("用法: python run_stage7_layerwise.py <model_path> <model_tag>")
    sys.exit(1)

MODEL_PATH = sys.argv[1]
MODEL_TAG  = sys.argv[2]

PACK_DIR  = Path("d:/pps/paperSELFatten/ist_attention_v1_runs/06_stage6_full")
# Qwen3 was run before the generic naming convention (uses 07_stage7/ not 07_stage7_qwen3_8b/)
_stage_dir = Path(f"d:/pps/paperSELFatten/ist_attention_v1_runs/07_stage7_{MODEL_TAG}")
if not _stage_dir.exists() and MODEL_TAG == "qwen3_8b":
    _stage_dir = Path("d:/pps/paperSELFatten/ist_attention_v1_runs/07_stage7")
SPAN_LOG  = _stage_dir / "span_alignment_log.jsonl"
OUT_CSV   = _stage_dir / f"stage7_layerwise_{MODEL_TAG}.csv"

ALL_DIMS = ["WHAT","WHY","WHO","WHEN","WHERE","HOW_TO_DO","HOW_MUCH","HOW_FEEL"]


def attention_per_layer(attentions, ts: int, te: int, last_pos: int) -> np.ndarray:
    if ts >= te:
        return np.zeros(len(attentions))
    out = []
    for layer_attn in attentions:
        a = layer_attn[0, :, last_pos, ts:te]
        out.append(float(a.mean().cpu()))
    return np.array(out)


def main():
    print(f"模型：{MODEL_PATH}  标签：{MODEL_TAG}")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    print(f"VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    with open(SPAN_LOG, encoding="utf-8") as f:
        span_records = {json.loads(l)["sample_id"]: json.loads(l) for l in f}

    with open(PACK_DIR / "full_manifest.csv", encoding="utf-8-sig") as f:
        manifest = {r["sample_id"]: r for r in csv.DictReader(f)}

    sample_ids = sorted(span_records.keys())
    n_layers = None

    rows = []
    for i, sid in enumerate(sample_ids, 1):
        rec  = span_records[sid]
        meta = manifest.get(sid, {})
        print(f"  [{i:02d}/{len(sample_ids)}] {sid}", end=" ", flush=True)

        prompt_text = (PACK_DIR / "prompts" / f"{sid}__prompt.txt").read_text(encoding="utf-8")
        inputs      = tok(prompt_text, return_tensors="pt").to("cuda")
        prompt_len  = inputs["input_ids"].shape[1]
        last_pos    = prompt_len - 1

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        if n_layers is None:
            n_layers = len(outputs.attentions)
            print(f"n_layers={n_layers}")

        dims = rec["dims"]
        # 归一化：先算每层每维度的原始 attention，再对所有维度做归一化
        raw = {}
        for dim in ALL_DIMS:
            if dim not in dims:
                continue
            ts, te = dims[dim]["token"]
            if te <= ts:
                continue
            raw[dim] = attention_per_layer(outputs.attentions, ts, te, last_pos)

        if not raw:
            del outputs; torch.cuda.empty_cache(); gc.collect(); continue

        # 每层归一化
        n_l = len(next(iter(raw.values())))
        for layer_i in range(n_l):
            layer_vals = {d: raw[d][layer_i] for d in raw}
            total = sum(layer_vals.values()) or 1.0
            for d in layer_vals:
                raw[d][layer_i] = layer_vals[d] / total

        for dim, arr in raw.items():
            row = {
                "sample_id": sid,
                "task_id":   meta.get("task_id", ""),
                "domain":    meta.get("domain", ""),
                "condition": rec["condition"],
                "model_tag": MODEL_TAG,
                "dim":       dim,
            }
            for li in range(n_l):
                row[f"L{li:03d}"] = round(float(arr[li]), 5)
            rows.append(row)

        del outputs; torch.cuda.empty_cache(); gc.collect()

    # 写出
    if rows:
        fields = list(rows[0].keys())
        with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
        print(f"\n✅ 完成 → {OUT_CSV}  ({len(rows)} rows, {n_layers} layers)")
    else:
        print("No rows written.")


if __name__ == "__main__":
    main()
