"""
Stage 7 步骤2（通用版）：自注意力提取与机制指标计算
用法：python run_stage7_attention_generic.py <model_path> <model_tag>

示例：
  python run_stage7_attention_generic.py D:/models/Ministral-8B-Instruct-2410 ministral8b
  python run_stage7_attention_generic.py D:/models/gemma-3-4b-it gemma3_4b
"""
import sys, csv, json, math, gc
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if len(sys.argv) < 3:
    print("用法: python run_stage7_attention_generic.py <model_path> <model_tag>")
    sys.exit(1)

MODEL_PATH = sys.argv[1]
MODEL_TAG  = sys.argv[2]

PACK_DIR   = Path("d:/pps/paperSELFatten/ist_attention_v1_runs/06_stage6_full")
STAGE_DIR  = Path(f"d:/pps/paperSELFatten/ist_attention_v1_runs/07_stage7_{MODEL_TAG}")
SPAN_LOG   = STAGE_DIR / "span_alignment_log.jsonl"
OUT_CSV    = STAGE_DIR / f"stage7_attention_metrics_{MODEL_TAG}.csv"

PRIVATE_DIMS     = {"WHO", "HOW_MUCH"}
PUBLIC_DIMS      = {"HOW_TO_DO"}
ALL_CONTENT_DIMS = {"WHAT","WHY","WHO","WHEN","WHERE","HOW_TO_DO","HOW_MUCH","HOW_FEEL"}


def load_span_records() -> dict:
    records = {}
    with open(SPAN_LOG, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records[r["sample_id"]] = r
    return records


def attention_to_span(attentions, ts: int, te: int, last_pos: int) -> np.ndarray:
    if ts >= te:
        return np.zeros(len(attentions))
    per_layer = []
    for layer_attn in attentions:
        # layer_attn: (batch=1, heads, seq, seq)
        a = layer_attn[0, :, last_pos, ts:te]
        per_layer.append(float(a.mean().cpu()))
    return np.array(per_layer)


def compute_metrics(span_record: dict, attentions, prompt_len: int) -> dict:
    last_pos = prompt_len - 1
    dims = span_record["dims"]

    dim_attn = {}
    for dim in ALL_CONTENT_DIMS:
        if dim not in dims:
            continue
        ts, te = dims[dim]["token"]
        if te <= ts:
            continue
        dim_attn[dim] = attention_to_span(attentions, ts, te, last_pos)

    if not dim_attn:
        return {}

    AS_raw  = {d: float(v.mean()) for d, v in dim_attn.items()}
    total   = sum(AS_raw.values()) or 1.0
    AS_norm = {d: v / total for d, v in AS_raw.items()}

    AS_private = float(np.mean([AS_norm[d] for d in PRIVATE_DIMS if d in AS_norm])) \
                 if any(d in AS_norm for d in PRIVATE_DIMS) else 0.0
    AS_public  = float(np.mean([AS_norm[d] for d in PUBLIC_DIMS  if d in AS_norm])) \
                 if any(d in AS_norm for d in PUBLIC_DIMS)  else 0.0
    AS_ratio   = AS_private / AS_public if AS_public > 1e-9 else float("nan")

    probs  = [v for v in AS_norm.values() if v > 1e-12]
    AE     = -sum(p * math.log2(p) for p in probs)
    AE_max = math.log2(len(dim_attn)) if len(dim_attn) > 1 else 1.0
    AE_norm_val = AE / AE_max

    priv_per_layer = [dim_attn[d] for d in PRIVATE_DIMS if d in dim_attn]
    if priv_per_layer:
        combined = np.stack(priv_per_layer).mean(axis=0)
        mean_v = combined.mean()
        std_v  = combined.std()
        CV = std_v / mean_v if mean_v > 1e-9 else 1.0
        LP = max(0.0, 1.0 - CV)
    else:
        LP = float("nan")

    result = {
        "AS_private": round(float(AS_private), 4),
        "AS_public":  round(float(AS_public),  4),
        "AS_ratio":   round(float(AS_ratio),   4) if not math.isnan(AS_ratio) else "",
        "AE":         round(float(AE),         4),
        "AE_norm":    round(float(AE_norm_val),4),
        "LP_private": round(float(LP),         4) if not math.isnan(LP) else "",
    }
    for d in ALL_CONTENT_DIMS:
        result[f"AS_{d}"] = round(AS_norm.get(d, 0.0), 4)
    return result


def main():
    print(f"模型：{MODEL_PATH}  标签：{MODEL_TAG}")
    print("加载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("加载模型（4-bit + eager attention）...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM 占用：{vram:.2f} GB")

    span_records = load_span_records()
    sample_ids   = sorted(span_records.keys())
    print(f"待处理样本：{len(sample_ids)}")

    with open(PACK_DIR / "full_manifest.csv", encoding="utf-8-sig") as f:
        manifest = {r["sample_id"]: r for r in csv.DictReader(f)}

    fields = [
        "sample_id","task_id","domain","condition","total_tokens","model_tag",
        "AS_private","AS_public","AS_ratio","AE","AE_norm","LP_private",
        "AS_WHAT","AS_WHY","AS_WHO","AS_WHEN","AS_WHERE",
        "AS_HOW_TO_DO","AS_HOW_MUCH","AS_HOW_FEEL",
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()

        for i, sid in enumerate(sample_ids, 1):
            rec  = span_records[sid]
            meta = manifest.get(sid, {})
            n    = len(sample_ids)
            print(f"  [{i:02d}/{n}] {sid} (tokens={rec['total_tokens']}) ...", end=" ", flush=True)

            prompt_path = PACK_DIR / "prompts" / f"{sid}__prompt.txt"
            prompt_text = prompt_path.read_text(encoding="utf-8")

            inputs     = tok(prompt_text, return_tensors="pt").to("cuda")
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)

            metrics = compute_metrics(rec, outputs.attentions, prompt_len)

            row = {
                "sample_id":    sid,
                "task_id":      meta.get("task_id",""),
                "domain":       meta.get("domain",""),
                "condition":    rec["condition"],
                "total_tokens": rec["total_tokens"],
                "model_tag":    MODEL_TAG,
            }
            row.update(metrics)
            for f_name in fields:
                if f_name not in row:
                    row[f_name] = ""

            writer.writerow(row)
            fout.flush()

            print(f"AS_priv={metrics.get('AS_private','?')} "
                  f"AS_pub={metrics.get('AS_public','?')} "
                  f"AE_n={metrics.get('AE_norm','?')} "
                  f"LP={metrics.get('LP_private','?')}")

            del outputs
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\n✅ 完成 → {OUT_CSV}")


if __name__ == "__main__":
    main()
