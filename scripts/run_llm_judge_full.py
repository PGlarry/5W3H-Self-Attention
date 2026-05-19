"""
Stage 6 LLM-as-Judge 全量标注脚本

标注员 A → GPT-4o (laozhang.ai)
标注员 B → Claude-Sonnet-4.5 (laozhang.ai)

支持断点续标（已完成的样本自动跳过）。
用法：
    python run_llm_judge_full.py          # 运行 A 和 B
    python run_llm_judge_full.py A        # 只运行 A
    python run_llm_judge_full.py B        # 只运行 B
"""
import csv
import json
import os
import re
import time
from pathlib import Path

# 加载 API key
_env = Path("d:/pps/backend/.env")
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

import openai

PACK_DIR = Path("d:/pps/paperSELFatten/ist_attention_v1_runs/06_stage6_full")
OUT_DIR  = PACK_DIR / "submitted"
OUT_DIR.mkdir(exist_ok=True)

JUDGE_CONFIGS = {
    "A": {
        "model":     "gpt-4o",
        "client":    openai.OpenAI(
                         api_key=os.environ.get("LAOZHANG_API_KEY", ""),
                         base_url="https://api.laozhang.ai/v1",
                     ),
        "annotator": "A_GPT4o",
    },
    "B": {
        "model":     "claude-sonnet-4-5-20250929",
        "client":    openai.OpenAI(
                         api_key=os.environ.get("LAOZHANG_API_KEY", ""),
                         base_url="https://api.laozhang.ai/v1",
                     ),
        "annotator": "B_ClaudeSonnet45",
    },
}

SYSTEM_PROMPT = """你是一位严格的 NLP 研究评审员，正在评估语言模型在"意图恢复"任务中的表现。

## 实验背景

每个任务都有一个完整的 5W3H 结构 prompt（FULL），以及两种缺失条件版本：
- **PUB-ABSENT**：删除 HOW_TO_DO（公共信息，模型可从先验补偿）
- **PRIV-ABSENT**：删除 WHO 或 HOW_MUCH（私有信息，对象/约束专有）

你的任务是：评估模型在给定（可能缺失部分维度的）prompt 下生成的输出，是否忠实还原了原始意图。

## 评分指标（均为 0.0–1.0，精度 0.1）

**s-ICMw（结构覆盖度）**：输出是否包含任务所需的关键结构单元（有没有/全不全）
- 1.0 = 所有结构单元齐全
- 0.5 = 结构部分存在
- 0.0 = 基本无结构

**f-ICMw（内容保真度）**：输出内容是否忠于原始意图约束（对象、预算、场景）
- 1.0 = 所有关键约束保留
- 0.5 = 部分约束丢失，明显泛化
- 0.0 = 关键约束完全消失

**Ddrift（语义漂移度）**：输出整体偏离原始意图的程度（越高越差）
- 0.0 = 无漂移
- 0.5 = 中度漂移
- 1.0 = 严重漂移

## 重要原则

1. 结构好 ≠ 内容对：s-ICMw 和 f-ICMw 是独立维度
2. 不要因为"输出流畅"就给高分，评的是"忠于原始意图"
3. PRIV-ABSENT 条件下对象/约束漂移是**预期现象**，正确判断漂移程度即可
4. 分数用 0.1 精度（如 0.7、0.8，不要用 0.75）

## PRIV-ABSENT 专项锚点（必读）

PRIV-ABSENT 条件下，WHO 或 HOW_MUCH 被删除，模型无法知晓私有约束，会产生"合理化漂移"。

**f-ICMw 参考区间（PRIV-ABSENT）：**
- 结构完整、内容在无约束下合情合理，仅对象/数字变为泛化默认值 → **0.5–0.7**
- 对象约束彻底消失，内容面向"任意人群"，用途场景也改变 → **0.3–0.4**
- 任务性质根本改变（蜜月规划变亲子游、投资材料变营销文案）→ **0.1–0.2**

**核心原则：若输出结构完整且"移除私有约束后内容仍合理"，f-ICMw 不应低于 0.5。**

**Ddrift 参考区间（PRIV-ABSENT）：**
- 内容略泛化，方向仍正确 → **0.3–0.5**
- 内容明显泛化，对象变通用，但任务类型不变 → **0.5–0.6**
- 任务类型发生变化 → **0.7–1.0**

## 输出格式（严格 JSON，不要有其他内容）

```json
{
  "s_ICMw": 0.0,
  "f_ICMw": 0.0,
  "Ddrift": 0.0,
  "public_private_note": "一句话：此样本删掉的是哪类意图，模型恢复情况如何",
  "main_drift_note": "一句话：最主要的漂移点是什么（如无漂移写'无明显漂移'）"
}
```
"""


def build_user_prompt(sample_id: str) -> str:
    parts = sample_id.split("__")
    condition = parts[1]

    gold   = (PACK_DIR / "gold_context" / f"{sample_id}__gold.txt").read_text(encoding="utf-8")
    prompt = (PACK_DIR / "prompts"      / f"{sample_id}__prompt.txt").read_text(encoding="utf-8")
    output = (PACK_DIR / "outputs"      / f"{sample_id}__output.txt").read_text(encoding="utf-8")

    return f"""## 当前评分任务

**sample_id**：{sample_id}
**condition**：{condition}

---

## Gold Context（任务背景与评分要点）

{gold}

---

## 模型实际收到的 Prompt

```
{prompt}
```

---

## 模型输出（待评分）

```
{output}
```

---

请严格按照系统提示中的 JSON 格式输出评分结果，不要有任何额外内容。
"""


def parse_scores(text: str) -> dict | None:
    m = re.search(r"```json\s*([\s\S]+?)```", text)
    if m:
        text = m.group(1)
    else:
        m = re.search(r"\{[\s\S]+\}", text)
        if m:
            text = m.group(0)
    try:
        data = json.loads(text)
        required = {"s_ICMw", "f_ICMw", "Ddrift", "public_private_note", "main_drift_note"}
        if not required.issubset(data.keys()):
            return None
        for key in ("s_ICMw", "f_ICMw", "Ddrift"):
            val = round(float(data[key]), 1)
            data[key] = max(0.0, min(1.0, val))
        return data
    except (json.JSONDecodeError, ValueError, KeyError):
        return None


def load_completed(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    with open(out_path, encoding="utf-8-sig") as f:
        return {r["sample_id"] for r in csv.DictReader(f) if r.get("s_ICMw", "") != ""}


def judge_sample(sample_id: str, judge_key: str, max_retries: int = 3) -> dict | None:
    cfg = JUDGE_CONFIGS[judge_key]
    user_prompt = build_user_prompt(sample_id)

    for attempt in range(1, max_retries + 1):
        try:
            resp = cfg["client"].chat.completions.create(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content.strip()
            scores = parse_scores(raw)
            if scores:
                return scores
            else:
                print(f"\n    [解析失败 attempt {attempt}] {raw[:150]}")
        except Exception as e:
            print(f"\n    [API错误 attempt {attempt}] {e}")
            time.sleep(5)

    return None


def run_judge(judge_key: str):
    cfg = JUDGE_CONFIGS[judge_key]
    annotator = cfg["annotator"]
    out_path = OUT_DIR / f"full_annotations_{judge_key}_{annotator}.csv"

    fields = ["sample_id", "task_id", "domain", "model", "condition",
              "s_ICMw", "f_ICMw", "Ddrift", "public_private_note", "main_drift_note", "annotator"]

    print(f"\n{'='*60}")
    print(f"  标注员 {judge_key}：{annotator}（{cfg['model']}）")
    print(f"{'='*60}")

    # 读取 manifest
    manifest_path = PACK_DIR / "full_manifest.csv"
    with open(manifest_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # 断点续标：读取已完成样本
    completed = load_completed(out_path)
    if completed:
        print(f"  断点续标：已完成 {len(completed)} 个，跳过")

    # 追加写入模式（首次创建时写表头）
    write_header = not out_path.exists()
    f_out = open(out_path, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f_out, fieldnames=fields)
    if write_header:
        writer.writeheader()

    success, failed = len(completed), 0
    total = len(rows)

    for i, row in enumerate(rows, 1):
        sid = row["sample_id"]
        if sid in completed:
            continue

        print(f"  [{i:03d}/{total}] {sid} ...", end=" ", flush=True)
        scores = judge_sample(sid, judge_key)

        if scores:
            record = {
                "sample_id":           sid,
                "task_id":             row["task_id"],
                "domain":              row["domain"],
                "model":               row["model"],
                "condition":           row["condition"],
                "s_ICMw":              scores["s_ICMw"],
                "f_ICMw":              scores["f_ICMw"],
                "Ddrift":              scores["Ddrift"],
                "public_private_note": scores["public_private_note"],
                "main_drift_note":     scores["main_drift_note"],
                "annotator":           annotator,
            }
            writer.writerow(record)
            f_out.flush()
            success += 1
            print(f"s={scores['s_ICMw']} f={scores['f_ICMw']} d={scores['Ddrift']}")
        else:
            record = {
                "sample_id": sid, "task_id": row["task_id"], "domain": row["domain"],
                "model": row["model"], "condition": row["condition"],
                "s_ICMw": "", "f_ICMw": "", "Ddrift": "",
                "public_private_note": "JUDGE_FAILED", "main_drift_note": "JUDGE_FAILED",
                "annotator": annotator,
            }
            writer.writerow(record)
            f_out.flush()
            failed += 1
            print("FAILED")

        time.sleep(0.8)

    f_out.close()
    print(f"\n  完成：{success}/{total} 成功  {failed} 失败  → {out_path.name}")
    return success, failed


if __name__ == "__main__":
    import sys
    judges = sys.argv[1:] if len(sys.argv) > 1 else ["A", "B"]

    for j in judges:
        if j in JUDGE_CONFIGS:
            run_judge(j)
        else:
            print(f"未知标注员：{j}，可选 A / B")

    print(f"\n{'='*60}")
    print("  Stage 6 全量 LLM Judge 完成")
    print(f"  结果目录：{OUT_DIR}")
    print(f"{'='*60}")
