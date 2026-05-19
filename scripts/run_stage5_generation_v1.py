"""
Stage 5 Generation Script v1
IST × Self-Attention 实验 — 正式批量生成

模型配置：
  closed_global_strong    → GPT-4o (OpenAI API)
  commercial_cn_deepseek  → DeepSeek-chat (DeepSeek API)
  open_weight_qwen_main   → qwen3:latest (ollama 本地)

用法：
  python run_stage5_generation_v1.py --output-dir ist_attention_v1_runs
  python run_stage5_generation_v1.py --only-model open_weight_qwen_main
  python run_stage5_generation_v1.py --only-domain travel --only-model commercial_cn_deepseek
  python run_stage5_generation_v1.py --retry-failed

环境变量（API 模型必需）：
  OPENAI_API_KEY       → GPT-4o
  DEEPSEEK_API_KEY     → DeepSeek-chat
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# 自动加载 backend/.env（laozhang / DeepSeek key）
_env_file = Path(__file__).parent.parent / "backend" / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# ── 冻结配置 ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一个谨慎且服从指令的助手。"
    "请直接用中文完成用户请求，不要追问，不要解释提示信息是否缺失，"
    "也不要讨论提示词结构；"
    "在信息不足时，基于现有信息给出最合理、最通用的完成版本。"
    "不要输出思考过程、推理过程或任何类似 <think> 的中间内容，只输出最终答案。"
)

GEN_PARAMS = {
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 1536,  # 可由 --max-tokens 覆盖
}

MODEL_CONFIGS = {
    "closed_global_strong": {
        "full_name": "gpt-4o (laozhang.ai)",
        "type": "laozhang",
        "model_id": "gpt-4o",
    },
    "commercial_cn_deepseek": {
        "full_name": "DeepSeek-chat",
        "type": "deepseek",
        "model_id": "deepseek-chat",
    },
    "open_weight_qwen_main": {
        "full_name": "qwen3:latest (ollama)",
        "type": "ollama",
        "model_id": "qwen3:latest",
    },
}

CONDITIONS = ["full", "pub_absent", "priv_absent"]

CONDITION_LABEL = {
    "full": "FULL",
    "pub_absent": "PUB-ABSENT",
    "priv_absent": "PRIV-ABSENT",
}

DOMAIN_ORDER = ["travel", "business", "technical"]

MAX_RETRIES = 2

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def strip_think_block(text: str) -> tuple[str, bool]:
    """
    剥离 Qwen3 的 <think>...</think> 块。
    返回 (剥离后文本, 是否含非空思维链)。
    /no_think 模式下 think 块内容为空，可安全剥除。
    """
    pattern = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)
    has_real_thinking = False
    def replacer(m):
        nonlocal has_real_thinking
        content = m.group(1).strip()
        if content:
            has_real_thinking = True
        return ""
    cleaned = pattern.sub(replacer, text).strip()
    return cleaned, has_real_thinking


def contains_thinking(text: str) -> bool:
    _, has_real = strip_think_block(text)
    return has_real


def is_truncated(text: str) -> bool:
    """
    判断输出是否真实截断。只有在以下情况才报警：
    - 输出为空
    - 最后一行明显是未完成的句子（逗号、顿号结尾，或行末有未闭合括号/引号）
    - 输出过短（< 100 字）
    """
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) < 100:
        return True
    last_line = stripped.split("\n")[-1].strip()
    if not last_line:
        last_line = stripped
    last_char = last_line[-1] if last_line else ""
    # 逗号/顿号/分号结尾 = 明显截断
    if last_char in {"，", "、", "；", ",", ";", "：", ":"}:
        return True
    return False


def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_prompt(task_dir: Path, condition: str) -> str:
    fname = f"{condition}.txt"
    fpath = task_dir / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Prompt file not found: {fpath}")
    return fpath.read_text(encoding="utf-8").strip()


# ── 模型调用 ──────────────────────────────────────────────────────────────────

def call_openai(model_id: str, system: str, user: str, api_key: str, base_url: str = None) -> str:
    try:
        import openai
    except ImportError:
        sys.exit("请先安装 openai: pip install openai")

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    client = openai.OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=GEN_PARAMS["temperature"],
        top_p=GEN_PARAMS["top_p"],
        max_tokens=GEN_PARAMS["max_tokens"],
        n=1,
    )
    return resp.choices[0].message.content.strip()


def call_ollama(model_id: str, system: str, user: str, base_url: str = "http://localhost:11434") -> str:
    # /no_think 是 Qwen3 关闭 thinking 的官方方式，产生空 <think></think> 块，后处理剥除
    user_with_nothink = user + "\n/no_think"
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_with_nothink},
        ],
        "stream": False,
        "options": {
            "temperature": GEN_PARAMS["temperature"],
            "top_p": GEN_PARAMS["top_p"],
            "num_predict": GEN_PARAMS["max_tokens"],
        },
    }
    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    raw = data["message"]["content"].strip()
    # 剥离空 think 块；若 think 块有内容会触发 contains_thinking → needs_review
    cleaned, _ = strip_think_block(raw)
    return cleaned


def call_model(model_label: str, system: str, user: str) -> str:
    cfg = MODEL_CONFIGS[model_label]
    mtype = cfg["type"]

    if mtype == "laozhang":
        api_key = os.environ.get("LAOZHANG_API_KEY", "")
        if not api_key:
            raise EnvironmentError("LAOZHANG_API_KEY 未设置（检查 backend/.env）")
        return call_openai(cfg["model_id"], system, user, api_key,
                           base_url="https://api.laozhang.ai/v1")

    elif mtype == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY 未设置")
        return call_openai(cfg["model_id"], system, user, api_key)

    elif mtype == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise EnvironmentError("DEEPSEEK_API_KEY 未设置")
        return call_openai(cfg["model_id"], system, user, api_key, base_url="https://api.deepseek.com")

    elif mtype == "ollama":
        return call_ollama(cfg["model_id"], system, user)

    else:
        raise ValueError(f"Unknown model type: {mtype}")


# ── 输出路径 ──────────────────────────────────────────────────────────────────

def output_path(output_dir: Path, model_label: str, domain: str, task_id: str, condition: str) -> Path:
    cond_label = CONDITION_LABEL[condition]
    fname = f"{task_id}__{cond_label}__{model_label}.txt"
    return output_dir / "02_generations" / model_label / domain / fname


def sample_id(task_id: str, condition: str, model_label: str) -> str:
    return f"{task_id}__{CONDITION_LABEL[condition]}__{model_label}"


# ── 日志 ──────────────────────────────────────────────────────────────────────

def append_log(log_path: Path, record: dict):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_completed(log_path: Path) -> set:
    """从 generations_master.jsonl 读取已成功完成的 sample_id 集合"""
    completed = set()
    if not log_path.exists():
        return completed
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("run_status") == "success":
                    completed.add(rec["sample_id"])
            except json.JSONDecodeError:
                pass
    return completed


def load_failed(log_path: Path) -> list:
    """从 generations_master.jsonl 读取最终失败的 sample_id（无对应 success）"""
    all_records = {}
    if not log_path.exists():
        return []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sid = rec["sample_id"]
                if sid not in all_records or rec["run_status"] == "success":
                    all_records[sid] = rec
            except (json.JSONDecodeError, KeyError):
                pass
    return [r for r in all_records.values() if r.get("run_status") != "success"]


# ── 单样本执行 ────────────────────────────────────────────────────────────────

def run_sample(task: dict, condition: str, model_label: str,
               input_dir: Path, output_dir: Path, log_path: Path,
               failed_path: Path, system: str, dry_run: bool = False) -> str:
    """
    返回 'success' / 'skip' / 'failed'
    """
    tid = task["task_id"]
    domain = task["domain"]
    sid = sample_id(tid, condition, model_label)

    # 读 prompt
    try:
        task_dir = input_dir / tid
        prompt_text = load_prompt(task_dir, condition)
    except FileNotFoundError as e:
        print(f"  [SKIP] {sid} — prompt 文件缺失: {e}")
        return "skip"

    out_path = output_path(output_dir, model_label, domain, tid, condition)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_hash = sha256(prompt_text)
    system_hash = sha256(system)
    cfg = MODEL_CONFIGS[model_label]

    base_record = {
        "sample_id": sid,
        "task_id": tid,
        "domain": domain,
        "condition": CONDITION_LABEL[condition],
        "model_label": model_label,
        "model_full_name": cfg["full_name"],
        "model_revision_or_date": now_iso()[:10],
        "input_path": str(task_dir / f"{condition}.txt"),
        "output_path": str(out_path),
        "system_prompt_hash": system_hash,
        "temperature": GEN_PARAMS["temperature"],
        "top_p": GEN_PARAMS["top_p"],
        "max_new_tokens": GEN_PARAMS["max_tokens"],
        "prompt_hash": prompt_hash,
        "timestamp": now_iso(),
        "notes": "",
    }

    if dry_run:
        print(f"  [DRY-RUN] {sid}")
        return "skip"

    # 重试循环
    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            output_text = call_model(model_label, system, prompt_text)
        except Exception as e:
            last_error = str(e)
            if attempt <= MAX_RETRIES:
                print(f"  [RETRY {attempt}] {sid} — {e}")
                time.sleep(3)
                continue
            break
        else:
            last_error = None
            break

    if last_error:
        rec = {**base_record, "run_status": "failed", "notes": last_error}
        append_log(log_path, rec)
        append_log(failed_path, {
            "sample_id": sid, "model_label": model_label,
            "error_type": "api_error", "error_message": last_error,
            "timestamp": now_iso(), "retry_status": f"failed_after_{MAX_RETRIES}_retries",
        })
        print(f"  [FAILED] {sid} — {last_error}")
        return "failed"

    # 内容检查（不自动重跑，记录需要人工处理的情况）
    issues = []
    if contains_thinking(output_text):
        issues.append("THINKING_DETECTED")
    if not output_text:
        issues.append("EMPTY_OUTPUT")
    if is_truncated(output_text):
        issues.append("POSSIBLE_TRUNCATION")

    if issues:
        flag_str = "|".join(issues)
        rec = {**base_record, "run_status": "needs_review", "notes": flag_str}
        append_log(log_path, rec)
        append_log(failed_path, {
            "sample_id": sid, "model_label": model_label,
            "error_type": "content_issue", "error_message": flag_str,
            "timestamp": now_iso(), "retry_status": "needs_human_review",
        })
        # 仍然保存输出，但标记为需要审查
        out_path.write_text(output_text, encoding="utf-8")
        print(f"  [REVIEW] {sid} — {flag_str}")
        return "failed"

    # 正常写出
    out_path.write_text(output_text, encoding="utf-8")
    rec = {**base_record, "run_status": "success", "notes": ""}
    append_log(log_path, rec)
    print(f"  [OK] {sid}")
    return "success"


# ── 主流程 ────────────────────────────────────────────────────────────────────

def build_task_list(manifest: dict, only_domain: str = None) -> list:
    tasks = manifest["tasks"]
    if only_domain:
        tasks = [t for t in tasks if t["domain"] == only_domain]
    # 按 domain 顺序排序
    order = {d: i for i, d in enumerate(DOMAIN_ORDER)}
    return sorted(tasks, key=lambda t: order.get(t["domain"], 99))


def run_model(model_label: str, tasks: list, input_dir: Path, output_dir: Path,
              log_path: Path, failed_path: Path, system: str,
              completed: set, dry_run: bool = False) -> dict:
    counts = {"success": 0, "skip": 0, "failed": 0}
    total = len(tasks) * len(CONDITIONS)
    done = 0

    print(f"\n{'='*60}")
    print(f"  模型：{model_label}  ({MODEL_CONFIGS[model_label]['full_name']})")
    print(f"  样本总数：{total}")
    print(f"{'='*60}")

    for task in tasks:
        for condition in CONDITIONS:
            sid = sample_id(task["task_id"], condition, model_label)
            done += 1
            print(f"[{done}/{total}] {sid}", end="")

            if sid in completed:
                print(f"  → 已完成，跳过")
                counts["skip"] += 1
                continue
            print()

            result = run_sample(
                task, condition, model_label,
                input_dir, output_dir, log_path, failed_path,
                system, dry_run=dry_run,
            )
            counts[result] = counts.get(result, 0) + 1

            # API 调用间隔（避免速率限制）
            if not dry_run and MODEL_CONFIGS[model_label]["type"] != "ollama":
                time.sleep(0.5)

    return counts


def write_run_summary(output_dir: Path, summary: dict):
    import csv
    path = output_dir / "03_logs" / "run_summary.csv"
    fieldnames = ["model_label", "total_expected", "total_success",
                  "total_failed", "retried", "final_pass", "final_fail"]
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for model_label, counts in summary.items():
            total = counts.get("success", 0) + counts.get("failed", 0) + counts.get("skip", 0)
            writer.writerow({
                "model_label": model_label,
                "total_expected": len(CONDITIONS) * counts.get("_task_count", 0),
                "total_success": counts.get("success", 0),
                "total_failed": counts.get("failed", 0),
                "retried": 0,
                "final_pass": counts.get("success", 0),
                "final_fail": counts.get("failed", 0),
            })


def main():
    parser = argparse.ArgumentParser(description="Stage 5 Generation Script v1")
    parser.add_argument("--input-dir", default="paperSELFatten/ist_attention_v1",
                        help="包含 12 个任务子目录的根目录")
    parser.add_argument("--manifest", default="paperSELFatten/manifest_attention_v1.json",
                        help="manifest_attention_v1.json 路径")
    parser.add_argument("--output-dir", default="paperSELFatten/ist_attention_v1_runs",
                        help="输出根目录")
    parser.add_argument("--system-prompt", default=None,
                        help="system_prompt.txt 路径（不指定则用内置冻结版）")
    parser.add_argument("--models", nargs="+",
                        choices=list(MODEL_CONFIGS.keys()),
                        default=["open_weight_qwen_main", "commercial_cn_deepseek", "closed_global_strong"],
                        help="要运行的模型（按顺序）")
    parser.add_argument("--only-domain", choices=["travel", "business", "technical"],
                        default=None, help="只处理指定域")
    parser.add_argument("--only-model", choices=list(MODEL_CONFIGS.keys()),
                        default=None, help="只运行指定模型")
    parser.add_argument("--retry-failed", action="store_true",
                        help="只重跑 failed_cases.jsonl 中的失败样本")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="覆盖默认 max_tokens（默认 1536），用于重跑截断样本")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将运行的 sample_id，不实际调用模型")
    args = parser.parse_args()

    # 路径解析（支持相对路径从 d:\pps 出发）
    base = Path("d:/pps")
    input_dir = base / args.input_dir if not Path(args.input_dir).is_absolute() else Path(args.input_dir)
    manifest_path = base / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest)
    output_dir = base / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    # 覆盖 max_tokens（用于重跑截断样本）
    if args.max_tokens:
        GEN_PARAMS["max_tokens"] = args.max_tokens
        print(f"[INFO] max_tokens 覆盖为 {args.max_tokens}")

    # 加载 manifest
    manifest = load_manifest(manifest_path)

    # System prompt
    if args.system_prompt:
        sp_path = Path(args.system_prompt)
        system = sp_path.read_text(encoding="utf-8").strip()
    else:
        system = SYSTEM_PROMPT

    # 建立目录结构
    for sub in ["01_inputs", "03_logs", "04_qc"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    # 保存 system_prompt.txt
    sp_file = output_dir / "01_inputs" / "system_prompt.txt"
    sp_file.write_text(system, encoding="utf-8")

    # 日志路径
    log_path = output_dir / "03_logs" / "generations_master.jsonl"
    failed_path = output_dir / "03_logs" / "failed_cases.jsonl"

    # 已完成集合（支持断点续跑）
    completed = load_completed(log_path)
    if completed:
        print(f"[INFO] 检测到 {len(completed)} 个已完成样本，将跳过")

    # 模型列表
    models_to_run = [args.only_model] if args.only_model else args.models

    # 任务列表
    tasks = build_task_list(manifest, args.only_domain)
    print(f"[INFO] 任务数：{len(tasks)}  条件数：{len(CONDITIONS)}  模型数：{len(models_to_run)}")
    print(f"[INFO] 理论总输出：{len(tasks) * len(CONDITIONS) * len(models_to_run)}")

    # retry-failed 模式：只重跑失败样本
    if args.retry_failed:
        failed_records = load_failed(log_path)
        if not failed_records:
            print("[INFO] 没有找到需要重跑的失败样本")
            return
        print(f"[INFO] 重跑模式：{len(failed_records)} 个失败样本")
        # 从 completed 中移除这些 sample_id，让它们可以重跑
        for rec in failed_records:
            completed.discard(rec["sample_id"])

    # 按模型顺序执行
    all_summary = {}
    for model_label in models_to_run:
        counts = run_model(
            model_label, tasks, input_dir, output_dir,
            log_path, failed_path, system, completed,
            dry_run=args.dry_run,
        )
        counts["_task_count"] = len(tasks)
        all_summary[model_label] = counts

        # 打印本模型汇总
        print(f"\n  ── {model_label} 完成 ──")
        print(f"     成功：{counts.get('success', 0)}")
        print(f"     跳过：{counts.get('skip', 0)}")
        print(f"     失败/需审查：{counts.get('failed', 0)}")

    # 写 run_summary.csv
    if not args.dry_run:
        write_run_summary(output_dir, all_summary)

    # 最终汇总
    total_success = sum(v.get("success", 0) for v in all_summary.values())
    total_failed = sum(v.get("failed", 0) for v in all_summary.values())
    print(f"\n{'='*60}")
    print(f"  Stage 5 执行完毕")
    print(f"  总成功：{total_success}")
    print(f"  总失败/需审查：{total_failed}")
    if total_failed > 0:
        print(f"  → 请检查 {failed_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
