"""
build_attention_conditions_v1.py
IST × Self-Attention 实验 — 正式三条件 prompt 生成脚本 v1

包含 Stage 2（micro-clean）+ Stage 3（条件生成）。
只处理 manifest 中 audit_status = "pending" 或 "pass" 的任务。
所有 TYPE_B micro-clean 规则内嵌于此文件。

用法:
    python build_attention_conditions_v1.py \
        --manifest manifest_attention_v1.json \
        --base-dir D:/pps \
        --output-dir ist_attention_v1 \
        --audit-summary audit_output/attention_target_audit_summary.json

输出（每个 task_id 子目录）:
    ist_attention_v1/{task_id}/full.txt
    ist_attention_v1/{task_id}/pub_absent.txt
    ist_attention_v1/{task_id}/priv_absent.txt

日志:
    ist_attention_v1/microclean_change_log.md
    ist_attention_v1/build_report.md
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime


# ─── 维度解析（与审计脚本保持一致）────────────────────────────────────────────

DIMENSION_PATTERNS = {
    "WHAT":      r"任务目标\s*\(What\)\s*:\s*",
    "WHY":       r"执行原因\s*\(Why\)\s*:\s*",
    "WHO":       r"执行角色\s*\(Who\)\s*:\s*",
    "WHEN":      r"时间安排\s*\(When\)\s*:\s*",
    "WHERE":     r"执行场所\s*\(Where\)\s*:\s*",
    "HOW_TO_DO": r"执行方法\s*\(How to do\)\s*:\s*",
    "HOW_MUCH":  r"量化要素\s*\(How much\)\s*:\s*",
    "HOW_FEEL":  r"预期效果\s*\(How feel\)\s*:\s*",
}

DIM_ORDER = ["WHAT", "WHY", "WHO", "WHEN", "WHERE", "HOW_TO_DO", "HOW_MUCH", "HOW_FEEL"]
END_MARKER = "请按照以上内容执行任务"


def parse_dimensions(text: str) -> dict[str, tuple[int, int]]:
    """返回每个维度的 (header_start, content_start, content_end) 三元组。"""
    positions = []
    for dim in DIM_ORDER:
        m = re.search(DIMENSION_PATTERNS[dim], text)
        if m:
            positions.append((dim, m.start(), m.end()))

    positions.sort(key=lambda x: x[1])

    result = {}
    for i, (dim, hdr_start, content_start) in enumerate(positions):
        if i + 1 < len(positions):
            next_hdr_start = positions[i + 1][1]
            content_end = next_hdr_start
        else:
            end_m = re.search(END_MARKER, text)
            content_end = end_m.start() if end_m else len(text)
        result[dim] = (hdr_start, content_start, content_end)

    return result


def remove_dimension(text: str, dim_spans: dict, target_dim: str) -> str:
    """从 prompt 文本中完整删除目标维度（标头 + 内容）。"""
    if target_dim not in dim_spans:
        return text

    hdr_start, _, content_end = dim_spans[target_dim]

    # 往前找换行起始
    line_start = text.rfind('\n', 0, hdr_start) + 1

    # 往后找内容结束后的换行
    line_end = content_end
    while line_end < len(text) and text[line_end] in ('\n', '\r'):
        line_end += 1

    return text[:line_start] + text[line_end:]


# ─── Micro-clean 规则（TYPE_B 任务的文本替换规则）──────────────────────────────
# 格式：{ task_id: { condition: [(old, new, dim_hint), ...] } }
# dim_hint 用于日志记录，不影响替换逻辑（全文替换）

MICROCLEAN_RULES = {

    # ── TR02：PRIV-ABSENT (删WHO后，"一家三口"残留在WHAT+WHY) ──────────────────
    "TR02": {
        "PRIV-ABSENT": [
            (
                "为一家三口规划一次成都五日游方案。",
                "规划一次成都五日游方案。",
                "WHAT"
            ),
            (
                "为了让一家三口在忙碌的生活中放松身心，增进家庭成员之间的感情，同时体验成都丰富的文化与美食。",
                "为了在忙碌的生活中放松身心，体验成都丰富的文化与美食。",
                "WHY"
            ),
        ]
    },

    # ── TR05：PRIV-ABSENT (删WHO后，"老人/退休老人/老年人"扩散全文) ──────────────
    "TR05": {
        "PRIV-ABSENT": [
            # WHAT
            (
                "为退休老人推荐一条张家界三日游行程。",
                "推荐一条张家界三日游行程。",
                "WHAT"
            ),
            # WHY
            (
                "为退休老人提供一次舒适、安全且充满自然美景的旅行体验，帮助他们放松身心，享受晚年生活。",
                "提供一次舒适、安全且充满自然美景的旅行体验，帮助旅行者放松身心，享受自然风光。",
                "WHY"
            ),
            # WHEN
            (
                "这两个季节气候宜人，适合老年人出行；",
                "这两个季节气候宜人，适合出行；",
                "WHEN"
            ),
            # WHERE
            (
                "行程中将安排舒适的住宿和餐饮，确保老人的健康与安全。",
                "行程中将安排舒适的住宿和餐饮，确保旅行者的健康与安全。",
                "WHERE"
            ),
            # HOW_MUCH
            (
                "以确保每位老人得到充分照顾",
                "以确保旅行者得到充分照顾",
                "HOW_MUCH"
            ),
            # HOW_FEEL（整段替换）
            (
                "整体氛围轻松愉悦，注重安全与舒适；行程安排节奏适中，避免过度劳累；情感基调温馨关怀，让老人感受到家人的陪伴与社会的关爱。",
                "整体氛围轻松愉悦，注重安全与舒适；行程安排节奏适中，避免过度劳累；情感基调温馨，让旅行者充分享受旅途中的自然美景。",
                "HOW_FEEL"
            ),
        ]
    },

    # ── TR08：PRIV-ABSENT (删WHO后，蜜月/新婚/浪漫等词扩散全文) ────────────────
    "TR08": {
        "PRIV-ABSENT": [
            # WHAT
            (
                "为蜜月旅行推荐三亚海岛游方案。",
                "推荐三亚海岛游方案。",
                "WHAT"
            ),
            # WHY（整段替换）
            (
                "为新婚夫妇提供一个浪漫、难忘的蜜月旅行体验，同时让他们充分感受三亚的自然美景和文化魅力；通过精心策划的行程，增强彼此的情感联系。",
                "提供一次难忘的旅行体验，充分感受三亚的自然美景和文化魅力；通过精心策划的行程，留下美好回忆。",
                "WHY"
            ),
            # WHEN
            (
                "具体时间可以根据新人的工作安排和个人偏好来定",
                "具体时间可以根据旅行者的工作安排和个人偏好来定",
                "WHEN"
            ),
            # HOW_FEEL（整段替换）
            (
                "整体氛围应保持轻松愉快且充满爱意，让新人能够完全沉浸在二人世界中；行程设计上既要有浪漫元素如海边晚餐、情侣SPA等，也要包含冒险刺激的部分如潜水、冲浪等，以满足不同兴趣需求；此外，还应该注重细节处理，比如房间布置、小礼物赠送等，增加惊喜感。",
                "整体氛围轻松愉快，旅行者能够完全放松身心；行程设计上既要有休闲体验如海边晚餐、SPA等，也包含水上活动如潜水、冲浪等，满足不同兴趣；注重细节，提升旅行体验质量。",
                "HOW_FEEL"
            ),
        ]
    },

    # ── BZ02：PRIV-ABSENT (删WHO后，"制造业/工厂/车间"残留在WHEN+WHERE) ─────────
    "BZ02": {
        "PRIV-ABSENT": [
            # WHEN
            (
                "处于制造业转型升级阶段；",
                "处于企业数字化转型阶段；",
                "WHEN"
            ),
            # WHERE（整段替换）
            (
                "中国某工业城市的制造工厂，覆盖生产车间与物流仓储区域。",
                "中国某城市的企业生产与运营场所。",
                "WHERE"
            ),
        ]
    },

    # ── BZ09：PRIV-ABSENT (删WHO后，"投资者/融资"残留在WHY+WHEN) ────────────────
    "BZ09": {
        "PRIV-ABSENT": [
            # WHY（整段替换）
            (
                "吸引潜在投资者关注项目，展示市场潜力、盈利能力、可行性与创新性。",
                "展示项目的市场潜力、技术可行性与商业价值，为相关决策提供参考。",
                "WHY"
            ),
            # WHEN
            (
                "计划在未来3个月内完成摘要并启动融资。",
                "计划在未来3个月内完成摘要。",
                "WHEN"
            ),
        ]
    },
}


# ─── 条件生成核心函数 ─────────────────────────────────────────────────────────

def apply_microclean(text: str, task_id: str, condition: str) -> tuple[str, list[dict]]:
    """
    对给定文本应用 micro-clean 规则。
    返回 (cleaned_text, changes_log)
    """
    rules = MICROCLEAN_RULES.get(task_id, {}).get(condition, [])
    changes = []

    for old, new, dim_hint in rules:
        if old in text:
            text = text.replace(old, new, 1)
            changes.append({"dimension": dim_hint, "before": old, "after": new})
        else:
            changes.append({"dimension": dim_hint, "before": old, "after": new,
                            "warning": "原文未找到，跳过"})

    return text, changes


def build_conditions(task: dict, base_dir: Path, output_dir: Path,
                     audit_results: dict) -> dict:
    """
    为一个任务生成 full.txt / pub_absent.txt / priv_absent.txt。
    返回结果摘要（含 change log）。
    """
    task_id = task["task_id"]
    prompt_path = base_dir / task["full_prompt_file"]

    result = {
        "task_id": task_id,
        "domain": task["domain"],
        "status": "ok",
        "files_written": [],
        "microclean_changes": {"PRIV-ABSENT": []},
        "errors": []
    }

    if not prompt_path.exists():
        result["status"] = "error"
        result["errors"].append(f"文件不存在: {prompt_path}")
        return result

    raw_text = prompt_path.read_text(encoding="utf-8")
    dim_spans = parse_dimensions(raw_text)

    task_out_dir = output_dir / task_id
    task_out_dir.mkdir(parents=True, exist_ok=True)

    # ── FULL ──────────────────────────────────────────────────────────────
    full_path = task_out_dir / "full.txt"
    full_path.write_text(raw_text, encoding="utf-8")
    result["files_written"].append(str(full_path))

    # ── PUB-ABSENT ────────────────────────────────────────────────────────
    pub_dim = task["pub_absent"]["dimension"]
    pub_text = remove_dimension(raw_text, dim_spans, pub_dim)
    pub_path = task_out_dir / "pub_absent.txt"
    pub_path.write_text(pub_text, encoding="utf-8")
    result["files_written"].append(str(pub_path))
    result["pub_absent_dim"] = pub_dim

    # ── PRIV-ABSENT ───────────────────────────────────────────────────────
    priv_dim = task["priv_absent"]["dimension"]

    # Step 1: 先应用 micro-clean（在删维度之前，因为 micro-clean 修改其他维度）
    cleaned_text, changes = apply_microclean(raw_text, task_id, "PRIV-ABSENT")
    result["microclean_changes"]["PRIV-ABSENT"] = changes

    # Step 2: 在 micro-clean 后的文本上重新解析维度 span，再删目标维度
    cleaned_spans = parse_dimensions(cleaned_text)
    priv_text = remove_dimension(cleaned_text, cleaned_spans, priv_dim)

    priv_path = task_out_dir / "priv_absent.txt"
    priv_path.write_text(priv_text, encoding="utf-8")
    result["files_written"].append(str(priv_path))
    result["priv_absent_dim"] = priv_dim

    return result


# ─── 日志生成 ─────────────────────────────────────────────────────────────────

def generate_microclean_log(all_results: list[dict]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# microclean_change_log.md",
        f"\n生成时间: {now}\n",
        "---\n",
    ]

    for r in all_results:
        changes = r["microclean_changes"].get("PRIV-ABSENT", [])
        real_changes = [c for c in changes if "warning" not in c]
        if not real_changes:
            continue

        lines.append(f"## {r['task_id']} — PRIV-ABSENT micro-clean\n")
        for c in real_changes:
            lines.append(f"**维度**: `{c['dimension']}`\n")
            lines.append(f"- **Before**: {c['before']}")
            lines.append(f"- **After**: {c['after']}\n")

    return "\n".join(lines)


def generate_build_report(all_results: list[dict], manifest: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# build_report.md — 三条件目录生成报告",
        f"\n**生成时间**: {now}",
        f"**Manifest版本**: {manifest.get('manifest_version', 'v1')}",
        "\n---\n",
        "## 任务级结果\n",
        "| task_id | domain | pub_dim | priv_dim | micro-clean | status |",
        "|---------|--------|---------|----------|-------------|--------|",
    ]

    for r in all_results:
        mc_count = len([c for c in r.get("microclean_changes", {}).get("PRIV-ABSENT", [])
                        if "warning" not in c])
        mc_str = f"{mc_count}处" if mc_count else "—"
        lines.append(
            f"| {r['task_id']} | {r['domain']} | {r.get('pub_absent_dim','?')} | "
            f"{r.get('priv_absent_dim','?')} | {mc_str} | {r['status']} |"
        )

    total = len(all_results)
    ok = sum(1 for r in all_results if r["status"] == "ok")
    total_files = sum(len(r.get("files_written", [])) for r in all_results)

    lines += [
        f"\n**合计**: {ok}/{total} 任务成功，生成 {total_files} 个文件",
        "\n---\n",
        "## 冻结说明\n",
        "本目录生成后视为冻结。后续变更需重新运行本脚本并记录原因。",
        "进入 Stage 4（人工 spot-check）和 Stage 5（模型生成）前，请先核查以下内容：",
        "- 每个任务的 priv_absent.txt 是否无目标维度词汇残留",
        "- 每个任务的 pub_absent.txt 是否无 HOW_TO_DO 段落",
        "- micro-clean 的改写是否保持了 prompt 的自然性",
    ]

    return "\n".join(lines)


# ─── 主程序 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IST × Self-Attention 三条件 prompt 生成（Stage 2+3）")
    parser.add_argument("--manifest", default="manifest_attention_v1.json")
    parser.add_argument("--base-dir", default="D:/pps")
    parser.add_argument("--output-dir", default="ist_attention_v1")
    parser.add_argument("--audit-summary", default="audit_output/attention_target_audit_summary.json",
                        help="审计结果文件（用于过滤 TYPE_D 样本，可选）")
    parser.add_argument("--task", default=None, help="只处理特定 task_id")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    # 加载审计结果（可选）
    type_d_tasks = set()
    audit_results = {}
    if Path(args.audit_summary).exists():
        with open(args.audit_summary, encoding="utf-8") as f:
            audit_data = json.load(f)
        for r in audit_data.get("results", []):
            if r["pass_status"] == "TYPE_D":
                type_d_tasks.add(r["task_id"])
            audit_results[f"{r['task_id']}_{r['condition']}"] = r

    tasks = manifest["tasks"]
    if args.task:
        tasks = [t for t in tasks if t["task_id"] == args.task]

    if type_d_tasks:
        excluded = [t["task_id"] for t in tasks if t["task_id"] in type_d_tasks]
        if excluded:
            print(f"跳过 TYPE_D 任务（需人工复核）: {excluded}")
        tasks = [t for t in tasks if t["task_id"] not in type_d_tasks]

    print(f"生成任务数: {len(tasks)}")
    print(f"输出目录: {output_dir.resolve()}\n")

    all_results = []
    for task in tasks:
        print(f"  处理 {task['task_id']} [{task['domain']}] ...", end=" ")
        result = build_conditions(task, base_dir, output_dir, audit_results)
        all_results.append(result)

        mc = len([c for c in result["microclean_changes"].get("PRIV-ABSENT", [])
                  if "warning" not in c])
        if result["status"] == "ok":
            print(f"完成  (micro-clean: {mc}处)")
        else:
            print(f"ERROR: {result['errors']}")

    # 写日志
    log_path = output_dir / "microclean_change_log.md"
    log_path.write_text(generate_microclean_log(all_results), encoding="utf-8")

    report_path = output_dir / "build_report.md"
    report_path.write_text(generate_build_report(all_results, manifest), encoding="utf-8")

    ok_count = sum(1 for r in all_results if r["status"] == "ok")
    print(f"\n完成: {ok_count}/{len(all_results)} 任务")
    print(f"  微清理日志: {log_path}")
    print(f"  构建报告:   {report_path}")


if __name__ == "__main__":
    main()
