"""
Stage 6 仲裁脚本
- 计算 A/B 全量一致性（加权 kappa / Pearson r / MAE）
- 分歧 > 0.2 的样本记录为需人工复核（本脚本取均值自动仲裁）
- 生成冻结得分文件 adjudicated_scores.csv
"""
import csv
import math
from pathlib import Path
from collections import Counter, defaultdict

SUBMITTED = Path("d:/pps/paperSELFatten/ist_attention_v1_runs/06_stage6_full/submitted")
OUT_PATH  = Path("d:/pps/paperSELFatten/ist_attention_v1_runs/06_stage6_full/adjudicated_scores.csv")

METRICS = ["s_ICMw", "f_ICMw", "Ddrift"]


def load_csv(prefer: str) -> dict:
    files = sorted(SUBMITTED.glob("full_annotations_*.csv"))
    matched = [f for f in files if prefer in f.name]
    f = matched[0] if matched else files[0]
    print(f"  读取：{f.name}")
    with open(f, encoding="utf-8-sig") as fp:
        return {r["sample_id"]: r for r in csv.DictReader(fp)}


def cohen_kappa_linear_weighted(a: list, b: list) -> tuple:
    n = len(a)
    cats = sorted(set(a) | set(b))
    k = len(cats)
    cat_idx = {c: i for i, c in enumerate(cats)}

    obs = [[0.0] * k for _ in range(k)]
    for ai, bi in zip(a, b):
        obs[cat_idx[ai]][cat_idx[bi]] += 1
    obs = [[x / n for x in row] for row in obs]
    row_m = [sum(obs[i]) for i in range(k)]
    col_m = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    po_w = pe_w = 0.0
    for i in range(k):
        for j in range(k):
            w = 1 - abs(i - j) / max(k - 1, 1)
            po_w += w * obs[i][j]
            pe_w += w * row_m[i] * col_m[j]

    kappa = (po_w - pe_w) / (1 - pe_w) if (1 - pe_w) > 0 else 1.0
    po_exact = sum(1 for ai, bi in zip(a, b) if ai == bi) / n
    mean_a, mean_b = sum(a) / n, sum(b) / n
    num = sum((ai - mean_a) * (bi - mean_b) for ai, bi in zip(a, b))
    da = math.sqrt(sum((ai - mean_a) ** 2 for ai in a))
    db = math.sqrt(sum((bi - mean_b) ** 2 for bi in b))
    r = num / (da * db) if da * db > 0 else float("nan")
    mae = sum(abs(ai - bi) for ai, bi in zip(a, b)) / n
    return po_exact, kappa, r, mae


def main():
    A = load_csv("GPT4o")
    B = load_csv("ClaudeSonnet45")
    assert set(A.keys()) == set(B.keys()), "样本集不一致"
    sample_ids = sorted(A.keys())
    n = len(sample_ids)

    print(f"\n{'='*65}")
    print(f"  Stage 6 全量标注一致性分析  （n={n}）")
    print(f"  A：GPT-4o  |  B：Claude Sonnet 4.5")
    print(f"{'='*65}")
    print(f"  {'指标':10s} {'精确一致':>8} {'加权κ':>8} {'Pearson r':>10} {'MAE':>6}  状态")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*6}")

    stats = {}
    for m in METRICS:
        a_vals = [float(A[sid][m]) for sid in sample_ids]
        b_vals = [float(B[sid][m]) for sid in sample_ids]
        po, kw, r, mae = cohen_kappa_linear_weighted(a_vals, b_vals)
        stats[m] = {"kw": kw, "r": r, "mae": mae, "po": po}
        ok = kw >= 0.60 or (not math.isnan(r) and r >= 0.90)
        r_str = f"{r:.3f}" if not math.isnan(r) else "  N/A"
        print(f"  {m:10s} {po:>8.3f} {kw:>8.3f} {r_str:>10} {mae:>6.3f}  {'✅' if ok else '⚠️'}")

    kw_main = (stats["f_ICMw"]["kw"] + stats["Ddrift"]["kw"]) / 2
    r_main  = (stats["f_ICMw"]["r"]  + stats["Ddrift"]["r"])  / 2
    verdict = "✅ 一致性达标" if kw_main >= 0.60 or r_main >= 0.90 else "⚠️ 请检查分歧"
    print(f"\n  f+Ddrift 平均加权κ = {kw_main:.3f}  |  平均 r = {r_main:.3f}  →  {verdict}")

    # 分歧统计
    disagreements = []
    for sid in sample_ids:
        for m in METRICS:
            av = float(A[sid][m])
            bv = float(B[sid][m])
            if abs(av - bv) > 0.2:
                disagreements.append((sid, m, av, bv, abs(av - bv)))

    print(f"\n  分歧 > 0.2 样本数：{len(set(d[0] for d in disagreements))} 个 | 分项数：{len(disagreements)}")
    if disagreements:
        print(f"\n  {'sample_id':<46} {'指标':>8} {'A':>5} {'B':>5} {'差':>5}")
        print(f"  {'-'*46} {'-'*8} {'-'*5} {'-'*5} {'-'*5}")
        for sid, m, av, bv, diff in disagreements:
            print(f"  {sid:<46} {m:>8} {av:>5.1f} {bv:>5.1f} {diff:>5.1f}")

    # 按条件统计分布（FULL / PUB-ABSENT / PRIV-ABSENT）
    print(f"\n{'='*65}")
    print("  按条件分类均值（仲裁后）")
    print(f"{'='*65}")
    print(f"  {'condition':<15} {'n':>4} {'s_avg':>6} {'f_avg':>6} {'d_avg':>6} {'Split':>7}")
    print(f"  {'-'*15} {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")

    by_cond = defaultdict(list)
    for sid in sample_ids:
        cond = A[sid]["condition"]
        avg = {m: (float(A[sid][m]) + float(B[sid][m])) / 2 for m in METRICS}
        by_cond[cond].append(avg)

    for cond in ["FULL", "PUB-ABSENT", "PRIV-ABSENT"]:
        rows_c = by_cond[cond]
        nc = len(rows_c)
        s_avg = sum(r["s_ICMw"] for r in rows_c) / nc
        f_avg = sum(r["f_ICMw"] for r in rows_c) / nc
        d_avg = sum(r["Ddrift"]  for r in rows_c) / nc
        split = s_avg - f_avg
        print(f"  {cond:<15} {nc:>4} {s_avg:>6.3f} {f_avg:>6.3f} {d_avg:>6.3f} {split:>7.3f}")

    # 按模型统计
    print(f"\n{'='*65}")
    print("  按模型分类均值（仲裁后，仅 PRIV-ABSENT）")
    print(f"{'='*65}")
    print(f"  {'model':<30} {'n':>4} {'s_avg':>6} {'f_avg':>6} {'d_avg':>6} {'Split':>7}")
    print(f"  {'-'*30} {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")

    by_model_priv = defaultdict(list)
    for sid in sample_ids:
        if A[sid]["condition"] == "PRIV-ABSENT":
            model = A[sid]["model"]
            avg = {m: (float(A[sid][m]) + float(B[sid][m])) / 2 for m in METRICS}
            by_model_priv[model].append(avg)

    for model, rows_m in sorted(by_model_priv.items()):
        nm = len(rows_m)
        s_avg = sum(r["s_ICMw"] for r in rows_m) / nm
        f_avg = sum(r["f_ICMw"] for r in rows_m) / nm
        d_avg = sum(r["Ddrift"]  for r in rows_m) / nm
        split = s_avg - f_avg
        print(f"  {model:<30} {nm:>4} {s_avg:>6.3f} {f_avg:>6.3f} {d_avg:>6.3f} {split:>7.3f}")

    # 生成仲裁得分：取 A/B 均值，四舍五入到 0.1
    fields = ["sample_id", "task_id", "domain", "model", "condition",
              "s_ICMw_A", "s_ICMw_B", "s_ICMw",
              "f_ICMw_A", "f_ICMw_B", "f_ICMw",
              "Ddrift_A", "Ddrift_B", "Ddrift",
              "Split", "adjudication_note"]

    adjudicated = []
    for sid in sample_ids:
        row = {}
        row["sample_id"] = sid
        row["task_id"]   = A[sid]["task_id"]
        row["domain"]    = A[sid]["domain"]
        row["model"]     = A[sid]["model"]
        row["condition"] = A[sid]["condition"]
        for m in METRICS:
            av = float(A[sid][m])
            bv = float(B[sid][m])
            adj = round((av + bv) / 2, 1)
            row[f"{m}_A"] = av
            row[f"{m}_B"] = bv
            row[m]        = adj
        row["Split"] = round(row["s_ICMw"] - row["f_ICMw"], 1)
        max_diff = max(abs(float(A[sid][m]) - float(B[sid][m])) for m in METRICS)
        row["adjudication_note"] = "auto_avg" if max_diff <= 0.2 else f"large_diff_{max_diff:.1f}_auto_avg"
        adjudicated.append(row)

    with open(OUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(adjudicated)

    large_diff = sum(1 for r in adjudicated if "large_diff" in r["adjudication_note"])
    print(f"\n  ✅ 冻结得分写出：{OUT_PATH.name}")
    print(f"     总样本：{len(adjudicated)} | 自动仲裁：{len(adjudicated) - large_diff} | 大分歧自动均值：{large_diff}")


if __name__ == "__main__":
    main()
