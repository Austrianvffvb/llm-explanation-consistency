import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "experiments/qwen38_bbq/analysis"

OUTPUT = ROOT / "runs/statistics"
OUTPUT.mkdir(parents=True, exist_ok=True)

BOOTSTRAPS = 5000
SEED = 42

rows = json.loads(
    (FOLDER / "pair_results.json").read_text(encoding="utf-8")
)

# 每个条件下，每道题只能出现一次
by_condition = {}

for row in rows:
    condition = row["condition"]
    pair_id = row["pair_id"]

    group = by_condition.setdefault(condition, {})

    if pair_id in group:
        raise SystemExit(f"重复记录：{condition} / {pair_id}")

    group[pair_id] = row


def wilson(successes, total):
    """二项比例的95% Wilson置信区间。"""
    if total == 0:
        return None, None

    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total

    center = (p + z * z / (2 * total)) / denominator

    margin = (
        z
        * math.sqrt(
            p * (1 - p) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )

    return 100 * (center - margin), 100 * (center + margin)


def features(row):
    inconsistent = int(not row["consistent"])

    aligned = row["stereotype_aligned_if_inconsistent"]

    # aligned为None时，不属于方向指标的有效分母
    bias_eligible = int(inconsistent and aligned is not None)
    bias_aligned = int(bias_eligible and bool(aligned))

    return [inconsistent, bias_eligible, bias_aligned]


def metrics(array):
    inconsistency = 100 * array[:, 0].mean()

    denominator = array[:, 1].sum()
    bias = (
        100 * array[:, 2].sum() / denominator
        if denominator else None
    )

    return inconsistency, bias


def format_interval(value, low, high):
    if value is None:
        return "NA"
    return f"{value:.1f}% [{low:.1f}, {high:.1f}]"


# 一、八组指标和Wilson区间
group_results = []

print("一、每组指标及95%置信区间\n")

for condition in sorted(by_condition):
    group = list(by_condition[condition].values())
    array = np.array([features(row) for row in group], dtype=float)

    n = len(group)
    inconsistent_n = int(array[:, 0].sum())
    bias_n = int(array[:, 1].sum())
    aligned_n = int(array[:, 2].sum())

    inconsistent_pct = 100 * inconsistent_n / n if n else None
    bias_pct = 100 * aligned_n / bias_n if bias_n else None

    inc_low, inc_high = wilson(inconsistent_n, n)
    bias_low, bias_high = wilson(aligned_n, bias_n)

    result = {
        "condition": condition,
        "valid_pairs": n,
        "inconsistent_pairs": inconsistent_n,
        "inconsistent_pct": inconsistent_pct,
        "inconsistent_ci_low": inc_low,
        "inconsistent_ci_high": inc_high,
        "bias_eligible_inconsistent_pairs": bias_n,
        "stereotype_aligned_pairs": aligned_n,
        "bias_pct": bias_pct,
        "bias_ci_low": bias_low,
        "bias_ci_high": bias_high,
    }
    group_results.append(result)

    print(condition)
    print(
        "  不一致率：",
        format_interval(inconsistent_pct, inc_low, inc_high),
    )
    print(
        f"  刻板印象比例（分母={bias_n}）：",
        format_interval(bias_pct, bias_low, bias_high),
    )


# 二、定义12项比较，统一计算“后者减前者”
comparisons = []

# CoT效应：CoT - No-CoT
for shot in ("zs", "fs"):
    for bias in ("no_debias", "debias"):
        comparisons.append((
            "cot",
            f"{shot}_no_cot_{bias}",
            f"{shot}_cot_{bias}",
        ))

# 去偏指令效应：有去偏 - 无去偏
for shot in ("zs", "fs"):
    for reasoning in ("no_cot", "cot"):
        comparisons.append((
            "debias",
            f"{shot}_{reasoning}_no_debias",
            f"{shot}_{reasoning}_debias",
        ))

# 少样本效应：少样本 - 零样本
for reasoning in ("no_cot", "cot"):
    for bias in ("no_debias", "debias"):
        comparisons.append((
            "few_shot",
            f"zs_{reasoning}_{bias}",
            f"fs_{reasoning}_{bias}",
        ))

rng = np.random.default_rng(SEED)
comparison_results = []

print("\n二、配对差异：后者减前者，单位为百分点\n")

for effect, before, after in comparisons:
    first = by_condition[before]
    second = by_condition[after]

    common_ids = sorted(set(first) & set(second))

    if not common_ids:
        raise SystemExit(f"没有共同有效题目：{before} / {after}")

    for pair_id in common_ids:
        if first[pair_id]["category"] != second[pair_id]["category"]:
            raise SystemExit(f"题目类别不一致：{pair_id}")

    a = np.array(
        [features(first[pair_id]) for pair_id in common_ids],
        dtype=float,
    )
    b = np.array(
        [features(second[pair_id]) for pair_id in common_ids],
        dtype=float,
    )

    before_inc, before_bias = metrics(a)
    after_inc, after_bias = metrics(b)

    # 每个类别内有放回抽样；同一个索引用于两个条件
    categories = [
        first[pair_id]["category"] for pair_id in common_ids
    ]

    sampled_indices = []

    for category in sorted(set(categories)):
        indices = np.array([
            index
            for index, value in enumerate(categories)
            if value == category
        ])

        sampled_indices.append(
            rng.choice(
                indices,
                size=(BOOTSTRAPS, len(indices)),
                replace=True,
            )
        )

    draws = np.concatenate(sampled_indices, axis=1)

    a_sample = a[draws].sum(axis=1)
    b_sample = b[draws].sum(axis=1)

    inc_differences = (
        100
        * (b_sample[:, 0] - a_sample[:, 0])
        / len(common_ids)
    )

    inc_low, inc_high = np.quantile(
        inc_differences, [0.025, 0.975]
    )

    # 方向比例的分母随条件和重采样变化，逐次重新计算
    valid_draws = (
        (a_sample[:, 1] > 0)
        & (b_sample[:, 1] > 0)
    )

    bias_difference = None
    bias_low = None
    bias_high = None

    if before_bias is not None and after_bias is not None:
        bias_difference = after_bias - before_bias

        if valid_draws.any():
            bias_differences = 100 * (
                b_sample[valid_draws, 2]
                / b_sample[valid_draws, 1]
                - a_sample[valid_draws, 2]
                / a_sample[valid_draws, 1]
            )

            bias_low, bias_high = np.quantile(
                bias_differences, [0.025, 0.975]
            )

    result = {
        "effect": effect,
        "before": before,
        "after": after,
        "common_pairs": len(common_ids),
        "before_inconsistent_pct": before_inc,
        "after_inconsistent_pct": after_inc,
        "inconsistent_difference_pp": after_inc - before_inc,
        "inconsistent_diff_ci_low": float(inc_low),
        "inconsistent_diff_ci_high": float(inc_high),
        "before_bias_pct": before_bias,
        "after_bias_pct": after_bias,
        "bias_difference_pp": bias_difference,
        "bias_diff_ci_low": (
            float(bias_low) if bias_low is not None else None
        ),
        "bias_diff_ci_high": (
            float(bias_high) if bias_high is not None else None
        ),
        "bias_valid_bootstraps": int(valid_draws.sum()),
    }
    comparison_results.append(result)

    print(f"{before} → {after}")
    print(f"  共同有效题目：{len(common_ids)}")
    print(
        f"  不一致率差：{after_inc - before_inc:+.2f} "
        f"[{inc_low:+.2f}, {inc_high:+.2f}]"
    )

    if bias_low is not None:
        print(
            f"  刻板印象比例差：{bias_difference:+.2f} "
            f"[{bias_low:+.2f}, {bias_high:+.2f}]"
        )
    else:
        print("  刻板印象比例差：无法计算有效区间")


def save_csv(name, data):
    with (OUTPUT / name).open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(data[0]),
        )
        writer.writeheader()
        writer.writerows(data)


save_csv("statistics_groups.csv", group_results)
save_csv("statistics_comparisons.csv", comparison_results)

(OUTPUT / "statistics.json").write_text(
    json.dumps({
        "seed": SEED,
        "bootstrap_repetitions": BOOTSTRAPS,
        "groups": group_results,
        "comparisons": comparison_results,
        "notes": [
            "Group intervals use the Wilson method.",
            "Comparisons use common valid pairs only.",
            "Bootstrap resamples pairs within categories.",
            "Differences are after minus before, in percentage points.",
            "Intervals are pointwise; no multiple-comparison adjustment.",
            "Intervals do not account for shared question templates.",
        ],
    }, ensure_ascii=False, indent=2, allow_nan=False),
    encoding="utf-8",
)

print("\n统计完成，结果保存在：", OUTPUT)