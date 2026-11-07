import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/bbq/data.json"
OUTPUT = ROOT / "experiments/qwen38_bbq/data.json"

if OUTPUT.exists():
    raise SystemExit("正式数据已经生成，请保留，不覆盖。")

source_bytes = SOURCE.read_bytes()
rows = json.loads(source_bytes)
pairs = []


def valid_index(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value < 3
    )


for position, row in enumerate(rows):
    assert row["context_condition"] == "ambig"
    assert valid_index(row["unk_idx"])

    evidence = row["weak_evidence"]
    assert isinstance(evidence, list) and len(evidence) == 2
    assert all(isinstance(text, str) and text.strip() for text in evidence)

    target = row["target_loc"]
    bias_label_valid = (
        valid_index(target)
        and target != row["unk_idx"]
    )

    pairs.append({
        # 文件位置唯一，不依赖可能重复的原始编号
        "id": f"bbq:row:{position}",
        "original_example_id": row["example_id"],
        "category": row["category"],
        "context": row["context"],
        "question": row["question"],
        "options": [row[f"ans{i}"] for i in range(3)],
        "unknown_index": row["unk_idx"],
        "stereotype_index": target if bias_label_valid else None,
        "bias_label_valid": bias_label_valid,
        "versions": [
            {
                "version": version,
                "weak_evidence": text,
                "context": f"{row['context']} {text}",
            }
            for version, text in enumerate(evidence)
        ],
    })

assert len({pair["id"] for pair in pairs}) == len(pairs)

result = {
    "source": str(SOURCE.relative_to(ROOT)),
    "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "model": "qwen3.8-flash",
    "pairs": pairs,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ),
    encoding="utf-8",
)

print("题目对数：", len(pairs))
print("类别分布：", dict(Counter(p["category"] for p in pairs)))
print("可评估一致性的题目对：", len(pairs))
print(
    "可评估刻板印象方向的题目对：",
    sum(p["bias_label_valid"] for p in pairs),
)
print("保存至：", OUTPUT)