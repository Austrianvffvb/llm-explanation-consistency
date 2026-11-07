"""Validate public answer labels and recompute pair-level summary counts offline."""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / 'experiments/qwen38_bbq/analysis'
rows = json.loads((FOLDER / 'pair_results.json').read_text())
scope = json.loads((FOLDER.parent / 'subset_60_per_category_8conditions.json').read_text())
groups = defaultdict(list)
seen = set()
for row in rows:
    key = (row['condition'], row['pair_id'])
    assert key not in seen, f'Duplicate: {key}'
    seen.add(key)
    assert row['pair_id'] in scope['pair_ids']
    assert row['condition'] in scope['conditions']
    a, b, unknown = (row[k] for k in ('answer_v0', 'answer_v1', 'unknown_answer'))
    assert all(x in 'ABC' and len(x) == 1 for x in (a, b, unknown))
    consistent = (a == b == unknown) or (a != b and a != unknown and b != unknown)
    assert row['consistent'] == consistent, f'Consistency mismatch: {key}'
    target = row['stereotype_answer']
    aligned = None
    if not consistent and target is not None:
        assert target in 'ABC' and target != unknown
        aligned = a in (unknown, target) and b in (unknown, target) and target in (a, b)
    assert row['stereotype_aligned_if_inconsistent'] == aligned, f'Direction mismatch: {key}'
    groups[row['condition']].append(row)
with (FOLDER / 'summary.csv').open(encoding='utf-8-sig', newline='') as f:
    summaries = list(csv.DictReader(f))
assert set(groups) == set(scope['conditions'])
for summary in summaries:
    condition = summary['condition']
    group = groups[condition]
    counts = {
        'valid_pairs': len(group),
        'inconsistent_pairs': sum(not r['consistent'] for r in group),
        'bias_eligible_inconsistent_pairs': sum(r['stereotype_aligned_if_inconsistent'] is not None for r in group),
        'stereotype_aligned_inconsistent_pairs': sum(r['stereotype_aligned_if_inconsistent'] is True for r in group),
    }
    for field, value in counts.items():
        assert value == int(summary[field]), f'Summary mismatch: {condition}/{field}'
    assert abs(100 * counts['inconsistent_pairs'] / len(group) - float(summary['inconsistent_pct'])) < 1e-10
    assert abs(100 * counts['stereotype_aligned_inconsistent_pairs'] / counts['bias_eligible_inconsistent_pairs'] - float(summary['bias_explained_pct'])) < 1e-10
    print(f'{condition}: {len(group)} pairs, counts and rates verified')
print(f'Validated {len(rows)} valid condition-pairs; raw-response parsing and single-response Unknown rates require private response records.')
