import hashlib
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI, APIConnectionError, APIStatusError


ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "experiments/qwen38_bbq"
RESULTS = FOLDER / "results"

WORKERS = 5
MAX_ATTEMPTS = 3

manifest_bytes = (FOLDER / "requests.json").read_bytes()
dataset = json.loads(manifest_bytes)
records = dataset["requests"]

config = json.loads(
    (FOLDER / "run_config.json").read_text(encoding="utf-8")
)

subset = json.loads(
    (FOLDER / "subset_60_per_category.json").read_text(
        encoding="utf-8"
    )
)

manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

if config["manifest_sha256"] != manifest_hash:
    raise SystemExit("原始提示清单已改变，停止运行。")

if subset["manifest_sha256"] != manifest_hash:
    raise SystemExit("抽样清单与原始提示清单不一致。")

if (
    config["model"] != dataset["model"]
    or config["base_url"] != dataset["base_url"]
):
    raise SystemExit("模型或接口配置不一致。")

selected_pairs = set(subset["pair_ids"])

# 全部八组实验条件
selected_conditions = {
    f"{shot}_{reasoning}_{bias}"
    for shot in ("zs", "fs")
    for reasoning in ("no_cot", "cot")
    for bias in ("no_debias", "debias")
}

if len(selected_pairs) != 540:
    raise SystemExit("抽样题目不是540对，请检查。")

# 保留原始 index，继续对应已有结果文件
selected = [
    (index, record)
    for index, record in enumerate(records)
    if record["pair_id"] in selected_pairs
    and record["condition"] in selected_conditions
]

if len(selected) != 8640:
    raise SystemExit("本轮请求数量不是8640，请检查。")

if len({record["request_id"] for _, record in selected}) != 8640:
    raise SystemExit("请求编号存在重复，请检查。")

# 检查九类各60对，以及每对题目的条件和版本
pair_categories = {}
pair_cells = {}

for _, record in selected:
    pair_id = record["pair_id"]
    category = record["category"]

    if (
        pair_id in pair_categories
        and pair_categories[pair_id] != category
    ):
        raise SystemExit(f"同一题目的类别不一致：{pair_id}")

    pair_categories[pair_id] = category
    pair_cells.setdefault(pair_id, []).append(
        (record["condition"], record["version"])
    )

category_counts = {}
for category in pair_categories.values():
    category_counts[category] = category_counts.get(category, 0) + 1

if len(category_counts) != 9 or any(
    count != 60 for count in category_counts.values()
):
    raise SystemExit("类别分布不是九类各60对，请检查。")

expected_cells = {
    (condition, version)
    for condition in selected_conditions
    for version in (0, 1)
}

for pair_id, cells in pair_cells.items():
    if len(cells) != 16 or set(cells) != expected_cells:
        raise SystemExit(f"题目的条件或证据版本不完整：{pair_id}")

# 保存八组实验范围，不覆盖原来的两组范围文件
scope_path = FOLDER / "subset_60_per_category_8conditions.json"
scope = {
    **subset,
    "conditions": sorted(selected_conditions),
}

if scope_path.exists():
    previous_scope = json.loads(
        scope_path.read_text(encoding="utf-8")
    )
    if previous_scope != scope:
        raise SystemExit("已有八组实验范围与当前设置不一致。")
else:
    scope_path.write_text(
        json.dumps(scope, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

api_key = os.environ.get("TOKENDANCE_API_KEY")
if not api_key:
    raise SystemExit("请先配置 TOKENDANCE_API_KEY。")

RESULTS.mkdir(parents=True, exist_ok=True)

tasks = queue.Queue()
stop = threading.Event()
lock = threading.Lock()

stats = {
    "completed": 0,
    "needs_check": 0,
    "failed": 0,
}


def save(path, item):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(item, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def body_for(record):
    return {
        "model": config["model"],
        "messages": [
            {"role": "user", "content": record["prompt"]}
        ],
        "max_tokens": config["max_tokens"],
        "stream": False,
    }


def normal_response(response):
    choices = response.get("choices", [])
    if not choices:
        return False

    content = choices[0].get("message", {}).get("content")

    return (
        choices[0].get("finish_reason") == "stop"
        and isinstance(content, str)
        and bool(content.strip())
    )


# 发起调用前，检查全部目标请求和已有结果
reused = 0
existing_needs_check = 0
exhausted = 0
missing = 0
retryable_existing = 0

for index, record in selected:
    path = RESULTS / f"request_{index:06d}.json"
    body = body_for(record)

    if path.exists():
        item = json.loads(path.read_text(encoding="utf-8"))

        if (
            item.get("record") != record
            or item.get("request_body") != body
        ):
            raise SystemExit(f"旧记录与当前请求不一致：{path}")

        response = item.get("response")

        if response is not None:
            if normal_response(response):
                reused += 1
            else:
                existing_needs_check += 1
            continue

        if len(item.get("attempts", [])) >= MAX_ATTEMPTS:
            exhausted += 1
            continue

        retryable_existing += 1
    else:
        missing += 1

    tasks.put((index, record))

initial_count = tasks.qsize()

print("固定题目对数：", len(selected_pairs))
print("类别分布：", category_counts)
print("本轮实验条件：", sorted(selected_conditions))
print("本轮所需请求：", len(selected))
print("正常返回，直接复用：", reused)
print("已有响应但需检查：", existing_needs_check)
print("已达尝试上限，暂时跳过：", exhausted)
print("尚未请求：", missing)
print("未完成、可继续尝试：", retryable_existing)
print("本次待处理：", initial_count)
print("并发数：", WORKERS)
print("按 Control+C 停止，等待进行中的请求保存后退出。")

started = time.monotonic()


def report(message, category):
    with lock:
        stats[category] += 1

        handled = sum(stats.values())
        elapsed = time.monotonic() - started
        rate = handled / elapsed if elapsed else 0
        remaining = initial_count - handled
        hours = remaining / rate / 3600 if rate else 0

        print(
            f"\n{message}\n"
            f"本轮已处理 {handled}/{initial_count}；"
            f"平均速度 {rate * 60:.1f} 条/分钟；"
            f"预计剩余 {hours:.2f} 小时",
            flush=True,
        )


def worker():
    with OpenAI(
        api_key=api_key,
        base_url=config["base_url"],
        max_retries=0,
        timeout=180,
    ) as client:
        while not stop.is_set():
            try:
                index, record = tasks.get_nowait()
            except queue.Empty:
                return

            path = RESULTS / f"request_{index:06d}.json"
            body = body_for(record)

            if path.exists():
                item = json.loads(path.read_text(encoding="utf-8"))
            else:
                item = {
                    "state": "new",
                    "record": record,
                    "request_body": body,
                    "attempts": [],
                }

            attempts = item.setdefault("attempts", [])

            # 上次中断时，请求是否完成和扣费可能未知
            for attempt in attempts:
                if attempt["state"] == "pending":
                    attempt["state"] = "uncertain"

            while len(attempts) < MAX_ATTEMPTS:
                if stop.is_set():
                    return

                attempt = {
                    "number": len(attempts) + 1,
                    "state": "pending",
                    "started_at": time.time(),
                }

                attempts.append(attempt)
                item["state"] = "pending"
                save(path, item)

                try:
                    response = client.chat.completions.create(**body)

                except (APIConnectionError, APIStatusError) as error:
                    status = getattr(error, "status_code", None)
                    code = getattr(error, "code", None)

                    attempt.update({
                        "state": "error",
                        "error_type": type(error).__name__,
                        "status_code": status,
                        "error_code": code,
                        "message": str(error),
                        "finished_at": time.time(),
                    })

                    item["state"] = "failed"
                    save(path, item)

                    quota_error = any(
                        word in str(code).lower()
                        for word in ("quota", "balance", "credit")
                    )

                    retryable = (
                        isinstance(error, APIConnectionError)
                        or status in (408, 409, 429)
                        or (status is not None and status >= 500)
                    )

                    if (
                        status in (401, 402, 403)
                        or quota_error
                        or not retryable
                    ):
                        stop.set()
                        report(f"停止：{error}", "failed")
                        return

                    if len(attempts) < MAX_ATTEMPTS:
                        delay = 10 * 2 ** (len(attempts) - 1)

                        with lock:
                            print(
                                f"\n{record['request_id']} 调用失败，"
                                f"{delay} 秒后重试："
                                f"{type(error).__name__}，HTTP {status}",
                                flush=True,
                            )

                        stop.wait(delay)

                    continue

                except Exception as error:
                    attempt.update({
                        "state": "error",
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "finished_at": time.time(),
                    })

                    item["state"] = "failed"
                    save(path, item)

                    stop.set()
                    report(f"异常停止：{error}", "failed")
                    return

                raw = response.model_dump(mode="json")

                attempt["state"] = "returned"
                attempt["finished_at"] = time.time()

                item["state"] = "returned"
                item["response"] = raw
                save(path, item)

                finish = (
                    response.choices[0].finish_reason
                    if response.choices else "no_choices"
                )

                category = (
                    "completed"
                    if normal_response(raw)
                    else "needs_check"
                )

                usage_text = ""
                if response.usage:
                    usage_text = (
                        f"；输入 {response.usage.prompt_tokens} token"
                        f"；输出 {response.usage.completion_tokens} token"
                    )

                report(
                    f"已保存 {record['request_id']}；"
                    f"结束原因：{finish}{usage_text}",
                    category,
                )
                break

            else:
                report(
                    f"达到尝试上限：{record['request_id']}",
                    "failed",
                )


executor = ThreadPoolExecutor(max_workers=WORKERS)

try:
    futures = [
        executor.submit(worker)
        for _ in range(WORKERS)
    ]

    for future in futures:
        future.result()

except KeyboardInterrupt:
    print(
        "\n正在停止，等待已发出的请求返回并保存……",
        flush=True,
    )
    stop.set()

except Exception:
    stop.set()
    raise

finally:
    stop.set()
    executor.shutdown(wait=True)

print("\n本轮统计：", stats)
print("本次未开始处理的队列数量：", tasks.qsize())
print("结果目录：", RESULTS)
print("最终是否齐全，需要重新核查全部8640条目标请求。")