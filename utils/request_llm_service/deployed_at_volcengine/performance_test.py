"""
火山云部署 LLM 服务压测脚本。

在指定时长内，以固定并发数随机抽取 news_1000.csv 样本持续请求，
统计吞吐量、延迟分位数、成功率及 Token 消耗等指标。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import random
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from req_qwen_example import GENERATE_CFG, MODEL, llm_chat, load_api_url

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config" / "performance_test.yaml"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else SCRIPT_DIR / p


def load_performance_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_system_prompt(config: dict[str, Any]) -> str:
    prompt_file = config.get("system_prompt_file")
    if not prompt_file:
        raise ValueError("配置中缺少 system_prompt_file 字段")
    path = resolve_path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"未找到 system prompt 文件: {path}")
    return path.read_text(encoding="utf-8").strip()


@dataclass
class RequestRecord:
    success: bool
    latency_s: float
    news_id: str
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.records: list[RequestRecord] = []
        self._in_flight = 0

    def begin_request(self) -> None:
        with self._lock:
            self._in_flight += 1

    def end_request(self) -> None:
        with self._lock:
            self._in_flight -= 1

    def add(self, record: RequestRecord) -> None:
        with self._lock:
            self.records.append(record)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            total = len(self.records)
            ok = sum(1 for r in self.records if r.success)
            return {
                "total": total,
                "ok": ok,
                "fail": total - ok,
                "in_flight": self._in_flight,
            }

    @property
    def total(self) -> int:
        return len(self.records)

    @property
    def successes(self) -> list[RequestRecord]:
        return [r for r in self.records if r.success]

    @property
    def failures(self) -> list[RequestRecord]:
        return [r for r in self.records if not r.success]


def load_news_samples(csv_path: Path) -> list[dict[str, str]]:
    """加载 CSV，返回含 news_id / title / content 的样本列表。"""
    samples: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = (row.get("TITLE") or "").strip()
            content = (row.get("CONTTEXT") or "").strip()
            if not title and not content:
                continue
            samples.append(
                {
                    "news_id": (row.get("INFOCODE") or "").strip(),
                    "title": title,
                    "content": content,
                }
            )
    if not samples:
        raise ValueError(f"{csv_path} 中未找到有效样本（需 TITLE 或 CONTTEXT）")
    return samples


def build_messages(sample: dict[str, str], system_prompt: str) -> list[dict[str, str]]:
    """构造单次请求的 messages；system prompt 前拼接当前时间以防 KV cache 命中。"""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    system_content = f"当前时间：{now_str}\n\n{system_prompt}"
    user_content = f"标题：{sample['title']}\n\n正文：{sample['content']}"
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def extract_usage(rsp: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    usage = rsp.get("usage") or {}
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    return prompt, completion, total


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p / 100.0
    f_idx = int(k)
    c_idx = min(f_idx + 1, len(sorted_vals) - 1)
    return sorted_vals[f_idx] + (sorted_vals[c_idx] - sorted_vals[f_idx]) * (k - f_idx)


def send_one_request(
    api_url: str,
    model: str,
    sample: dict[str, str],
    generate_cfg: dict[str, Any],
    system_prompt: str,
) -> RequestRecord:
    news_id = sample["news_id"]
    start = time.perf_counter()
    try:
        rsp = llm_chat(
            url=api_url,
            model=model,
            messages=build_messages(sample, system_prompt),
            stream=False,
            generate_cfg=generate_cfg,
        )
        latency_s = time.perf_counter() - start
        if not rsp or not rsp.get("choices"):
            return RequestRecord(
                success=False,
                latency_s=latency_s,
                news_id=news_id,
                error=f"接口返回异常: {rsp}",
            )
        prompt_t, completion_t, total_t = extract_usage(rsp)
        return RequestRecord(
            success=True,
            latency_s=latency_s,
            news_id=news_id,
            prompt_tokens=prompt_t,
            completion_tokens=completion_t,
            total_tokens=total_t,
        )
    except Exception as e:
        latency_s = time.perf_counter() - start
        return RequestRecord(
            success=False,
            latency_s=latency_s,
            news_id=news_id,
            error=str(e),
        )


def worker_loop(
    worker_id: int,
    api_url: str,
    model: str,
    samples: list[dict[str, str]],
    generate_cfg: dict[str, Any],
    system_prompt: str,
    deadline: float,
    metrics: MetricsCollector,
    detail_file: Path | None,
    detail_lock: threading.Lock,
) -> None:
    """单个 worker 在 deadline 前持续发请求。"""
    while time.time() < deadline:
        sample = random.choice(samples)
        metrics.begin_request()
        try:
            record = send_one_request(
                api_url, model, sample, generate_cfg, system_prompt
            )
        finally:
            metrics.end_request()
        metrics.add(record)

        if detail_file is not None:
            detail = {
                "worker_id": worker_id,
                "news_id": record.news_id,
                "success": record.success,
                "latency_s": round(record.latency_s, 4),
                "prompt_tokens": record.prompt_tokens,
                "completion_tokens": record.completion_tokens,
                "total_tokens": record.total_tokens,
                "error": record.error,
                "timestamp": record.timestamp,
            }
            with detail_lock:
                with open(detail_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(detail, ensure_ascii=False) + "\n")


def run_progress_monitor(
    metrics: MetricsCollector,
    wall_start: float,
    duration_s: float,
    stop_event: threading.Event,
) -> None:
    """后台线程：每秒刷新进度条，展示时间与请求统计。"""
    total_s = max(int(duration_s), 1)
    with tqdm(total=total_s, unit="s", desc="压测进度") as pbar:
        while not stop_event.is_set():
            elapsed = time.perf_counter() - wall_start
            snap = metrics.snapshot()
            rps = snap["total"] / elapsed if elapsed > 0 else 0.0

            if elapsed >= duration_s:
                pbar.set_description("收尾(在途请求)")
                pbar.n = total_s
            else:
                pbar.set_description("压测进度")
                pbar.n = min(int(elapsed), total_s)

            pbar.set_postfix(
                请求=snap["total"],
                成功=snap["ok"],
                进行中=snap["in_flight"],
                RPS=f"{rps:.2f}",
                refresh=False,
            )
            pbar.refresh()
            stop_event.wait(1)

        pbar.n = total_s
        pbar.refresh()


def summarize(
    metrics: MetricsCollector,
    wall_time_s: float,
    concurrency: int,
    duration_s: float,
    model: str,
) -> dict[str, Any]:
    all_records = metrics.records
    ok_records = metrics.successes
    fail_records = metrics.failures

    latencies = [r.latency_s for r in ok_records]
    latencies_all = [r.latency_s for r in all_records]

    prompt_tokens = [r.prompt_tokens for r in ok_records if r.prompt_tokens is not None]
    completion_tokens = [
        r.completion_tokens for r in ok_records if r.completion_tokens is not None
    ]
    total_tokens = [r.total_tokens for r in ok_records if r.total_tokens is not None]

    error_counts: dict[str, int] = {}
    for r in fail_records:
        key = (r.error or "unknown")[:120]
        error_counts[key] = error_counts.get(key, 0) + 1

    summary: dict[str, Any] = {
        "test_config": {
            "duration_s": duration_s,
            "wall_time_s": round(wall_time_s, 2),
            "concurrency": concurrency,
            "model": model,
        },
        "throughput": {
            "total_requests": len(all_records),
            "successful_requests": len(ok_records),
            "failed_requests": len(fail_records),
            "success_rate": round(len(ok_records) / len(all_records), 4)
            if all_records
            else 0.0,
            "requests_per_second": round(len(all_records) / wall_time_s, 4),
            "successful_rps": round(len(ok_records) / wall_time_s, 4),
            "seconds_per_request": round(wall_time_s / len(all_records), 4)
            if all_records
            else None,
        },
        "latency_s": {},
        "tokens": {},
        "errors": error_counts,
    }

    if latencies:
        summary["latency_s"] = {
            "successful_only": True,
            "min": round(min(latencies), 4),
            "max": round(max(latencies), 4),
            "mean": round(statistics.mean(latencies), 4),
            "median_p50": round(percentile(latencies, 50) or 0, 4),
            "p90": round(percentile(latencies, 90) or 0, 4),
            "p95": round(percentile(latencies, 95) or 0, 4),
            "p99": round(percentile(latencies, 99) or 0, 4),
        }
        if len(latencies) > 1:
            summary["latency_s"]["stdev"] = round(statistics.stdev(latencies), 4)
    elif latencies_all:
        summary["latency_s"] = {
            "successful_only": False,
            "note": "无成功请求，以下为全部请求延迟",
            "mean": round(statistics.mean(latencies_all), 4),
        }

    if completion_tokens:
        total_completion = sum(completion_tokens)
        summary["tokens"] = {
            "total_prompt_tokens": sum(prompt_tokens) if prompt_tokens else None,
            "total_completion_tokens": total_completion,
            "total_tokens": sum(total_tokens) if total_tokens else None,
            "avg_prompt_tokens": round(statistics.mean(prompt_tokens), 2)
            if prompt_tokens
            else None,
            "avg_completion_tokens": round(statistics.mean(completion_tokens), 2),
            "completion_tokens_per_second": round(total_completion / wall_time_s, 2),
        }

    return summary


def format_summary(summary: dict[str, Any]) -> str:
    """生成终端可读压测报告文本。"""
    cfg = summary["test_config"]
    tp = summary["throughput"]
    lat = summary.get("latency_s") or {}
    tok = summary.get("tokens") or {}

    lines: list[str] = [
        "",
        "=" * 60,
        "压测结果摘要",
        "=" * 60,
        f"模型:           {cfg['model']}",
        f"目标时长:       {cfg['duration_s']:.0f}s",
        f"实际耗时 T:     {cfg['wall_time_s']:.1f}s",
        f"并发数:         {cfg['concurrency']}",
        "-" * 60,
        "吞吐量",
        f"  总请求数 N:   {tp['total_requests']}",
        f"  成功 / 失败:  {tp['successful_requests']} / {tp['failed_requests']}",
        f"  成功率:       {tp['success_rate'] * 100:.2f}%",
        f"  总 RPS:       {tp['requests_per_second']:.4f} req/s",
        f"  成功 RPS:     {tp['successful_rps']:.4f} req/s",
    ]

    if tp.get("seconds_per_request") is not None:
        t_per_n = tp["seconds_per_request"]
        lines.extend(
            [
                f"  平均吞吐间隔 T/N: {t_per_n:.4f} s/req",
                "    (墙钟时间 / 完成请求数，用于估算批量任务总耗时)",
                f"    例: 处理 1000 条约需 {1000 * t_per_n / 60:.1f} 分钟",
            ]
        )

    if lat:
        lines.append("-" * 60)
        lines.append(
            "端到端延迟 (秒，仅统计成功请求)"
            if lat.get("successful_only")
            else "端到端延迟 (秒)"
        )
        lines.append("  (单条请求的等待时间，含排队与生成；与上方 T/N 含义不同)")
        for key in ("min", "mean", "median_p50", "p90", "p95", "p99", "max"):
            if key in lat:
                label = {
                    "min": "最小",
                    "mean": "平均",
                    "median_p50": "P50",
                    "p90": "P90",
                    "p95": "P95",
                    "p99": "P99",
                    "max": "最大",
                }[key]
                lines.append(f"  {label:8s}     {lat[key]:.4f}s")

    if tok:
        lines.extend(
            [
                "-" * 60,
                "Token 消耗 (成功请求)",
            ]
        )
        if tok.get("avg_prompt_tokens") is not None:
            lines.append(f"  平均 prompt:     {tok['avg_prompt_tokens']:.1f}")
        lines.append(f"  平均 completion: {tok['avg_completion_tokens']:.1f}")
        lines.append(
            f"  输出 Token 吞吐: {tok['completion_tokens_per_second']:.2f} tokens/s"
        )
        if tok.get("total_tokens") is not None:
            lines.append(f"  总 Token:        {tok['total_tokens']}")

    if summary.get("errors"):
        lines.extend(["-" * 60, "错误分布 (Top)"])
        for err, cnt in sorted(summary["errors"].items(), key=lambda x: -x[1])[:5]:
            lines.append(f"  [{cnt}x] {err}")

    lines.extend(["=" * 60, ""])
    return "\n".join(lines)


def print_summary(summary: dict[str, Any]) -> str:
    text = format_summary(summary)
    print(text, end="")
    return text


def save_summary_report(summary: dict[str, Any], report_path: Path) -> str:
    text = format_summary(summary)
    report_path.write_text(text, encoding="utf-8")
    print(text, end="")
    return text


def build_run_tag(concurrency: int, duration_s: float) -> str:
    """生成含压测配置与时间戳的文件名后缀，便于 outputs 目录下区分不同配置。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dur = int(duration_s) if duration_s == int(duration_s) else str(duration_s).replace(".", "p")
    return f"c{concurrency}_d{dur}s_{ts}"


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=argparse.SUPPRESS,
    )
    pre_args, _ = pre_parser.parse_known_args()
    cfg = load_performance_config(pre_args.config)
    test_cfg = cfg.get("test") or {}

    parser = argparse.ArgumentParser(
        description="火山云 LLM 服务压测：在指定时长内以固定并发持续请求。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 压测 5 分钟，并发 4（使用 config/performance_test.yaml 默认值）
  python performance_test.py --duration 300 --concurrency 4

  # 压测 10 分钟，并发 8，保存明细
  python performance_test.py --duration 600 --concurrency 8 --save-details
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"配置文件路径，默认 {DEFAULT_CONFIG_PATH.relative_to(SCRIPT_DIR)}",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=test_cfg.get("duration_s", 300),
        help="压测时长（秒），默认从配置文件读取",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=test_cfg.get("concurrency", 4),
        help="并发 worker 数，默认从配置文件读取",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=resolve_path(test_cfg.get("csv", "inputs/news_1000.csv")),
        help="输入 CSV 路径，默认从配置文件读取",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=test_cfg.get("model", MODEL),
        help="模型名称，默认从配置文件读取",
    )
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="将每条请求的明细写入 outputs/performance_test_details.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="汇总结果 JSON 输出路径（默认 outputs/performance_test_summary_c{N}_d{T}s_<timestamp>.json）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（可选，便于复现抽样序列）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.duration <= 0:
        print("错误: --duration 必须大于 0", file=sys.stderr)
        return 1
    if args.concurrency <= 0:
        print("错误: --concurrency 必须大于 0", file=sys.stderr)
        return 1

    if args.seed is not None:
        random.seed(args.seed)

    cfg = load_performance_config(args.config)
    system_prompt = load_system_prompt(cfg)
    api_url = load_api_url()
    samples = load_news_samples(args.csv)

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_tag = build_run_tag(args.concurrency, args.duration)
    summary_path = args.output or (
        DEFAULT_OUTPUT_DIR / f"performance_test_summary_{run_tag}.json"
    )
    report_path = DEFAULT_OUTPUT_DIR / f"performance_test_report_{run_tag}.txt"
    detail_path: Path | None = None
    if args.save_details:
        detail_path = DEFAULT_OUTPUT_DIR / f"performance_test_details_{run_tag}.jsonl"
        detail_path.write_text("", encoding="utf-8")

    metrics = MetricsCollector()
    detail_lock = threading.Lock()

    print(f"加载样本: {len(samples)} 条 ({args.csv})")
    print(f"配置文件: {args.config}")
    print(f"System prompt: {cfg.get('system_prompt_file')}")
    print(f"开始压测: duration={args.duration}s, concurrency={args.concurrency}, model={args.model}")
    print(f"API: {api_url[:60]}...")

    deadline = time.time() + args.duration
    wall_start = time.perf_counter()

    print(f"准备并发压测，最大并发数：{args.concurrency}")
    print(
        "提示: 压测将持续发请求，首批 LLM 响应可能需数十秒；"
        "进度条每秒更新，全部 worker 结束后输出汇总。"
    )

    stop_event = threading.Event()
    progress_thread = threading.Thread(
        target=run_progress_monitor,
        args=(metrics, wall_start, args.duration, stop_event),
        daemon=True,
    )
    progress_thread.start()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [
            executor.submit(
                worker_loop,
                i,
                api_url,
                args.model,
                samples,
                GENERATE_CFG,
                system_prompt,
                deadline,
                metrics,
                detail_path,
                detail_lock,
            )
            for i in range(args.concurrency)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    stop_event.set()
    progress_thread.join()
    print()

    wall_time_s = time.perf_counter() - wall_start
    summary = summarize(
        metrics, wall_time_s, args.concurrency, args.duration, args.model
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_summary_report(summary, report_path)

    print(f"报告已保存: {report_path}")
    print(f"汇总已保存: {summary_path}")
    if detail_path:
        print(f"明细已保存: {detail_path}")

    return 0 if summary["throughput"]["failed_requests"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
