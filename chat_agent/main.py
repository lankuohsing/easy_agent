#!/usr/bin/env python3
"""
Chat Agent 入口脚本

用法：
  cd chat_agent
  pip install -r requirements.txt
  cp config/secrets.example.yaml config/secrets.yaml   # 首次接入真实 API
  python main.py                    # 使用 config.yaml 中的 llm.active
  python main.py --provider qwen    # 指定模型服务
  python main.py --mock             # Mock 模式（无需 secrets.yaml）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chat_agent.agent import ChatAgent
from chat_agent.cli import run_cli
from chat_agent.config_loader import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_SECRETS_PATH,
    SECRETS_EXAMPLE_PATH,
    load_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="最小多轮对话 Chat Agent（支持多模型服务 / Mock）"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help=f"主配置文件路径（默认: {DEFAULT_CONFIG_PATH}）",
    )
    parser.add_argument(
        "--secrets",
        type=Path,
        default=None,
        help="敏感凭证文件路径（默认读取 config.yaml 中的 secrets_file）",
    )
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        default=None,
        help="指定模型服务（覆盖 config.yaml 的 llm.active）",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="强制 Mock 模式（无需 secrets.yaml，覆盖 runtime.use_mock）",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="列出 config.yaml 中定义的所有模型服务并退出",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    use_mock = args.mock
    strict = not (use_mock or args.list_providers)
    try:
        config = load_config(
            args.config,
            secrets_path=args.secrets,
            active_provider=args.provider,
            strict=strict,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        print(
            f"\n提示: cp {SECRETS_EXAMPLE_PATH} {DEFAULT_SECRETS_PATH} 并填入凭证，"
            f"或使用 Mock 模式: python main.py --mock",
            file=sys.stderr,
        )
        return 1

    if args.list_providers:
        names = sorted(config.llm.providers.keys())
        active = config.llm.active
        print("已配置的模型服务:")
        for name in names:
            marker = " (active)" if name == active else ""
            model = config.llm.providers[name].model
            print(f"  - {name}: {model}{marker}")
        return 0

    if use_mock:
        config.runtime.use_mock = True

    try:
        agent = ChatAgent(config)
    except ValueError as exc:
        print(f"初始化失败: {exc}", file=sys.stderr)
        return 1

    run_cli(agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
