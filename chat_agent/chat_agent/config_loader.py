"""
配置加载模块

配置分为两层：
  - config.yaml       非敏感配置（模型参数、active 服务、Agent 行为等），可提交 Git
  - secrets.yaml      敏感凭证（各 provider 的 base_url、api_key），已被 gitignore

支持在 config.yaml 中定义多个模型服务（providers），通过 llm.active 选择当前使用的服务。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_SECRETS_PATH = PROJECT_ROOT / "config" / "secrets.yaml"
SECRETS_EXAMPLE_PATH = PROJECT_ROOT / "config" / "secrets.example.yaml"

# 占位符 api_key，用于检测用户是否尚未填写真实密钥
_PLACEHOLDER_KEY_PREFIXES = ("sk-your-", "your-api-key", "changeme")


@dataclass
class ModelProviderConfig:
    """
    单个模型服务的完整配置（主配置 + 敏感凭证合并后）。

    Attributes:
        name: provider 标识，与 config.yaml / secrets.yaml 中的 key 对应
        type: 客户端类型：openai_compatible | http_chat_completions
        model: 模型名称
        base_url: openai_compatible 时为 API 前缀（如 .../v1）；
                  http_chat_completions 时为完整 endpoint（如 .../v1/chat/completions）
        api_key: API Key（无鉴权服务可为 None）
        auth_required: 是否必须提供 api_key
        temperature, max_tokens, timeout: 推理参数
        connect_timeout: http 请求连接超时（秒）
        generate_cfg: 额外合并进请求体的参数（top_p、top_k 等）
        enable_thinking: Qwen thinking 开关；None 表示不写入 chat_template_kwargs
        stop_token: 停止词
        stream: 是否流式（Agent CLI 默认 false）
    """

    name: str
    type: str = "openai_compatible"
    model: str = ""
    base_url: str = ""
    api_key: str | None = None
    auth_required: bool = True
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60
    connect_timeout: int = 10
    generate_cfg: dict[str, Any] = field(default_factory=dict)
    enable_thinking: bool | None = None
    stop_token: str | None = None
    stream: bool = False


@dataclass
class LLMConfig:
    """大模型相关配置。"""

    active: str = "qwen"
    secrets_file: str = "config/secrets.yaml"
    providers: dict[str, ModelProviderConfig] = field(default_factory=dict)

    def get_active_provider(self) -> ModelProviderConfig:
        """返回当前选中的模型服务配置。"""
        if self.active not in self.providers:
            available = ", ".join(sorted(self.providers)) or "（无）"
            raise ValueError(
                f"未知的 active provider: '{self.active}'。"
                f"请在 config.yaml 的 llm.providers 中定义，或在 secrets.yaml 中补充凭证。"
                f"当前已配置: {available}"
            )
        return self.providers[self.active]

    def list_provider_names(self) -> list[str]:
        return sorted(self.providers.keys())


@dataclass
class MemoryConfig:
    """会话记忆持久化配置。"""

    enabled: bool = True
    storage_dir: str = "memories"


@dataclass
class AgentConfig:
    """Agent 行为参数。"""

    system_prompt_file: str = "prompts/system_prompt.txt"
    system_prompt: str | None = None
    show_system_prompt_on_start: bool = False
    memory: MemoryConfig = field(default_factory=MemoryConfig)


@dataclass
class RuntimeConfig:
    """运行时开关。"""

    use_mock: bool = False
    mock_delay: float = 0.3


@dataclass
class AppConfig:
    """应用全局配置。"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def get_active_provider(self) -> ModelProviderConfig:
        return self.llm.get_active_provider()


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"配置文件不存在: {path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _is_placeholder_api_key(api_key: str | None) -> bool:
    if not api_key:
        return True
    lowered = api_key.strip().lower()
    return any(lowered.startswith(p.lower()) for p in _PLACEHOLDER_KEY_PREFIXES)


def _parse_provider(
    name: str,
    provider_raw: dict[str, Any],
    secrets_raw: dict[str, Any] | None,
    *,
    strict: bool = True,
) -> ModelProviderConfig:
    """合并主配置与 secrets 中的单个 provider。"""
    creds = (secrets_raw or {}).get(name, {}) or {}
    if not isinstance(creds, dict):
        creds = {}

    auth_required = bool(provider_raw.get("auth_required", True))
    # api_url 与 base_url 均支持；http_chat_completions 类型应填完整 endpoint
    api_url = str(creds.get("api_url") or creds.get("base_url", "")).strip()
    api_key_raw = creds.get("api_key")
    api_key = str(api_key_raw).strip() if api_key_raw is not None else None
    if api_key == "":
        api_key = None

    if not api_url and strict:
        raise ValueError(
            f"模型服务 '{name}' 缺少 API 地址。"
            f"请在 secrets.yaml 的 providers.{name}.api_url（或 base_url）中配置。"
        )

    if auth_required and strict:
        if not api_key or _is_placeholder_api_key(api_key):
            raise ValueError(
                f"模型服务 '{name}' 需要 api_key。"
                f"请在 secrets.yaml 的 providers.{name}.api_key 中填入真实密钥，"
                f"或将 config.yaml 中 providers.{name}.auth_required 设为 false。"
            )

    generate_cfg = provider_raw.get("generate_cfg") or {}
    if not isinstance(generate_cfg, dict):
        generate_cfg = {}

    enable_thinking = provider_raw.get("enable_thinking")
    if enable_thinking is not None:
        enable_thinking = bool(enable_thinking)

    return ModelProviderConfig(
        name=name,
        type=str(provider_raw.get("type", "openai_compatible")),
        model=str(provider_raw.get("model", "")),
        base_url=api_url,
        api_key=api_key,
        auth_required=auth_required,
        temperature=float(provider_raw.get("temperature", 0.7)),
        max_tokens=int(provider_raw.get("max_tokens", 2048)),
        timeout=int(provider_raw.get("timeout", 60)),
        connect_timeout=int(provider_raw.get("connect_timeout", 10)),
        generate_cfg=generate_cfg,
        enable_thinking=enable_thinking,
        stop_token=provider_raw.get("stop_token"),
        stream=bool(provider_raw.get("stream", False)),
    )


def _parse_memory_config(raw: Any) -> MemoryConfig:
    if not isinstance(raw, dict):
        return MemoryConfig()
    return MemoryConfig(
        enabled=bool(raw.get("enabled", True)),
        storage_dir=str(raw.get("storage_dir", MemoryConfig.storage_dir)),
    )


def load_config(
    config_path: Path | None = None,
    secrets_path: Path | None = None,
    *,
    active_provider: str | None = None,
    strict: bool = True,
) -> AppConfig:
    """
    加载并合并主配置与敏感凭证。

    Args:
        config_path: 主配置文件路径，默认 config/config.yaml
        secrets_path: 敏感凭证路径，默认读取 config 中的 secrets_file
        active_provider: 覆盖 config 中的 llm.active
        strict: True 时校验 base_url / api_key；Mock 模式可设为 False
    """
    path = config_path or DEFAULT_CONFIG_PATH
    raw = load_yaml(path, required=True)

    llm_raw = raw.get("llm", {})
    agent_raw = raw.get("agent", {})
    runtime_raw = raw.get("runtime", {})

    secrets_file = secrets_path or _resolve_path(
        str(llm_raw.get("secrets_file", DEFAULT_SECRETS_PATH.relative_to(PROJECT_ROOT)))
    )
    secrets_data = load_yaml(secrets_file, required=False)
    secrets_providers = secrets_data.get("providers", {}) or {}

    providers_raw = llm_raw.get("providers", {}) or {}
    if not providers_raw:
        raise ValueError("config.yaml 中 llm.providers 为空，请至少定义一个模型服务。")

    providers: dict[str, ModelProviderConfig] = {} # 有效的模型服务信息
    merge_errors: list[str] = []

    for name, provider_def in providers_raw.items():
        if not isinstance(provider_def, dict):
            continue
        try:
            providers[name] = _parse_provider(
                name, provider_def, secrets_providers, strict=strict
            )
        except ValueError as exc:
            merge_errors.append(str(exc))

    active = active_provider or str(llm_raw.get("active", "qwen"))

    config = AppConfig(
        llm=LLMConfig(
            active=active,
            secrets_file=str(secrets_file.relative_to(PROJECT_ROOT))
            if secrets_file.is_relative_to(PROJECT_ROOT)
            else str(secrets_file),
            providers=providers,
        ),
        agent=AgentConfig(
            system_prompt_file=str(
                agent_raw.get("system_prompt_file", AgentConfig.system_prompt_file)
            ),
            system_prompt=agent_raw.get("system_prompt"),
            show_system_prompt_on_start=bool(
                agent_raw.get("show_system_prompt_on_start", False)
            ),
            memory=_parse_memory_config(agent_raw.get("memory", {})),
        ),
        runtime=RuntimeConfig(
            use_mock=bool(runtime_raw.get("use_mock", False)),
            mock_delay=float(runtime_raw.get("mock_delay", RuntimeConfig.mock_delay)),
        ),
    )

    # active provider 必须可用；其他 provider 配置不完整时仅警告性记录在异常中
    if active not in providers:
        detail = "\n  - ".join(merge_errors) if merge_errors else "请检查 config.yaml 与 secrets.yaml"
        raise ValueError(
            f"无法加载 active 模型服务 '{active}'。\n  - {detail}\n"
            f"提示: cp {SECRETS_EXAMPLE_PATH} {DEFAULT_SECRETS_PATH}"
        )

    return config


def load_system_prompt(config: AppConfig) -> str:
    """加载 System Prompt（内联配置优先于文件）。"""
    if config.agent.system_prompt:
        return config.agent.system_prompt.strip()

    prompt_path = _resolve_path(config.agent.system_prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"System Prompt 文件不存在: {prompt_path}\n"
            f"请创建该文件，或在 config.yaml 的 agent.system_prompt 中直接填写。"
        )
    return prompt_path.read_text(encoding="utf-8").strip()
