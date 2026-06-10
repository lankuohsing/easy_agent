"""
大模型客户端模块

提供统一的 LLM 调用接口，支持：
  - OpenAICompatibleClient：OpenAI 兼容 API（Qwen、DeepSeek、Ollama 等）
  - MockLLMClient：本地 Mock，无需凭证，便于开发与测试
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from openai import OpenAI

from chat_agent.config_loader import ModelProviderConfig


Message = dict[str, str]


class BaseLLMClient(ABC):
    """大模型客户端抽象基类。"""

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        ...


class OpenAICompatibleClient(BaseLLMClient):
    """
    OpenAI 兼容 API 客户端。

    适用于 DashScope（Qwen）、DeepSeek、Ollama、vLLM 等提供 OpenAI 格式接口的服务。
    无鉴权服务（auth_required=false）时使用占位 api_key 即可。
    """

    def __init__(self, provider: ModelProviderConfig) -> None:
        self._provider = provider
        api_key = provider.api_key if provider.api_key else "EMPTY"
        self._client = OpenAI(
            api_key=api_key,
            base_url=provider.base_url,
            timeout=provider.timeout,
        )

    @property
    def provider_name(self) -> str:
        return self._provider.name

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._provider.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature if temperature is not None else self._provider.temperature,
            max_tokens=max_tokens if max_tokens is not None else self._provider.max_tokens,
        )
        content = response.choices[0].message.content
        return (content or "").strip()


class MockLLMClient(BaseLLMClient):
    """Mock 客户端，用于无凭证时的流程验证。"""

    def __init__(self, delay: float = 0.3, provider_name: str = "mock") -> None:
        self._delay = delay
        self._call_count = 0
        self._provider_name = provider_name

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self._call_count += 1
        if self._delay > 0:
            time.sleep(self._delay)

        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break

        turn_count = sum(1 for m in messages if m.get("role") == "user")

        return (
            f"[Mock 回复 #{self._call_count}] "
            f"我已收到你的消息（当前会话第 {turn_count} 轮）。\n"
            f"你说：「{last_user}」\n"
            f"（Mock 模式：请配置 secrets.yaml 并关闭 use_mock 以接入真实模型）"
        )


def create_llm_client(
    provider: ModelProviderConfig,
    *,
    use_mock: bool = False,
    mock_delay: float = 0.3,
) -> BaseLLMClient:
    """
    根据 provider 配置创建 LLM 客户端。

    Args:
        provider: 合并后的模型服务配置
        use_mock: True 时返回 MockLLMClient
        mock_delay: Mock 模拟延迟（秒）
    """
    if use_mock:
        return MockLLMClient(delay=mock_delay, provider_name=provider.name)

    if provider.type == "openai_compatible":
        return OpenAICompatibleClient(provider)

    raise ValueError(
        f"不支持的 provider 类型: '{provider.type}'（provider={provider.name}）。"
        f"目前仅支持 openai_compatible。"
    )


# 向后兼容旧名称
QwenClient = OpenAICompatibleClient
