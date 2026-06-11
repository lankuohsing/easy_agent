"""
大模型客户端模块

提供统一的 LLM 调用接口，支持：
  - OpenAICompatibleClient：OpenAI SDK，base_url 为 API 前缀（.../v1）
  - RequestsChatCompletionsClient：requests 直 POST 完整 endpoint，兼容自建网关
  - MockLLMClient：本地 Mock，无需凭证
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import requests
from openai import OpenAI

from chat_agent.config_loader import ModelProviderConfig


Message = dict[str, str]


class BaseLLMClient(ABC):
    """大模型客户端抽象基类。"""

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,# 它后面的参数不能用位置方式传，必须写参数名。
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        ...# 在抽象方法里，函数体只写 ...，表示：这里没有实现，子类必须重写。


class OpenAICompatibleClient(BaseLLMClient):
    """
    OpenAI 兼容 API 客户端（openai SDK）。

    secrets 中 base_url 应填 API 前缀，如 https://dashscope.../v1
    SDK 会自动追加 /chat/completions。
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
        extra: dict[str, Any] = {}
        if self._provider.generate_cfg:
            extra["extra_body"] = self._provider.generate_cfg

        response = self._client.chat.completions.create(
            model=self._provider.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature if temperature is not None else self._provider.temperature,
            max_tokens=max_tokens if max_tokens is not None else self._provider.max_tokens,
            stream=self._provider.stream,
            **extra,
        )
        content = response.choices[0].message.content
        return (content or "").strip()


class RequestsChatCompletionsClient(BaseLLMClient):
    """
    requests 直连接客户端（http_chat_completions 类型）。

    与 req_qwen_example.py 风格一致：
      - secrets 中 api_url / base_url 填完整 endpoint（含 /chat/completions）
      - 无 api_key 时不发送 Authorization 头
      - 支持 generate_cfg、chat_template_kwargs.enable_thinking 等扩展字段
    """

    def __init__(self, provider: ModelProviderConfig) -> None:
        self._provider = provider

    @property
    def provider_name(self) -> str:
        return self._provider.name

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._provider.api_key:
            headers["Authorization"] = f"Bearer {self._provider.api_key}"
        return headers

    def _build_payload(
        self,
        messages: list[Message],
        *,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._provider.model,
            "messages": messages,
            "stream": self._provider.stream,
            "max_tokens": max_tokens if max_tokens is not None else self._provider.max_tokens,
            "tools": [],
            "temperature": temperature if temperature is not None else self._provider.temperature,
        }
        payload = {**payload, **self._provider.generate_cfg}

        if self._provider.stop_token:
            payload["stop"] = self._provider.stop_token

        if self._provider.enable_thinking is not None:
            chat_kwargs = payload.setdefault("chat_template_kwargs", {})
            if not isinstance(chat_kwargs, dict):
                chat_kwargs = {}
                payload["chat_template_kwargs"] = chat_kwargs
            chat_kwargs["enable_thinking"] = self._provider.enable_thinking

        return payload

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload = self._build_payload(
            messages, temperature=temperature, max_tokens=max_tokens
        )
        timeout = (self._provider.connect_timeout, self._provider.timeout)

        response = requests.post(
            self._provider.base_url,
            headers=self._build_headers(),
            json=payload,
            timeout=timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"LLM 请求失败 (HTTP {response.status_code}): {response.text}"
            )

        if self._provider.stream:
            raise NotImplementedError(
                "http_chat_completions 流式模式尚未在 Agent CLI 中支持，"
                "请将 provider.stream 设为 false。"
            )

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM 返回异常（无 choices）: {data}")

        message = choices[0].get("message") or {}
        content = message.get("content")
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
    """根据 provider.type 创建 LLM 客户端。"""
    if use_mock:
        return MockLLMClient(delay=mock_delay, provider_name=provider.name)

    if provider.type == "openai_compatible":
        return OpenAICompatibleClient(provider)

    if provider.type == "http_chat_completions":
        return RequestsChatCompletionsClient(provider)

    raise ValueError(
        f"不支持的 provider 类型: '{provider.type}'（provider={provider.name}）。"
        f"支持: openai_compatible, http_chat_completions"
    )


QwenClient = OpenAICompatibleClient
