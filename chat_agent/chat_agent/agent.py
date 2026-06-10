"""
Agent 核心模块

编排「记忆 → 构建上下文 → 调用 LLM → 写回记忆」的最小 Agent 循环。
这是整个项目最核心的逻辑，后续扩展工具调用、RAG 等能力时可在此扩展。
"""

from __future__ import annotations

from chat_agent.config_loader import AppConfig, load_system_prompt
from chat_agent.conversation import ConversationMemory
from chat_agent.llm.client import BaseLLMClient, create_llm_client


class ChatAgent:
    """
    最小对话 Agent。

    职责：
      1. 维护 ConversationMemory（多轮历史）
      2. 将完整上下文发送给 LLM
      3. 将回复写入记忆

    不包含 CLI 逻辑，便于单元测试或在 Web/API 中复用。
    """

    def __init__(
        self,
        config: AppConfig,
        llm_client: BaseLLMClient | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.config = config
        provider = config.get_active_provider()
        self.llm = llm_client or create_llm_client(
            provider,
            use_mock=config.runtime.use_mock,
            mock_delay=config.runtime.mock_delay,
        )
        prompt = system_prompt or load_system_prompt(config)
        self.memory = ConversationMemory(system_prompt=prompt)

    @property
    def is_mock_mode(self) -> bool:
        """当前是否使用 Mock LLM。"""
        from chat_agent.llm.client import MockLLMClient

        return isinstance(self.llm, MockLLMClient)

    def chat(self, user_input: str) -> str:
        """
        处理一轮用户输入，返回助手回复。

        Args:
            user_input: 用户消息文本

        Returns:
            助手回复文本
        """
        self.memory.add_user_message(user_input)
        context = self.memory.build_context()
        reply = self.llm.chat(context)
        self.memory.add_assistant_message(reply)
        return reply

    def new_conversation(self) -> None:
        """开启新对话：清空历史，保留当前 system prompt。"""
        self.memory.clear()

    def reload_system_prompt(self) -> str:
        """
        从配置文件/文件重新加载 system prompt。

        适用于用户修改了 prompts/system_prompt.txt 后想在当前会话生效。
        """
        new_prompt = load_system_prompt(self.config)
        self.memory.update_system_prompt(new_prompt)
        return new_prompt

    def set_system_prompt(self, prompt: str) -> None:
        """运行时直接设置 system prompt（不写入文件）。"""
        self.memory.update_system_prompt(prompt)
