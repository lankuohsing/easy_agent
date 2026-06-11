"""
Agent 核心模块

编排「记忆 → 构建上下文 → 调用 LLM → 写回记忆」的最小 Agent 循环。
"""

from __future__ import annotations

from chat_agent.config_loader import AppConfig, load_system_prompt
from chat_agent.conversation import ConversationMemory
from chat_agent.llm.client import BaseLLMClient, create_llm_client
from chat_agent.memory_store import MemoryStore
from chat_agent.response_parser import parse_assistant_response


class ChatAgent:
    """
    最小对话 Agent。

    职责：
      1. 维护 ConversationMemory（多轮历史）
      2. 将完整上下文发送给 LLM
      3. 解析回复（分离 thinking / answer），写入记忆并持久化
      4. 向调用方返回正文答案（不含思考过程）
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

        self._memory_store: MemoryStore | None = None
        if config.agent.memory.enabled:
            self._memory_store = MemoryStore.create(
                config.agent.memory.storage_dir,
                provider=provider.name,
                model=provider.model,
            )

        self.memory = ConversationMemory(system_prompt=prompt)
        if self._memory_store is not None:
            self.memory.set_memory_store(self._memory_store)
            self._memory_store.save(self.memory)

    @property
    def is_mock_mode(self) -> bool:
        """当前是否使用 Mock LLM。"""
        from chat_agent.llm.client import MockLLMClient

        return isinstance(self.llm, MockLLMClient)

    @property
    def memory_file_path(self) -> str | None:
        """当前会话记忆文件路径（未启用持久化时为 None）。"""
        if self._memory_store is None:
            return None
        return str(self._memory_store.file_path)

    def chat(self, user_input: str) -> str:
        """
        处理一轮用户输入，返回助手正文答案（不含思考过程）。

        Args:
            user_input: 用户消息文本

        Returns:
            助手正文答案
        """
        self.memory.add_user_message(user_input)
        context = self.memory.build_context()
        raw_reply = self.llm.chat(context)
        parsed = parse_assistant_response(raw_reply)
        self.memory.add_assistant_message(
            parsed.answer,
            thinking=parsed.thinking,
            raw_content=parsed.raw,
        )
        return parsed.answer

    def new_conversation(self) -> str | None:
        """
        开启新对话：清空历史，保留 system prompt，并创建新的记忆文件。

        Returns:
            新会话记忆文件路径；未启用持久化时返回 None
        """
        if self._memory_store is not None:
            provider = self.config.get_active_provider()
            self._memory_store.start_new_session(
                provider=provider.name,
                model=provider.model,
            )
        self.memory.clear()
        return self.memory_file_path

    def reload_system_prompt(self) -> str:
        """从配置文件/文件重新加载 system prompt。"""
        new_prompt = load_system_prompt(self.config)
        self.memory.update_system_prompt(new_prompt)
        return new_prompt

    def set_system_prompt(self, prompt: str) -> None:
        """运行时直接设置 system prompt（不写入文件）。"""
        self.memory.update_system_prompt(prompt)
