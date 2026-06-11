"""
对话记忆模块

管理多轮对话的消息历史。当前实现为最简单的「全量拼接」策略：
将 system prompt + 全部 user/assistant 历史直接送入模型上下文。

assistant 消息在内存中可含 thinking / raw_content 字段（供持久化），
build_context 与 /history 展示时仅使用 content（正文答案，不含思考过程）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from chat_agent.llm.client import Message

Role = Literal["system", "user", "assistant"]

# 内存中的消息；assistant 可含 thinking、raw_content 扩展字段
StoredMessage = dict[str, Any]


@dataclass
class ConversationMemory:
    """
    会话记忆容器。

    Attributes:
        system_prompt: 系统提示词，始终作为第一条 system 消息
        messages: 用户与助手的历史消息（不含 system）
    """

    system_prompt: str
    messages: list[StoredMessage] = field(default_factory=list)
    _memory_store: Any = field(default=None, repr=False)

    def set_memory_store(self, store: Any) -> None:
        self._memory_store = store

    def _persist(self) -> None:
        if self._memory_store is not None:
            self._memory_store.save(self)

    def add_user_message(self, content: str) -> None:
        """追加一条用户消息。"""
        text = content.strip()
        if not text:
            return
        self.messages.append({"role": "user", "content": text})
        self._persist()

    def add_assistant_message(
        self,
        content: str,
        *,
        thinking: str | None = None,
        raw_content: str | None = None,
    ) -> None:
        """
        追加一条助手回复。

        Args:
            content: 正文答案（用于上下文拼接与用户展示）
            thinking: 思考过程（仅持久化，不进入上下文）
            raw_content: 模型原始输出（持久化备份）
        """
        text = content.strip()
        if not text and not thinking and not raw_content:
            return

        msg: StoredMessage = {"role": "assistant", "content": text}
        if thinking:
            msg["thinking"] = thinking
        if raw_content:
            msg["raw_content"] = raw_content
        self.messages.append(msg)
        self._persist()

    def build_context(self) -> list[Message]:
        """
        构建送入大模型的完整消息列表。

        仅使用 content 字段，不含 thinking。
        格式：[system] + [user, assistant, user, assistant, ...]
        """
        context: list[Message] = [
            {"role": "system", "content": self.system_prompt}
        ]
        for msg in self.messages:
            context.append(
                {"role": msg["role"], "content": msg.get("content", "")}
            )
        return context

    def clear(self) -> None:
        """清空对话历史（保留 system prompt）。"""
        self.messages.clear()
        self._persist()

    def update_system_prompt(self, new_prompt: str) -> None:
        """更新 system prompt。"""
        self.system_prompt = new_prompt.strip()
        self._persist()

    @property
    def turn_count(self) -> int:
        """当前会话的用户发言轮数。"""
        return sum(1 for m in self.messages if m.get("role") == "user")

    def format_history_summary(self, max_turns: int = 10) -> str:
        """格式化历史摘要（助手部分仅展示正文答案）。"""
        if not self.messages:
            return "（当前无对话历史）"

        lines: list[str] = []
        i = 0
        pairs: list[tuple[str, str]] = []
        while i < len(self.messages):
            msg = self.messages[i]
            if msg.get("role") == "user":
                user_text = msg.get("content", "")
                assistant_text = ""
                if i + 1 < len(self.messages) and self.messages[i + 1].get("role") == "assistant":
                    assistant_text = self.messages[i + 1].get("content", "")
                    i += 2
                else:
                    i += 1
                pairs.append((user_text, assistant_text))
            else:
                i += 1

        recent = pairs[-max_turns:]
        start_idx = len(pairs) - len(recent) + 1
        for idx, (user_text, assistant_text) in enumerate(recent, start=start_idx):
            lines.append(f"--- 第 {idx} 轮 ---")
            lines.append(f"用户: {user_text}")
            if assistant_text:
                lines.append(f"助手: {assistant_text}")
            lines.append("")

        if len(pairs) > max_turns:
            lines.insert(0, f"（仅显示最近 {max_turns} 轮，共 {len(pairs)} 轮）\n")

        return "\n".join(lines).rstrip()
