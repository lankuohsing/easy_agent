"""
对话记忆模块

管理多轮对话的消息历史。当前实现为最简单的「全量拼接」策略：
将 system prompt + 全部 user/assistant 历史直接送入模型上下文。

后续可在此模块扩展：滑动窗口、摘要压缩、向量检索等。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from chat_agent.llm.client import Message

Role = Literal["system", "user", "assistant"]


@dataclass
class ConversationMemory:
    """
    会话记忆容器。

    Attributes:
        system_prompt: 系统提示词，始终作为第一条 system 消息
        messages: 用户与助手的历史消息（不含 system）
    """

    system_prompt: str
    messages: list[Message] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        """追加一条用户消息。"""
        text = content.strip()
        if not text:
            return
        self.messages.append({"role": "user", "content": text})

    def add_assistant_message(self, content: str) -> None:
        """追加一条助手回复。"""
        text = content.strip()
        if not text:
            return
        self.messages.append({"role": "assistant", "content": text})

    def build_context(self) -> list[Message]:
        """
        构建送入大模型的完整消息列表。

        格式：[system] + [user, assistant, user, assistant, ...]
        """
        context: list[Message] = [
            {"role": "system", "content": self.system_prompt}
        ]
        context.extend(self.messages)
        return context

    def clear(self) -> None:
        """清空对话历史（保留 system prompt）。"""
        self.messages.clear()

    def update_system_prompt(self, new_prompt: str) -> None:
        """更新 system prompt（通常配合 /system 命令或外部文件修改）。"""
        self.system_prompt = new_prompt.strip()

    @property
    def turn_count(self) -> int:
        """当前会话的用户发言轮数。"""
        return sum(1 for m in self.messages if m.get("role") == "user")

    def format_history_summary(self, max_turns: int = 10) -> str:
        """
        格式化历史摘要，供 /history 命令展示。

        Args:
            max_turns: 最多展示最近几轮（每轮 = user + assistant）
        """
        if not self.messages:
            return "（当前无对话历史）"

        lines: list[str] = []
        # 按 user-assistant 配对展示
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
