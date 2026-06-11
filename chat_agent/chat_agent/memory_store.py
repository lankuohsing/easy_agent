"""
会话记忆持久化

将多轮对话以文本形式写入文件，文件名使用「日期_时间」标识，便于后续扩展记忆管理系统。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from chat_agent.config_loader import PROJECT_ROOT
from chat_agent.conversation import ConversationMemory


def new_session_id(when: datetime | None = None) -> str:
    """生成会话 ID，格式 YYYYMMDD_HHMMSS。"""
    dt = when or datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")


@dataclass
class SessionMetadata:
    """写入记忆文件头部的元信息。"""

    provider: str = ""
    model: str = ""
    session_id: str = ""


class MemoryStore:
    """
    会话记忆文件存储。

    每个会话对应一个文本文件，例如 memories/20260610_143052.txt。
    每次用户/助手消息更新后重写整个文件，保证内容与内存一致。
    """

    def __init__(
        self,
        storage_dir: Path,
        session_id: str,
        metadata: SessionMetadata,
    ) -> None:
        self.storage_dir = storage_dir
        self.session_id = session_id
        self.metadata = metadata
        self.file_path = storage_dir / f"{session_id}.txt"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        storage_dir: str | Path,
        *,
        provider: str = "",
        model: str = "",
        session_id: str | None = None,
    ) -> MemoryStore:
        sid = session_id or new_session_id()
        base = Path(storage_dir)
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        meta = SessionMetadata(provider=provider, model=model, session_id=sid)
        return cls(base, sid, meta)

    def start_new_session(self, *, provider: str = "", model: str = "") -> None:
        """开启新会话文件（/new 命令时调用）。"""
        self.session_id = new_session_id()
        self.metadata = SessionMetadata(
            provider=provider or self.metadata.provider,
            model=model or self.metadata.model,
            session_id=self.session_id,
        )
        self.file_path = self.storage_dir / f"{self.session_id}.txt"

    def save(self, memory: ConversationMemory) -> None:
        """将当前会话记忆写入文本文件。"""
        content = self._format(memory)
        self.file_path.write_text(content, encoding="utf-8")

    def _format(self, memory: ConversationMemory) -> str:
        created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "# Chat Agent 会话记忆",
            f"# session_id: {self.session_id}",
            f"# saved_at: {created}",
            f"# provider: {self.metadata.provider}",
            f"# model: {self.metadata.model}",
            "#",
            "# 说明：",
            "#   [User] 用户输入",
            "#   [Assistant/Thinking] 模型思考过程（如有）",
            "#   [Assistant/Answer] 模型正文答案（拼接上下文时使用此部分）",
            "",
            "[System Prompt]",
            memory.system_prompt,
            "",
        ]

        turn = 0
        i = 0
        while i < len(memory.messages):
            msg = memory.messages[i]
            if msg.get("role") != "user":
                i += 1
                continue

            turn += 1
            lines.append(f"=== Turn {turn} ===")
            lines.append("")
            lines.append("[User]")
            lines.append(msg.get("content", ""))
            lines.append("")

            if i + 1 < len(memory.messages) and memory.messages[i + 1].get("role") == "assistant":
                assistant = memory.messages[i + 1]
                thinking = assistant.get("thinking")
                if thinking:
                    lines.append("[Assistant/Thinking]")
                    lines.append(thinking)
                    lines.append("")

                lines.append("[Assistant/Answer]")
                lines.append(assistant.get("content", ""))
                lines.append("")

                if assistant.get("raw_content"):
                    lines.append("[Assistant/Raw]")
                    lines.append(assistant["raw_content"])
                    lines.append("")

                i += 2
            else:
                i += 1

        return "\n".join(lines).rstrip() + "\n"
