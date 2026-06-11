"""
模型回复解析

Qwen thinking 模式下，原始回复通常包含：
  <think>...</think>
  <answer>...</answer>  （或直接跟正文）

对话记忆文件保存完整信息；拼接上下文与展示给用户时仅使用正文答案。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

THINKING_PATTERN = re.compile(
    r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE
)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


@dataclass
class ParsedAssistantResponse:
    """解析后的助手回复。"""

    raw: str
    thinking: str | None
    answer: str

    @property
    def has_thinking(self) -> bool:
        return bool(self.thinking)


def parse_assistant_response(raw: str) -> ParsedAssistantResponse:
    """
    从模型原始输出中分离思考过程与正文答案。

    规则：
      1. thinking：提取 <think> 标签内文本
      2. answer：优先提取 <answer> 标签；否则去掉 thinking 段落后的剩余文本
      3. 若均无标签，整段视为 answer
    """
    text = (raw or "").strip()
    if not text:
        return ParsedAssistantResponse(raw=raw or "", thinking=None, answer="")

    thinking_match = THINKING_PATTERN.search(text)
    thinking = thinking_match.group(1).strip() if thinking_match else None

    answer_matches = ANSWER_PATTERN.findall(text)
    if answer_matches:
        answer = answer_matches[-1].strip()
    else:
        answer = THINKING_PATTERN.sub("", text).strip()

    if not answer:
        answer = text

    return ParsedAssistantResponse(raw=text, thinking=thinking, answer=answer)
