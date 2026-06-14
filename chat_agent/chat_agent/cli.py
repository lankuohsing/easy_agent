"""
终端交互模块

提供 REPL 风格的多轮对话界面，并支持斜杠命令（/help、/exit 等）。

输入模式（自动检测 VS Code / Cursor 终端）：
  - standard：Enter 发送，Shift+Enter 换行
  - vscode：Enter 换行，Ctrl+G / Ctrl+O 发送
    （VS Code 会拦截 Ctrl+Enter，且无法区分 Enter 与 Shift+Enter）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Literal

from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from chat_agent.agent import ChatAgent

InputMode = Literal["standard", "vscode"]


def _detect_input_mode() -> InputMode:
    """
    检测终端输入模式。

    VS Code / Cursor 集成终端里，Enter 与 Shift+Enter 通常都发送 \\r，
    程序无法区分，因此自动切换为 vscode 模式。
    可用环境变量 CHAT_AGENT_INPUT_MODE=standard|vscode 强制指定。
    """
    forced = os.environ.get("CHAT_AGENT_INPUT_MODE", "").strip().lower()
    if forced in ("standard", "vscode"):
        return forced  # type: ignore[return-value]
    if os.environ.get("TERM_PROGRAM") == "vscode":
        return "vscode"
    return "standard"


def _configure_terminal_sequences(mode: InputMode) -> None:
    """标准终端下 patch Shift+Enter 序列；VS Code 模式不做 patch。"""
    if mode == "vscode":
        return

    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES

    for seq in ("\x1b[27;2;13~", "\x1b[13;2~"):
        ANSI_SEQUENCES[seq] = Keys.ControlJ


def _submit_buffer(event) -> None:
    """缓冲区非空时提交输入。"""
    buf = event.current_buffer
    if buf.text.strip():
        buf.validate_and_handle()


def _insert_newline(event) -> None:
    event.current_buffer.insert_text("\n")


def _build_input_key_bindings(mode: InputMode) -> KeyBindings:
    kb = KeyBindings()

    if mode == "vscode":
        # VS Code 终端：Enter / Shift+Enter 均只能换行；用 Ctrl+G / Ctrl+O 发送
        @kb.add("enter")
        @kb.add("c-m")
        @kb.add("c-j")
        def _vscode_newline(event) -> None:
            _insert_newline(event)

        @kb.add("c-g")
        @kb.add("c-o")
        def _vscode_submit(event) -> None:
            _submit_buffer(event)

    else:
        @kb.add("enter")
        @kb.add("c-m")
        def _submit(event) -> None:
            _submit_buffer(event)

        @kb.add("c-j")
        def _newline(event) -> None:
            _insert_newline(event)

        @kb.add("escape", "enter")
        def _newline_alt(event) -> None:
            _insert_newline(event)

    return kb


INPUT_MODE = _detect_input_mode()
_configure_terminal_sequences(INPUT_MODE)
_INPUT_KEY_BINDINGS = _build_input_key_bindings(INPUT_MODE)


def _input_short_help(mode: InputMode = INPUT_MODE) -> str:
    if mode == "vscode":
        return "Enter 换行 | Ctrl+G / Ctrl+O 发送 | /help 查看命令"
    return "Enter 发送 | Shift+Enter 换行 | /help 查看命令"


def _build_help_text(mode: InputMode = INPUT_MODE) -> str:
    if mode == "vscode":
        input_section = """
输入方式（VS Code / Cursor 终端）：
  Enter / Shift+Enter  换行
  Ctrl+G / Ctrl+O      发送消息
  Ctrl+C / Ctrl+D      退出程序

说明：VS Code 会拦截 Ctrl+Enter，且 Enter 与 Shift+Enter 信号相同，
      无法在 VS Code 终端里做到「Enter 发送、Shift+Enter 换行」。
      请用 Ctrl+G（或 Ctrl+O）发送。
      若在系统 Terminal / iTerm 运行，可设 CHAT_AGENT_INPUT_MODE=standard
      恢复 Enter 发送、Shift+Enter 换行。
""".strip()
    else:
        input_section = """
输入方式：
  Enter              发送消息
  Shift+Enter        换行
  Alt+Enter          换行（备用）
  Ctrl+C / Ctrl+D    退出程序
""".strip()

    return f"""
可用命令（以 / 开头）：
  /help              显示本帮助
  /exit, /quit       退出程序
  /new, /reset       开启新对话（清空历史，保留 system prompt）
  /history           查看当前会话历史摘要
  /system            显示当前 system prompt
  /reload            从配置文件/文件重新加载 system prompt
  /prompt <文本>     临时设置 system prompt（仅当前会话，不写入文件）

{input_section}

直接输入文字即可与 Agent 对话；仅空白内容会被忽略。
""".strip()


@dataclass
class CommandResult:
    """命令执行结果。"""

    handled: bool
    message: str = ""
    should_exit: bool = False


class ChatCLI:
    """终端对话界面。"""

    def __init__(self, agent: ChatAgent) -> None:
        self.agent = agent
        self._commands: dict[str, Callable[[list[str]], CommandResult]] = {
            "help": self._cmd_help,
            "?": self._cmd_help,
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
            "q": self._cmd_exit,
            "new": self._cmd_new,
            "reset": self._cmd_new,
            "history": self._cmd_history,
            "system": self._cmd_system,
            "reload": self._cmd_reload,
            "prompt": self._cmd_prompt,
        }

    def run(self) -> None:
        """启动交互循环。"""
        self._print_banner()
        while True:
            try:
                user_input = self._read_user_input()
            except (EOFError, KeyboardInterrupt):
                print("\n\n再见！")
                break

            if user_input is None:
                continue

            if user_input.startswith("/"):
                result = self._handle_command(user_input)
                if result.message:
                    print(result.message)
                if result.should_exit:
                    print("再见！")
                    break
                continue

            print("\n助手> ", end="", flush=True)
            try:
                reply = self.agent.chat(user_input)
                print(reply)
            except Exception as exc:
                print(f"\n[错误] 调用大模型失败: {exc}")
                if (
                    self.agent.memory.messages
                    and self.agent.memory.messages[-1].get("role") == "user"
                ):
                    self.agent.memory.messages.pop()

    def _read_user_input(self) -> str | None:
        """读取用户输入（按键行为取决于 INPUT_MODE）。"""
        text = prompt(
            HTML("\n<b>你</b>> "),
            multiline=True,
            key_bindings=_INPUT_KEY_BINDINGS,
            prompt_continuation="...> ",
            wrap_lines=True,
        )
        stripped = text.strip()
        return stripped or None

    def _print_banner(self) -> None:
        provider = self.agent.config.get_active_provider()
        if self.agent.is_mock_mode:
            mode = f"Mock 模式（模拟 {provider.name} / {provider.model}）"
        else:
            mode = f"服务: {provider.name} | 模型: {provider.model}"
        print("=" * 56)
        print("  Chat Agent — 最小多轮对话 Agent")
        print(f"  {mode}")
        print(f"  {_input_short_help()}")
        print("=" * 56)
        if self.agent.memory_file_path:
            print(f"  记忆文件: {self.agent.memory_file_path}")
        if self.agent.config.agent.show_system_prompt_on_start:
            print("\n[System Prompt]\n")
            print(self.agent.memory.system_prompt)

    def _handle_command(self, raw: str) -> CommandResult:
        """解析并执行斜杠命令（仅使用首行解析命令名）。"""
        first_line, _, rest = raw.partition("\n")
        parts = first_line[1:].strip().split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        args_str = parts[1] if len(parts) > 1 else ""
        if cmd == "prompt":
            if rest.strip():
                args = [rest.strip()] if not args_str else [args_str + "\n" + rest.strip()]
            elif args_str:
                args = [args_str]
            else:
                args = []
        else:
            args = args_str.split() if args_str else []

        handler = self._commands.get(cmd)
        if handler is None:
            return CommandResult(
                handled=True,
                message=f"未知命令: /{cmd}\n输入 /help 查看可用命令。",
            )
        return handler(args)

    def _cmd_help(self, _args: list[str]) -> CommandResult:
        return CommandResult(handled=True, message=_build_help_text())

    def _cmd_exit(self, _args: list[str]) -> CommandResult:
        return CommandResult(handled=True, should_exit=True)

    def _cmd_new(self, _args: list[str]) -> CommandResult:
        memory_path = self.agent.new_conversation()
        msg = "已开启新对话，历史已清空（system prompt 保持不变）。"
        if memory_path:
            msg += f"\n新记忆文件: {memory_path}"
        return CommandResult(handled=True, message=msg)

    def _cmd_history(self, _args: list[str]) -> CommandResult:
        summary = self.agent.memory.format_history_summary()
        turns = self.agent.memory.turn_count
        header = f"当前会话共 {turns} 轮用户发言：\n"
        return CommandResult(handled=True, message=header + summary)

    def _cmd_system(self, _args: list[str]) -> CommandResult:
        return CommandResult(
            handled=True,
            message=f"[当前 System Prompt]\n\n{self.agent.memory.system_prompt}",
        )

    def _cmd_reload(self, _args: list[str]) -> CommandResult:
        try:
            prompt_text = self.agent.reload_system_prompt()
        except FileNotFoundError as exc:
            return CommandResult(handled=True, message=str(exc))
        preview = prompt_text[:200] + ("..." if len(prompt_text) > 200 else "")
        return CommandResult(
            handled=True,
            message=f"已从配置/文件重新加载 system prompt：\n{preview}",
        )

    def _cmd_prompt(self, args: list[str]) -> CommandResult:
        if not args:
            return CommandResult(
                handled=True,
                message="用法: /prompt <新的 system prompt 文本>（支持多行输入）",
            )
        new_prompt = args[0] if len(args) == 1 else " ".join(args)
        self.agent.set_system_prompt(new_prompt)
        return CommandResult(
            handled=True,
            message="已更新 system prompt（仅当前会话有效，未写入文件）。",
        )


def run_cli(agent: ChatAgent) -> None:
    """便捷入口：创建 CLI 并运行。"""
    ChatCLI(agent).run()
