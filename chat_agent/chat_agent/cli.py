"""
终端交互模块

提供 REPL 风格的多轮对话界面，并支持斜杠命令（/help、/exit 等）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from chat_agent.agent import ChatAgent


@dataclass
class CommandResult:
    """命令执行结果。"""

    handled: bool  # True 表示已处理，不再调用 LLM
    message: str = ""  # 展示给用户的信息
    should_exit: bool = False  # True 表示退出程序


HELP_TEXT = """
可用命令（以 / 开头）：
  /help              显示本帮助
  /exit, /quit       退出程序
  /new, /reset       开启新对话（清空历史，保留 system prompt）
  /history           查看当前会话历史摘要
  /system            显示当前 system prompt
  /reload            从配置文件/文件重新加载 system prompt
  /prompt <文本>     临时设置 system prompt（仅当前会话，不写入文件）

直接输入文字即可与 Agent 对话；空行会被忽略。
""".strip()


class ChatCLI:
    """终端对话界面。"""
    # 可以把它理解成 REPL（Read-Eval-Print Loop）：读取用户输入，执行命令，打印结果。
    def __init__(self, agent: ChatAgent) -> None:
        self.agent = agent
        # 一个字典，键是命令名字符串，值是「接收 list[str]、返回 CommandResult 的函数」。
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
                user_input = input("\n你> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n再见！")
                break

            if not user_input:
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

    def _print_banner(self) -> None:
        provider = self.agent.config.get_active_provider()
        if self.agent.is_mock_mode:
            mode = f"Mock 模式（模拟 {provider.name} / {provider.model}）"
        else:
            mode = f"服务: {provider.name} | 模型: {provider.model}"
        print("=" * 56)
        print("  Chat Agent — 最小多轮对话 Agent")
        print(f"  {mode}")
        print("  输入 /help 查看命令，/exit 退出")
        print("=" * 56)
        if self.agent.memory_file_path:
            print(f"  记忆文件: {self.agent.memory_file_path}")
        if self.agent.config.agent.show_system_prompt_on_start:
            print("\n[System Prompt]\n")
            print(self.agent.memory.system_prompt)

    def _handle_command(self, raw: str) -> CommandResult:
        """解析并执行斜杠命令。"""
        parts = raw[1:].strip().split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        args_str = parts[1] if len(parts) > 1 else ""
        args = args_str.split() if args_str else []

        if cmd == "prompt" and args_str:
            args = [args_str]

        handler = self._commands.get(cmd)
        if handler is None:
            return CommandResult(
                handled=True,
                message=f"未知命令: /{cmd}\n输入 /help 查看可用命令。",
            )
        return handler(args)

    def _cmd_help(self, _args: list[str]) -> CommandResult:
        return CommandResult(handled=True, message=HELP_TEXT)

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
            prompt = self.agent.reload_system_prompt()
        except FileNotFoundError as exc:
            return CommandResult(handled=True, message=str(exc))
        preview = prompt[:200] + ("..." if len(prompt) > 200 else "")
        return CommandResult(
            handled=True,
            message=f"已从配置/文件重新加载 system prompt：\n{preview}",
        )

    def _cmd_prompt(self, args: list[str]) -> CommandResult:
        if not args:
            return CommandResult(
                handled=True,
                message="用法: /prompt <新的 system prompt 文本>",
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
