from __future__ import annotations

import asyncio
import shlex
from typing import Optional

import apple_fm_sdk as fm

from .core import MrAppleSession, parse_toggle


class MrAppleTerminalApp:
    def __init__(self, runtime: MrAppleSession):
        self.runtime = runtime
        self.runtime.set_status_hooks(
            status_printer=self._status_tick,
            event_sink=self._event_sink,
        )

    def _event_sink(self, message: str) -> None:
        print(message, flush=True)

    def _status_tick(self) -> None:
        if self.runtime.context.trace:
            print(f"status> {self.runtime.status.line()}", flush=True)

    def _print_status(self, force: bool = False) -> None:
        if force or self.runtime.context.trace:
            print(f"status> {self.runtime.status.line()}", flush=True)

    def _print_help(self) -> None:
        print("Commands:")
        print("- /help                  Show this help")
        print("- /mode [chat|agent]     Set interaction mode (default: chat)")
        print("- /trace [on|off]        Toggle progressive tool/mode events")
        print("- /showcase \"prompt\"     Run same prompt in chat and agent modes")
        print("- /tools                 List model-callable tools")
        print("- /stream [on|off]       Toggle streaming responses")
        print("- /cwd [path]            Show or change default working directory")
        print("- /save [path]           Save transcript JSON (default: transcript.json)")
        print("- /reset                 Start a new model session")
        print("- /exit                  Exit terminal")
        print("- !<shell command>       Execute shell command directly")
        print("")
        print("Any other input is sent to the model.")

    async def run(self) -> None:
        await self.runtime.initialize_session()
        await self.runtime.refresh_context_usage()
        print("Mr.Apple CLI")
        print(f"Workspace: {self.runtime.context.workspace_root}")
        print("Type /help for commands. Type /exit to quit.")

        while True:
            try:
                self._print_status(force=True)
                user_input = await asyncio.to_thread(
                    input, f"{self.runtime.context.cwd}> "
                )
            except EOFError:
                print()
                return
            except KeyboardInterrupt:
                print()
                continue

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                should_continue = await self._handle_command(user_input)
                if not should_continue:
                    return
                continue

            if user_input.startswith("!"):
                await self._handle_direct_shell(user_input[1:].strip())
                continue

            await self._handle_model_prompt(user_input)

    async def _handle_model_prompt(self, user_input: str) -> None:
        if self.runtime.context.trace:
            print(f"info> mode={self.runtime.mode}", flush=True)
            print(f"info> {self.runtime.mode_policy()}", flush=True)

        try:
            if self.runtime.stream_mode:
                print("assistant> ", end="", flush=True)
                await self.runtime.respond(
                    user_input,
                    stream_callback=lambda chunk: print(chunk, end="", flush=True),
                )
                print()
            else:
                response = await self.runtime.respond(user_input)
                print(f"assistant> {response}")
        except fm.FoundationModelsError as exc:
            print(f"error> model request failed: {exc}")
        except asyncio.CancelledError:
            print("error> request cancelled")
        except Exception as exc:
            print(f"error> unexpected failure: {exc}")

    async def _handle_direct_shell(self, command: str) -> None:
        if not command:
            print("error> missing shell command after '!'.")
            return

        try:
            result = await self.runtime.run_shell(command)
            if result["stdout"]:
                print(result["stdout"], end="" if result["stdout"].endswith("\n") else "\n")
            if result["stderr"]:
                print(result["stderr"], end="" if result["stderr"].endswith("\n") else "\n")
            print(f"[exit={result['exit_code']} timed_out={result['timed_out']}]")
        except Exception as exc:
            print(f"error> command execution failed: {exc}")

    async def _handle_command(self, command_line: str) -> bool:
        try:
            parts = shlex.split(command_line)
        except ValueError as exc:
            print(f"error> invalid command syntax: {exc}")
            return True

        if not parts:
            return True

        command = parts[0].lower()
        if command in {"/exit", "/quit"}:
            return False
        if command == "/help":
            self._print_help()
            return True
        if command == "/tools":
            print("Model Tools:")
            for tool_line in self.runtime.tool_descriptions():
                print(f"- {tool_line}")
            return True
        if command == "/mode":
            self._command_mode(parts)
            return True
        if command == "/stream":
            self._command_stream(parts)
            return True
        if command == "/trace":
            self._command_trace(parts)
            return True
        if command == "/cwd":
            self._command_cwd(parts)
            return True
        if command in {"/save", "/transcript"}:
            await self._command_save(parts)
            return True
        if command == "/reset":
            await self.runtime.reset_session()
            print("info> started a new session (transcript reset).")
            return True
        if command == "/showcase":
            await self._command_showcase(parts)
            return True

        print(f"error> unknown command '{command}'. Type /help.")
        return True

    def _command_mode(self, parts: list[str]) -> None:
        if len(parts) == 1:
            print(f"info> mode={self.runtime.mode}")
            return

        mode = parts[1].lower()
        if mode not in {"chat", "agent"}:
            print("error> usage: /mode [chat|agent]")
            return

        self.runtime.set_mode(mode)
        print(f"info> mode changed to {self.runtime.mode}")

    def _command_stream(self, parts: list[str]) -> None:
        if len(parts) == 1:
            print(f"info> streaming is {'on' if self.runtime.stream_mode else 'off'}.")
            return

        parsed = parse_toggle(parts[1])
        if parsed is None:
            print("error> usage: /stream [on|off]")
            return

        self.runtime.set_stream_mode(parsed)
        print(f"info> streaming set to {'on' if self.runtime.stream_mode else 'off'}.")

    def _command_trace(self, parts: list[str]) -> None:
        if len(parts) == 1:
            print(f"info> trace={'on' if self.runtime.context.trace else 'off'}")
            return

        parsed = parse_toggle(parts[1])
        if parsed is None:
            print("error> usage: /trace [on|off]")
            return

        self.runtime.set_trace(parsed)
        print(f"info> trace set to {'on' if self.runtime.context.trace else 'off'}")

    def _command_cwd(self, parts: list[str]) -> None:
        if len(parts) == 1:
            print(f"info> cwd={self.runtime.context.cwd}")
            return

        raw_path = parts[1]
        try:
            new_cwd = self.runtime.change_cwd(raw_path)
            print(f"info> cwd changed to {new_cwd}")
        except Exception as exc:
            print(f"error> could not change cwd: {exc}")

    async def _command_save(self, parts: list[str]) -> None:
        raw_path = parts[1] if len(parts) > 1 else "transcript.json"
        try:
            saved_path = await self.runtime.save_transcript(raw_path)
            print(f"info> transcript saved to {saved_path}")
        except Exception as exc:
            print(f"error> could not save transcript: {exc}")

    async def _command_showcase(self, parts: list[str]) -> None:
        if len(parts) < 2:
            print('error> usage: /showcase "prompt"')
            return

        prompt = " ".join(parts[1:])
        original_mode = self.runtime.mode
        original_stream = self.runtime.stream_mode

        print("info> showcase: running prompt in chat mode then agent mode")
        self.runtime.set_stream_mode(True)
        for mode in ("chat", "agent"):
            self.runtime.set_mode(mode)
            await self.runtime.reset_session()
            print(f"----- showcase mode={mode} -----")
            await self._handle_model_prompt(prompt)
            print(f"----- end mode={mode} -----")

        self.runtime.set_mode(original_mode)
        self.runtime.set_stream_mode(original_stream)
        await self.runtime.reset_session()
        print("info> showcase complete")


async def run_cli(runtime: MrAppleSession) -> None:
    app = MrAppleTerminalApp(runtime)
    await app.run()
