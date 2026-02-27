from __future__ import annotations

import shlex

import apple_fm_sdk as fm
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Input, RichLog, Static

from .core import MrAppleSession, parse_toggle


class MrAppleTextualApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    #status {
        height: 1;
        padding: 0 1;
        background: #1f2937;
        color: #f8fafc;
    }

    #log {
        height: 1fr;
        border: tall #334155;
        margin: 0 1;
    }

    #input {
        height: 3;
        border: solid #334155;
        margin: 0 1 1 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_log", "Clear Log"),
        Binding("tab", "autocomplete", "Autocomplete"),
    ]

    COMMANDS = [
        "/help",
        "/mode",
        "/trace",
        "/showcase",
        "/tools",
        "/stream",
        "/cwd",
        "/save",
        "/transcript",
        "/reset",
        "/session",
        "/mcp",
        "/exit",
        "/quit",
    ]

    def __init__(self, runtime: MrAppleSession) -> None:
        super().__init__()
        self.runtime = runtime
        self.runtime.set_status_hooks(
            status_printer=self._status_tick,
            event_sink=self._event_sink,
        )

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("starting...", id="status")
        yield RichLog(id="log", wrap=True, markup=True)
        yield Input(placeholder="Type prompt, /command, or !shell", id="input")

    async def on_mount(self) -> None:
        await self.runtime.start()
        self._refresh_status_widget()
        self._log_info("Ready. Default mode is chat. Use /help for commands.")
        self.query_one("#input", Input).focus()

    async def on_unmount(self) -> None:
        await self.runtime.shutdown()

    def _status_tick(self) -> None:
        self.call_later(self._refresh_status_widget)

    def _event_sink(self, message: str) -> None:
        self.call_later(self._log_event, message)

    def _refresh_status_widget(self) -> None:
        self.runtime.status.set_mode(self.runtime.mode)
        self.runtime.status.set_stream(self.runtime.stream_mode)
        self.query_one("#status", Static).update(self.runtime.status.line())

    def _log(self, text: str) -> None:
        self.query_one("#log", RichLog).write(text)

    def _log_info(self, text: str) -> None:
        self._log(f"[cyan]info>[/cyan] {text}")

    def _log_error(self, text: str) -> None:
        self._log(f"[red]error>[/red] {text}")

    def _log_event(self, text: str) -> None:
        self._log(f"[yellow]{text}[/yellow]")

    def _suggest_completions(self, line: str) -> list[str]:
        stripped = line.strip()
        if not stripped.startswith("/"):
            return []

        try:
            parts = shlex.split(stripped)
        except Exception:
            parts = stripped.split()

        if not parts:
            return self.COMMANDS

        if len(parts) == 1 and not stripped.endswith(" "):
            return [item for item in self.COMMANDS if item.startswith(parts[0])]

        command = parts[0]
        if command == "/mode":
            return ["chat", "agent"]
        if command in {"/stream", "/trace"}:
            return ["on", "off"]
        if command == "/session":
            if len(parts) <= 2 and not stripped.endswith(" "):
                return ["list", "save", "load", "resume", "name"]
            if len(parts) >= 2 and parts[1] in {"load", "resume"}:
                return self.runtime.list_saved_sessions()
        if command == "/mcp":
            return ["status", "tools", "reload"]
        return []

    def action_autocomplete(self) -> None:
        input_widget = self.query_one("#input", Input)
        value = input_widget.value
        candidates = self._suggest_completions(value)
        if not candidates:
            return

        if value.strip() == "":
            return

        prefix = value.split()[-1] if value.split() else value
        match = next((item for item in candidates if item.startswith(prefix)), None)
        if match is None:
            return

        if " " in value.strip():
            idx = value.rfind(prefix)
            input_widget.value = f"{value[:idx]}{match}"
        else:
            input_widget.value = match
        input_widget.cursor_position = len(input_widget.value)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        event.input.disabled = True
        try:
            await self._handle_input(text)
        finally:
            event.input.disabled = False

    async def _handle_input(self, text: str) -> None:
        if text.startswith("/"):
            await self._handle_command(text)
            return
        if text.startswith("!"):
            await self._handle_shell(text[1:].strip())
            return
        await self._handle_prompt(text)

    async def _handle_prompt(self, text: str) -> None:
        self._log(f"[bold cyan]you>[/bold cyan] {text}")
        try:
            response = await self.runtime.respond(text)
            self._log(f"[bold green]assistant>[/bold green] {response}")
            self._refresh_status_widget()
        except fm.FoundationModelsError as exc:
            self._refresh_status_widget()
            self._log_error(f"model request failed: {exc}")
        except Exception as exc:
            self._refresh_status_widget()
            self._log_error(str(exc))

    async def _handle_shell(self, command: str) -> None:
        if not command:
            self._log_error("missing shell command after '!'.")
            return

        try:
            result = await self.runtime.run_shell(command)
            if result["stdout"]:
                self._log(result["stdout"].rstrip("\n"))
            if result["stderr"]:
                self._log(f"[red]{result['stderr'].rstrip()}[/red]")
            self._log(
                f"[magenta][exit={result['exit_code']} timed_out={result['timed_out']}][/magenta]"
            )
        except Exception as exc:
            self._log_error(f"command execution failed: {exc}")

    async def _handle_command(self, command_line: str) -> None:
        try:
            parts = shlex.split(command_line)
        except ValueError as exc:
            self._log_error(f"invalid command syntax: {exc}")
            return

        if not parts:
            return

        command = parts[0].lower()
        if command in {"/exit", "/quit"}:
            self.exit()
            return

        if command == "/help":
            self._log(
                "Commands: /help /mode [chat|agent] /trace [on|off] /showcase \"prompt\" "
                "/stream [on|off] /tools /cwd [path] /save [path] /reset "
                "/session [list|name|save|load|resume] /mcp [status|tools|reload] "
                "/exit and !<shell>"
            )
            return

        if command == "/tools":
            tools = ", ".join(tool.name for tool in self.runtime.tools)
            self._log(f"tools> {tools}")
            return

        if command == "/mode":
            if len(parts) == 1:
                self._log_info(f"mode={self.runtime.mode}")
                return
            choice = parts[1].lower()
            if choice not in {"chat", "agent"}:
                self._log_error("usage: /mode [chat|agent]")
                return
            self.runtime.set_mode(choice)
            self._log_info(f"mode changed to {self.runtime.mode}")
            self._refresh_status_widget()
            return

        if command == "/stream":
            if len(parts) == 1:
                self._log_info(f"stream={'on' if self.runtime.stream_mode else 'off'}")
                return
            parsed = parse_toggle(parts[1])
            if parsed is None:
                self._log_error("usage: /stream [on|off]")
                return
            self.runtime.set_stream_mode(parsed)
            self._refresh_status_widget()
            self._log_info(f"stream set to {'on' if self.runtime.stream_mode else 'off'}")
            return

        if command == "/trace":
            if len(parts) == 1:
                self._log_info(f"trace={'on' if self.runtime.context.trace else 'off'}")
                return
            parsed = parse_toggle(parts[1])
            if parsed is None:
                self._log_error("usage: /trace [on|off]")
                return
            self.runtime.set_trace(parsed)
            self._log_info(f"trace set to {'on' if self.runtime.context.trace else 'off'}")
            return

        if command == "/cwd":
            if len(parts) == 1:
                self._log_info(f"cwd={self.runtime.context.cwd}")
                return
            raw_path = parts[1]
            try:
                new_cwd = self.runtime.change_cwd(raw_path)
                self._log_info(f"cwd changed to {new_cwd}")
            except Exception as exc:
                self._log_error(str(exc))
            return

        if command in {"/save", "/transcript"}:
            raw_path = parts[1] if len(parts) > 1 else "transcript.json"
            try:
                saved = await self.runtime.save_transcript(raw_path)
                self._log_info(f"transcript saved to {saved}")
            except Exception as exc:
                self._log_error(str(exc))
            return

        if command == "/reset":
            await self.runtime.reset_session(clear_saved_context=True)
            self._log_info("started a new session (context reset)")
            self._refresh_status_widget()
            return

        if command == "/showcase":
            await self._command_showcase(parts)
            return

        if command == "/session":
            await self._command_session(parts)
            return

        if command == "/mcp":
            await self._command_mcp(parts)
            return

        self._log_error(f"unknown command '{command}'. Type /help.")

    async def _command_showcase(self, parts: list[str]) -> None:
        if len(parts) < 2:
            self._log_error('usage: /showcase "prompt"')
            return

        prompt = " ".join(parts[1:])
        original_mode = self.runtime.mode
        original_stream = self.runtime.stream_mode

        self._log_info("showcase: running prompt in chat mode then agent mode")
        self.runtime.set_stream_mode(True)
        for mode in ("chat", "agent"):
            self.runtime.set_mode(mode)
            await self.runtime.reset_session(clear_saved_context=True)
            self._log(f"[magenta]----- showcase mode={mode} -----[/magenta]")
            await self._handle_prompt(prompt)
            self._log(f"[magenta]----- end mode={mode} -----[/magenta]")

        self.runtime.set_mode(original_mode)
        self.runtime.set_stream_mode(original_stream)
        await self.runtime.reset_session(clear_saved_context=True)
        self._refresh_status_widget()
        self._log_info("showcase complete")

    async def _command_session(self, parts: list[str]) -> None:
        if len(parts) == 1:
            self._log_info(f"session {self.runtime.session_summary()}")
            return

        sub = parts[1].lower()
        if sub == "list":
            sessions = self.runtime.list_saved_sessions()
            if not sessions:
                self._log_info("no saved sessions")
                return
            self._log("Saved sessions:")
            for name in sessions:
                self._log(f"- {name}")
            return

        if sub == "name":
            if len(parts) < 3:
                self._log_error("usage: /session name <name>")
                return
            try:
                assigned = self.runtime.set_session_name(parts[2])
                self._log_info(f"active session name set to {assigned}")
            except Exception as exc:
                self._log_error(str(exc))
            return

        if sub == "save":
            name = parts[2] if len(parts) > 2 else None
            try:
                saved = await self.runtime.save_named_session(name)
                self._log_info(f"session saved to {saved}")
            except Exception as exc:
                self._log_error(str(exc))
            return

        if sub in {"load", "resume"}:
            if len(parts) < 3:
                self._log_error("usage: /session load <name>")
                return
            try:
                loaded = await self.runtime.load_named_session(parts[2])
                self._log_info(f"resumed session from {loaded}")
                self._refresh_status_widget()
            except Exception as exc:
                self._log_error(str(exc))
            return

        self._log_error("usage: /session [list|name|save|load|resume]")

    async def _command_mcp(self, parts: list[str]) -> None:
        if len(parts) == 1 or parts[1].lower() == "status":
            for line in self.runtime.mcp_server_status_lines():
                self._log(f"mcp> {line}")
            return

        sub = parts[1].lower()
        if sub == "tools":
            tools = self.runtime.mcp_tool_lines()
            if not tools:
                self._log("mcp> no MCP tools loaded")
                return
            self._log("MCP tools:")
            for line in tools:
                self._log(f"- {line}")
            return

        if sub == "reload":
            try:
                await self.runtime.reload_mcp()
                self._log_info("reloaded MCP servers and reset current model session")
                self._refresh_status_widget()
            except Exception as exc:
                self._log_error(str(exc))
            return

        self._log_error("usage: /mcp [status|tools|reload]")

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()
        self._log_info("log cleared")


def run_textual(runtime: MrAppleSession) -> None:
    app = MrAppleTextualApp(runtime)
    app.run()
