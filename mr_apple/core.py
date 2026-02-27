from __future__ import annotations

import asyncio
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import apple_fm_sdk as fm
except ImportError as exc:
    raise ImportError(
        "apple_fm_sdk is required. Install apple-fm-sdk first, then install Mr.Apple."
    ) from exc

DEFAULT_INSTRUCTIONS = (
    "You are Mr.Apple, a pragmatic terminal coding assistant running locally on "
    "Apple Foundation Models. "
    "You can call tools to inspect files, modify files, search code, and run shell commands. "
    "When a tool returns an error, explain it and choose the next best action."
)
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_OUTPUT_CHAR_LIMIT = 12_000
DEFAULT_MAX_LIST_ENTRIES = 200
DEFAULT_MAX_SEARCH_MATCHES = 100
DEFAULT_MAX_SUBAGENTS = 4
DEFAULT_CONTEXT_WINDOW_CHARS = 120_000


@dataclass
class RuntimeStats:
    mode: str = "chat"
    stream: bool = False
    turns: int = 0
    tool_calls: int = 0
    last_tool: str = "-"
    context_chars: int = 0
    context_window_chars: int = DEFAULT_CONTEXT_WINDOW_CHARS
    context_overflow_count: int = 0
    context_overflow_active: bool = False
    agent_batches: int = 0
    agent_tasks_total: int = 0
    agent_tasks_running: int = 0
    agent_tasks_done: int = 0
    agent_tasks_failed: int = 0
    last_event: str = "-"


class RuntimeStatus:
    def __init__(self, stats: RuntimeStats):
        self.stats = stats

    def set_mode(self, mode: str) -> None:
        self.stats.mode = mode

    def set_stream(self, enabled: bool) -> None:
        self.stats.stream = enabled

    def note_turn(self) -> None:
        self.stats.turns += 1

    def set_context_chars(self, chars: int) -> None:
        self.stats.context_chars = max(0, chars)
        if self.stats.context_overflow_active and self.stats.context_chars == 0:
            self.stats.context_overflow_active = False

    def note_context_overflow(self) -> None:
        self.stats.context_overflow_count += 1
        self.stats.context_overflow_active = True
        if (
            self.stats.context_chars > 0
            and self.stats.context_window_chars > self.stats.context_chars
        ):
            self.stats.context_window_chars = self.stats.context_chars
        self.stats.last_event = "context:overflow"

    def clear_context_overflow(self) -> None:
        self.stats.context_overflow_active = False

    def note_tool_start(self, name: str) -> None:
        self.stats.tool_calls += 1
        self.stats.last_tool = name
        self.stats.last_event = f"tool:{name}:start"

    def note_tool_end(self, name: str, ok: bool) -> None:
        self.stats.last_tool = name
        self.stats.last_event = f"tool:{name}:{'ok' if ok else 'error'}"

    def start_agent_batch(self, total_tasks: int) -> None:
        self.stats.agent_batches += 1
        self.stats.agent_tasks_total = max(0, total_tasks)
        self.stats.agent_tasks_running = 0
        self.stats.agent_tasks_done = 0
        self.stats.agent_tasks_failed = 0
        self.stats.last_event = f"agents:start:{total_tasks}"

    def note_agent_started(self) -> None:
        self.stats.agent_tasks_running += 1
        self.stats.last_event = "agents:worker:start"

    def note_agent_finished(self, ok: bool) -> None:
        self.stats.agent_tasks_running = max(0, self.stats.agent_tasks_running - 1)
        if ok:
            self.stats.agent_tasks_done += 1
        else:
            self.stats.agent_tasks_failed += 1
        self.stats.last_event = f"agents:worker:{'ok' if ok else 'error'}"

    def note_event(self, event: str) -> None:
        self.stats.last_event = event

    def context_pct(self) -> float:
        if self.stats.context_overflow_active:
            return 100.0
        if self.stats.context_window_chars <= 0:
            return 0.0
        return min(
            100.0,
            (self.stats.context_chars / self.stats.context_window_chars) * 100.0,
        )

    def line(self) -> str:
        overflow_marker = " OVERFLOW" if self.stats.context_overflow_active else ""
        return (
            f"mode={self.stats.mode} stream={'on' if self.stats.stream else 'off'} "
            f"ctx~{self.context_pct():.1f}%{overflow_marker} "
            f"({self.stats.context_chars}/{self.stats.context_window_chars} chars) "
            f"turns={self.stats.turns} tools={self.stats.tool_calls} "
            f"agents(r/d/f/t)={self.stats.agent_tasks_running}/"
            f"{self.stats.agent_tasks_done}/{self.stats.agent_tasks_failed}/"
            f"{self.stats.agent_tasks_total} batches={self.stats.agent_batches}"
        )


@dataclass
class ToolRuntimeContext:
    workspace_root: Path
    cwd: Path
    command_timeout_seconds: int
    output_char_limit: int
    file_read_limit: int
    context_window_chars: int
    trace: bool = False
    status: Optional[RuntimeStatus] = None
    status_printer: Optional[Callable[[], None]] = None
    event_sink: Optional[Callable[[str], None]] = None


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _is_within_workspace(workspace_root: Path, candidate: Path) -> bool:
    return candidate == workspace_root or workspace_root in candidate.parents


def _resolve_workspace_path(
    workspace_root: Path,
    cwd: Path,
    raw_path: Optional[str],
    *,
    default_to_cwd: bool = True,
) -> Path:
    if raw_path:
        candidate = Path(raw_path).expanduser()
    else:
        candidate = cwd if default_to_cwd else workspace_root

    if not candidate.is_absolute():
        candidate = cwd / candidate

    resolved = candidate.resolve()
    if not _is_within_workspace(workspace_root, resolved):
        raise ValueError(
            f"path '{raw_path}' resolves outside workspace root '{workspace_root}'"
        )
    return resolved


def _workspace_display_path(workspace_root: Path, path: Path) -> str:
    if path == workspace_root:
        return "."
    return str(path.relative_to(workspace_root))


async def _run_shell_command(
    command: str,
    cwd: Path,
    timeout_seconds: int,
    output_char_limit: int,
) -> dict[str, Any]:
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    timed_out = False
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        timed_out = True
        process.kill()
        stdout_b, stderr_b = await process.communicate()

    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")
    stdout, stdout_truncated = _truncate(stdout, output_char_limit)
    stderr, stderr_truncated = _truncate(stderr, output_char_limit)

    return {
        "command": command,
        "cwd": str(cwd),
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }


async def _run_exec_command(
    command: list[str],
    cwd: Path,
    timeout_seconds: int,
    output_char_limit: int,
) -> dict[str, Any]:
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    timed_out = False
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        timed_out = True
        process.kill()
        stdout_b, stderr_b = await process.communicate()

    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")
    stdout, stdout_truncated = _truncate(stdout, output_char_limit)
    stderr, stderr_truncated = _truncate(stderr, output_char_limit)

    return {
        "command": command,
        "cwd": str(cwd),
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }


class WorkspaceToolBase(fm.Tool):
    def __init__(self, context: ToolRuntimeContext):
        self.context = context
        super().__init__()

    def resolve_path(
        self, raw_path: Optional[str], *, default_to_cwd: bool = True
    ) -> Path:
        return _resolve_workspace_path(
            self.context.workspace_root,
            self.context.cwd,
            raw_path,
            default_to_cwd=default_to_cwd,
        )

    def as_json(self, payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False)

    def emit(self, message: str) -> None:
        if self.context.trace and self.context.event_sink:
            self.context.event_sink(message)
        if self.context.status:
            self.context.status.note_event(message)
        if self.context.status_printer:
            self.context.status_printer()

    def tool_started(self, tool_name: str) -> None:
        if self.context.status:
            self.context.status.note_tool_start(tool_name)
        if self.context.status_printer:
            self.context.status_printer()

    def tool_finished(self, tool_name: str, ok: bool) -> None:
        if self.context.status:
            self.context.status.note_tool_end(tool_name, ok)
        if self.context.status_printer:
            self.context.status_printer()


@fm.generable("Arguments for running a shell command in the workspace.")
class RunCommandArgs:
    command: str = fm.guide("Shell command to execute.")
    cwd: Optional[str] = fm.guide(
        "Optional working directory, relative to workspace root."
    )
    timeout_seconds: Optional[int] = fm.guide(
        "Optional timeout in seconds.", range=(1, 300)
    )


class RunCommandTool(WorkspaceToolBase):
    name = "run_command"
    description = (
        "Runs a shell command inside the workspace and returns exit code, stdout, and stderr."
    )

    @property
    def arguments_schema(self) -> fm.GenerationSchema:
        return RunCommandArgs.generation_schema()

    async def call(self, args: fm.GeneratedContent) -> str:
        command = args.value(str, for_property="command")
        if not command:
            return self.as_json({"error": "command is required"})
        self.tool_started(self.name)

        requested_cwd = args.value(Optional[str], for_property="cwd")
        timeout_seconds = args.value(Optional[int], for_property="timeout_seconds")
        timeout_seconds = timeout_seconds or self.context.command_timeout_seconds

        try:
            cwd = self.resolve_path(requested_cwd, default_to_cwd=True)
            if not cwd.is_dir():
                self.tool_finished(self.name, ok=False)
                return self.as_json(
                    {"error": f"cwd is not a directory: {requested_cwd}"}
                )
            self.emit(f"tool> run_command start cwd={cwd} command={command}")

            result = await _run_shell_command(
                command=command,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                output_char_limit=self.context.output_char_limit,
            )
            self.emit(
                f"tool> run_command done exit={result['exit_code']} timed_out={result['timed_out']}"
            )
            self.tool_finished(self.name, ok=True)
            return self.as_json(result)
        except Exception as exc:
            self.emit(f"tool> run_command error {exc}")
            self.tool_finished(self.name, ok=False)
            return self.as_json({"error": str(exc)})


@fm.generable("Arguments for reading a text file from the workspace.")
class ReadFileArgs:
    path: str = fm.guide("Path to the file to read.")
    max_chars: Optional[int] = fm.guide(
        "Optional max number of characters to return.", range=(1, 100000)
    )


class ReadFileTool(WorkspaceToolBase):
    name = "read_file"
    description = "Reads a UTF-8 text file from the workspace and returns content."

    @property
    def arguments_schema(self) -> fm.GenerationSchema:
        return ReadFileArgs.generation_schema()

    async def call(self, args: fm.GeneratedContent) -> str:
        raw_path = args.value(str, for_property="path")
        max_chars = args.value(Optional[int], for_property="max_chars")
        max_chars = max_chars or self.context.file_read_limit
        self.tool_started(self.name)

        try:
            target = self.resolve_path(raw_path, default_to_cwd=True)
            if not target.exists():
                self.tool_finished(self.name, ok=False)
                return self.as_json({"error": f"file does not exist: {raw_path}"})
            if not target.is_file():
                self.tool_finished(self.name, ok=False)
                return self.as_json({"error": f"path is not a file: {raw_path}"})
            self.emit(f"tool> read_file start path={target}")

            text = target.read_text(encoding="utf-8", errors="replace")
            text, truncated = _truncate(text, max_chars)
            self.emit(
                f"tool> read_file done path={target} chars={len(text)} truncated={truncated}"
            )
            self.tool_finished(self.name, ok=True)
            return self.as_json(
                {
                    "path": _workspace_display_path(self.context.workspace_root, target),
                    "truncated": truncated,
                    "content": text,
                }
            )
        except Exception as exc:
            self.emit(f"tool> read_file error {exc}")
            self.tool_finished(self.name, ok=False)
            return self.as_json({"error": str(exc)})


@fm.generable("Arguments for writing a text file in the workspace.")
class WriteFileArgs:
    path: str = fm.guide("Path to the file to write.")
    content: str = fm.guide("Text content to write.")
    mode: Optional[str] = fm.guide(
        "Write mode: overwrite or append.", anyOf=["overwrite", "append"]
    )


class WriteFileTool(WorkspaceToolBase):
    name = "write_file"
    description = "Writes UTF-8 text files in the workspace."

    @property
    def arguments_schema(self) -> fm.GenerationSchema:
        return WriteFileArgs.generation_schema()

    async def call(self, args: fm.GeneratedContent) -> str:
        raw_path = args.value(str, for_property="path")
        content = args.value(str, for_property="content")
        mode = args.value(Optional[str], for_property="mode") or "overwrite"
        self.tool_started(self.name)

        try:
            target = self.resolve_path(raw_path, default_to_cwd=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            self.emit(f"tool> write_file start path={target} mode={mode}")

            if mode == "append":
                with target.open("a", encoding="utf-8", errors="replace") as handle:
                    handle.write(content)
            elif mode == "overwrite":
                target.write_text(content, encoding="utf-8", errors="replace")
            else:
                self.tool_finished(self.name, ok=False)
                return self.as_json({"error": "mode must be either overwrite or append"})
            self.emit(
                f"tool> write_file done path={target} chars_written={len(content)}"
            )
            self.tool_finished(self.name, ok=True)

            return self.as_json(
                {
                    "path": _workspace_display_path(self.context.workspace_root, target),
                    "mode": mode,
                    "chars_written": len(content),
                }
            )
        except Exception as exc:
            self.emit(f"tool> write_file error {exc}")
            self.tool_finished(self.name, ok=False)
            return self.as_json({"error": str(exc)})


@fm.generable("Arguments for listing files in the workspace.")
class ListFilesArgs:
    path: Optional[str] = fm.guide(
        "Directory to list. Defaults to current tool working directory."
    )
    recursive: Optional[bool] = fm.guide("Whether to list recursively.")
    max_entries: Optional[int] = fm.guide(
        "Maximum number of entries to return.", range=(1, 1000)
    )


def _entry_kind(path: Path) -> str:
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    if path.is_symlink():
        return "symlink"
    return "other"


class ListFilesTool(WorkspaceToolBase):
    name = "list_files"
    description = "Lists files and directories in the workspace."

    @property
    def arguments_schema(self) -> fm.GenerationSchema:
        return ListFilesArgs.generation_schema()

    async def call(self, args: fm.GeneratedContent) -> str:
        raw_path = args.value(Optional[str], for_property="path")
        recursive = bool(args.value(Optional[bool], for_property="recursive") or False)
        max_entries = args.value(Optional[int], for_property="max_entries")
        max_entries = max_entries or DEFAULT_MAX_LIST_ENTRIES
        self.tool_started(self.name)

        try:
            target = self.resolve_path(raw_path, default_to_cwd=True)
            if not target.exists():
                self.tool_finished(self.name, ok=False)
                return self.as_json({"error": f"directory does not exist: {raw_path}"})
            if not target.is_dir():
                self.tool_finished(self.name, ok=False)
                return self.as_json({"error": f"path is not a directory: {raw_path}"})
            self.emit(
                f"tool> list_files start path={target} recursive={recursive} max_entries={max_entries}"
            )

            entries = []
            truncated = False
            iterator = target.rglob("*") if recursive else target.iterdir()
            for entry in iterator:
                if len(entries) >= max_entries:
                    truncated = True
                    break
                entries.append(
                    {
                        "path": _workspace_display_path(self.context.workspace_root, entry),
                        "kind": _entry_kind(entry),
                    }
                )

            entries.sort(key=lambda item: item["path"])
            self.emit(
                f"tool> list_files done path={target} entries={len(entries)} truncated={truncated}"
            )
            self.tool_finished(self.name, ok=True)
            return self.as_json(
                {
                    "path": _workspace_display_path(self.context.workspace_root, target),
                    "recursive": recursive,
                    "truncated": truncated,
                    "entries": entries,
                }
            )
        except Exception as exc:
            self.emit(f"tool> list_files error {exc}")
            self.tool_finished(self.name, ok=False)
            return self.as_json({"error": str(exc)})


@fm.generable("Arguments for searching text in workspace files.")
class SearchFilesArgs:
    pattern: str = fm.guide("Search pattern (plain text or regex for rg).")
    path: Optional[str] = fm.guide(
        "Path to search from. Defaults to current tool working directory."
    )
    max_matches: Optional[int] = fm.guide(
        "Maximum number of matching lines to return.", range=(1, 1000)
    )


def _display_search_path(workspace_root: Path, raw_path: str) -> str:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        return raw_path
    try:
        resolved = candidate.resolve()
        if _is_within_workspace(workspace_root, resolved):
            return _workspace_display_path(workspace_root, resolved)
    except Exception:
        return raw_path
    return raw_path


class SearchFilesTool(WorkspaceToolBase):
    name = "search_files"
    description = "Searches files for matching lines with ripgrep when available."

    @property
    def arguments_schema(self) -> fm.GenerationSchema:
        return SearchFilesArgs.generation_schema()

    async def call(self, args: fm.GeneratedContent) -> str:
        pattern = args.value(str, for_property="pattern")
        raw_path = args.value(Optional[str], for_property="path")
        max_matches = args.value(Optional[int], for_property="max_matches")
        max_matches = max_matches or DEFAULT_MAX_SEARCH_MATCHES
        self.tool_started(self.name)

        try:
            target = self.resolve_path(raw_path, default_to_cwd=True)
            if not target.exists():
                self.tool_finished(self.name, ok=False)
                return self.as_json({"error": f"path does not exist: {raw_path}"})
            self.emit(
                f"tool> search_files start path={target} pattern={pattern!r} max_matches={max_matches}"
            )

            rg = shutil.which("rg")
            if rg:
                command = [
                    rg,
                    "--line-number",
                    "--with-filename",
                    "--color=never",
                    "--max-count",
                    str(max_matches),
                    pattern,
                    str(target),
                ]
                result = await _run_exec_command(
                    command=command,
                    cwd=self.context.cwd,
                    timeout_seconds=self.context.command_timeout_seconds,
                    output_char_limit=self.context.output_char_limit,
                )

                matches = []
                if result["stdout"]:
                    for raw_line in result["stdout"].splitlines():
                        parts = raw_line.split(":", 2)
                        if len(parts) == 3:
                            file_path, line_no, text = parts
                            try:
                                line_num = int(line_no)
                            except ValueError:
                                line_num = None
                            matches.append(
                                {
                                    "file": _display_search_path(
                                        self.context.workspace_root, file_path
                                    ),
                                    "line": line_num,
                                    "text": text,
                                }
                            )

                if result["exit_code"] not in (0, 1):
                    self.emit(f"tool> search_files error ripgrep_exit={result['exit_code']}")
                    self.tool_finished(self.name, ok=False)
                    return self.as_json(
                        {
                            "error": "ripgrep failed",
                            "details": result,
                        }
                    )
                self.emit(
                    f"tool> search_files done engine=ripgrep matches={len(matches)}"
                )
                self.tool_finished(self.name, ok=True)
                return self.as_json(
                    {
                        "search_root": _workspace_display_path(
                            self.context.workspace_root, target
                        ),
                        "pattern": pattern,
                        "engine": "ripgrep",
                        "matches": matches,
                    }
                )

            file_iter = target.rglob("*") if target.is_dir() else [target]
            matches = []
            for file_path in file_iter:
                if not file_path.is_file():
                    continue
                try:
                    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                        for line_number, line in enumerate(handle, start=1):
                            if pattern in line:
                                matches.append(
                                    {
                                        "file": _workspace_display_path(
                                            self.context.workspace_root, file_path
                                        ),
                                        "line": line_number,
                                        "text": line.rstrip("\n"),
                                    }
                                )
                            if len(matches) >= max_matches:
                                break
                except Exception:
                    continue
                if len(matches) >= max_matches:
                    break

            self.emit(
                f"tool> search_files done engine=python_fallback matches={len(matches)}"
            )
            self.tool_finished(self.name, ok=True)
            return self.as_json(
                {
                    "search_root": _workspace_display_path(
                        self.context.workspace_root, target
                    ),
                    "pattern": pattern,
                    "engine": "python_fallback",
                    "matches": matches,
                }
            )
        except Exception as exc:
            self.emit(f"tool> search_files error {exc}")
            self.tool_finished(self.name, ok=False)
            return self.as_json({"error": str(exc)})


@fm.generable("Arguments for running independent subtasks in parallel with sub-agents.")
class SpawnSubagentsArgs:
    tasks: list[str] = fm.guide(
        "List of independent subtasks. Keep each item self-contained."
    )
    max_agents: Optional[int] = fm.guide(
        "Maximum number of sub-agents to run concurrently.", range=(1, 8)
    )


class SpawnSubagentsTool(WorkspaceToolBase):
    name = "spawn_subagents"
    description = (
        "Runs independent subtasks concurrently using separate model sessions and returns all results."
    )

    def __init__(
        self,
        context: ToolRuntimeContext,
        model: fm.SystemLanguageModel,
        base_instructions: str,
    ):
        self.model = model
        self.base_instructions = base_instructions
        super().__init__(context)

    @property
    def arguments_schema(self) -> fm.GenerationSchema:
        return SpawnSubagentsArgs.generation_schema()

    async def _run_one_subtask(
        self,
        task_index: int,
        task_text: str,
        sem: asyncio.Semaphore,
    ) -> dict[str, Any]:
        async with sem:
            self.emit(f"tool> spawn_subagents worker_start id={task_index + 1}")
            if self.context.status:
                self.context.status.note_agent_started()
            if self.context.status_printer:
                self.context.status_printer()
            try:
                sub_tools: list[fm.Tool] = [
                    RunCommandTool(self.context),
                    ReadFileTool(self.context),
                    WriteFileTool(self.context),
                    ListFilesTool(self.context),
                    SearchFilesTool(self.context),
                ]
                sub_session = fm.LanguageModelSession(
                    instructions=(
                        f"{self.base_instructions}\n\n"
                        "You are a worker sub-agent. Focus only on your assigned task. "
                        "Use tools when needed. Return concise, execution-focused output."
                    ),
                    model=self.model,
                    tools=sub_tools,
                )
                prompt = (
                    "Sub-agent runtime context:\n"
                    f"- workspace_root: {self.context.workspace_root}\n"
                    f"- default_cwd: {self.context.cwd}\n\n"
                    f"Assigned subtask #{task_index + 1}:\n{task_text}\n\n"
                    "Return key findings, outputs, and any errors."
                )
                response = await sub_session.respond(prompt)
                self.emit(f"tool> spawn_subagents worker_done id={task_index + 1} ok=true")
                if self.context.status:
                    self.context.status.note_agent_finished(ok=True)
                if self.context.status_printer:
                    self.context.status_printer()
                return {
                    "task_index": task_index,
                    "task": task_text,
                    "ok": True,
                    "response": response,
                }
            except Exception as exc:
                self.emit(
                    f"tool> spawn_subagents worker_done id={task_index + 1} ok=false error={exc}"
                )
                if self.context.status:
                    self.context.status.note_agent_finished(ok=False)
                if self.context.status_printer:
                    self.context.status_printer()
                return {
                    "task_index": task_index,
                    "task": task_text,
                    "ok": False,
                    "error": str(exc),
                }

    async def call(self, args: fm.GeneratedContent) -> str:
        raw_tasks = args.value(list, for_property="tasks")
        requested_max_agents = args.value(Optional[int], for_property="max_agents")
        self.tool_started(self.name)

        if not isinstance(raw_tasks, list):
            self.tool_finished(self.name, ok=False)
            return self.as_json({"error": "tasks must be a list of strings"})

        tasks = [str(item).strip() for item in raw_tasks if str(item).strip()]
        if not tasks:
            self.tool_finished(self.name, ok=False)
            return self.as_json({"error": "tasks must contain at least one task"})

        max_agents = requested_max_agents or min(DEFAULT_MAX_SUBAGENTS, len(tasks))
        max_agents = max(1, min(max_agents, 8, len(tasks)))
        if self.context.status:
            self.context.status.start_agent_batch(total_tasks=len(tasks))
        if self.context.status_printer:
            self.context.status_printer()

        self.emit(f"tool> spawn_subagents start tasks={len(tasks)} max_agents={max_agents}")
        semaphore = asyncio.Semaphore(max_agents)
        results = await asyncio.gather(
            *[
                self._run_one_subtask(task_index=idx, task_text=task, sem=semaphore)
                for idx, task in enumerate(tasks)
            ]
        )
        self.emit("tool> spawn_subagents done")
        self.tool_finished(self.name, ok=True)

        return self.as_json(
            {
                "task_count": len(tasks),
                "max_agents": max_agents,
                "results": results,
            }
        )


def parse_toggle(value: str) -> Optional[bool]:
    choice = value.strip().lower()
    if choice in {"on", "true", "1", "yes", "y"}:
        return True
    if choice in {"off", "false", "0", "no", "n"}:
        return False
    return None


class MrAppleSession:
    def __init__(
        self,
        context: ToolRuntimeContext,
        instructions: str,
        *,
        mode: str = "chat",
        stream_mode: bool = False,
        status_printer: Optional[Callable[[], None]] = None,
        event_sink: Optional[Callable[[str], None]] = None,
    ):
        self.context = context
        self.instructions = instructions
        self.mode = mode
        self.stream_mode = stream_mode
        self.user_facts: dict[str, str] = {}

        self.status = RuntimeStatus(
            RuntimeStats(
                mode=mode,
                stream=stream_mode,
                context_window_chars=context.context_window_chars,
            )
        )
        self.context.status = self.status
        self.context.status_printer = status_printer
        self.context.event_sink = event_sink

        self.model: Optional[fm.SystemLanguageModel] = None
        self.session: Optional[fm.LanguageModelSession] = None
        self.tools: list[fm.Tool] = []

    def set_status_hooks(
        self,
        *,
        status_printer: Optional[Callable[[], None]] = None,
        event_sink: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.context.status_printer = status_printer
        self.context.event_sink = event_sink

    def set_trace(self, enabled: bool) -> None:
        self.context.trace = enabled

    def set_mode(self, mode: str) -> None:
        if mode not in {"chat", "agent"}:
            raise ValueError("mode must be chat or agent")
        self.mode = mode
        self.status.set_mode(mode)
        if self.context.status_printer:
            self.context.status_printer()

    def set_stream_mode(self, enabled: bool) -> None:
        self.stream_mode = enabled
        self.status.set_stream(enabled)
        if self.context.status_printer:
            self.context.status_printer()

    def _session_instructions(self) -> str:
        return (
            f"{self.instructions}\n\n"
            f"Tool workspace root: {self.context.workspace_root}\n"
            "You have these callable tools:\n"
            "- run_command(command, cwd?, timeout_seconds?) for shell commands\n"
            "- read_file(path, max_chars?) for reading UTF-8 files\n"
            "- write_file(path, content, mode?) for writing/appending UTF-8 files\n"
            "- list_files(path?, recursive?, max_entries?) for file discovery\n"
            "- search_files(pattern, path?, max_matches?) for code/text search\n"
            "- spawn_subagents(tasks, max_agents?) for parallel independent subtasks\n"
            "Do not call tools for simple greetings, chit-chat, or pure conversational memory "
            "questions that can be answered from session context.\n"
            "When tool output has an error, summarize it and propose next action.\n"
            "Keep responses concise and technical."
        )

    def mode_policy(self) -> str:
        if self.mode == "agent":
            return (
                "Current mode: agent. "
                "Use tools proactively, split work into independent subtasks when useful, "
                "use spawn_subagents for parallel progress, and validate results."
            )
        return (
            "Current mode: chat. "
            "Prioritize concise conversational answers, but automatically call tools "
            "whenever shell/filesystem access is needed for accuracy or to fulfill "
            "the request. For greetings, chit-chat, or questions answerable from this "
            "conversation memory, do not call tools."
        )

    def _ingest_user_facts(self, user_prompt: str) -> None:
        text = user_prompt.strip()
        if not text:
            return

        patterns = [
            re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z .'\-]{0,40})", re.IGNORECASE),
            re.compile(r"\bcall me\s+([A-Za-z][A-Za-z .'\-]{0,40})", re.IGNORECASE),
        ]
        for pattern in patterns:
            match = pattern.search(text)
            if not match:
                continue
            name = match.group(1).strip().strip(".,!?")
            if not name:
                continue
            self.user_facts["name"] = name
            break

    def _facts_block(self) -> str:
        if not self.user_facts:
            return ""
        lines = [f"- {k}: {v}" for k, v in self.user_facts.items()]
        return "\n".join(lines)

    def decorate_prompt(self, user_prompt: str) -> str:
        facts = self._facts_block()
        facts_section = f"Known user facts:\n{facts}\n\n" if facts else ""
        return (
            "Runtime context:\n"
            f"- workspace_root: {self.context.workspace_root}\n"
            f"- default_cwd: {self.context.cwd}\n\n"
            f"{self.mode_policy()}\n\n"
            f"{facts_section}"
            "User request:\n"
            f"{user_prompt}"
        )

    async def initialize_session(self) -> None:
        self.model = fm.SystemLanguageModel()
        is_available, reason = self.model.is_available()
        if not is_available:
            raise RuntimeError(f"Foundation model unavailable: {reason}")
        assert self.model is not None

        self.status.clear_context_overflow()
        self.status.set_context_chars(0)

        self.tools = [
            RunCommandTool(self.context),
            ReadFileTool(self.context),
            WriteFileTool(self.context),
            ListFilesTool(self.context),
            SearchFilesTool(self.context),
            SpawnSubagentsTool(self.context, self.model, self.instructions),
        ]
        self.session = fm.LanguageModelSession(
            instructions=self._session_instructions(),
            model=self.model,
            tools=self.tools,
        )
        self.status.set_mode(self.mode)
        self.status.set_stream(self.stream_mode)

    async def refresh_context_usage(self) -> None:
        if not self.session:
            return
        try:
            transcript = await self.session.transcript.to_dict()
            transcript_chars = len(json.dumps(transcript, ensure_ascii=False))
            self.status.set_context_chars(transcript_chars)
        except Exception:
            pass

    async def respond(
        self,
        user_input: str,
        *,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        if not self.session:
            raise RuntimeError("session is not initialized")

        self._ingest_user_facts(user_input)
        prompt = self.decorate_prompt(user_input)

        try:
            if self.stream_mode:
                previous = ""
                full_response = ""
                async for snapshot in self.session.stream_response(prompt):
                    full_response = snapshot
                    if stream_callback:
                        if snapshot.startswith(previous):
                            delta = snapshot[len(previous) :]
                        else:
                            delta = snapshot
                        if delta:
                            stream_callback(delta)
                    previous = snapshot
                response = full_response
            else:
                response = await self.session.respond(prompt)

            self.status.note_turn()
            await self.refresh_context_usage()
            return response
        except fm.FoundationModelsError as exc:
            if isinstance(exc, fm.ExceededContextWindowSizeError):
                self.status.note_context_overflow()
            raise
        except Exception:
            await self.refresh_context_usage()
            raise

    async def run_shell(self, command: str) -> dict[str, Any]:
        return await _run_shell_command(
            command=command,
            cwd=self.context.cwd,
            timeout_seconds=self.context.command_timeout_seconds,
            output_char_limit=self.context.output_char_limit,
        )

    def change_cwd(self, raw_path: str) -> Path:
        new_cwd = _resolve_workspace_path(
            self.context.workspace_root,
            self.context.cwd,
            raw_path,
            default_to_cwd=True,
        )
        if not new_cwd.is_dir():
            raise ValueError(f"not a directory: {raw_path}")
        self.context.cwd = new_cwd
        return new_cwd

    async def save_transcript(self, raw_path: str) -> Path:
        if not self.session:
            raise RuntimeError("session is not initialized")

        output_path = _resolve_workspace_path(
            self.context.workspace_root,
            self.context.cwd,
            raw_path,
            default_to_cwd=True,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        transcript = await self.session.transcript.to_dict()
        output_path.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
        await self.refresh_context_usage()
        return output_path

    async def reset_session(self) -> None:
        await self.initialize_session()
        await self.refresh_context_usage()

    def tool_descriptions(self) -> list[str]:
        return [f"{tool.name}: {tool.description}" for tool in self.tools]


def create_context(
    workspace_root: Path,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    output_char_limit: int = DEFAULT_OUTPUT_CHAR_LIMIT,
    context_window_chars: int = DEFAULT_CONTEXT_WINDOW_CHARS,
    trace: bool = False,
) -> ToolRuntimeContext:
    return ToolRuntimeContext(
        workspace_root=workspace_root,
        cwd=workspace_root,
        command_timeout_seconds=timeout_seconds,
        output_char_limit=output_char_limit,
        file_read_limit=output_char_limit,
        context_window_chars=context_window_chars,
        trace=trace,
    )
