from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

DEFAULT_PROTOCOL_VERSION = "2024-11-05"
STDIO_MODE_CONTENT_LENGTH = "content-length"
STDIO_MODE_NDJSON = "ndjson"


class MCPError(RuntimeError):
    pass


@dataclass
class MCPToolDescriptor:
    server_name: str
    tool_name: str
    full_name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class MCPServerStatus:
    name: str
    connected: bool
    tool_count: int
    error: Optional[str] = None


@dataclass
class _MCPServerConfig:
    name: str
    command: str
    args: list[str]
    env: dict[str, str]
    cwd: Optional[Path]
    stdio_mode: str


def _sanitize_name(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", raw)
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = "tool"
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def _normalize_stdio_mode(raw: Any) -> str:
    text = str(raw or "").strip().lower().replace("_", "-")
    if text in {"ndjson", "jsonl", "json-lines", "newline", "line"}:
        return STDIO_MODE_NDJSON
    if text in {"content-length", "framed", "lsp"}:
        return STDIO_MODE_CONTENT_LENGTH
    return ""


def _looks_like_mcp_remote(command: str, args: list[str]) -> bool:
    candidates = [Path(command).name.lower()]
    candidates.extend(Path(item).name.lower() for item in args)
    for token in candidates:
        if token == "mcp-remote" or token.startswith("mcp-remote@"):
            return True
    return False


def _parse_mcp_config(config_path: Path) -> list[_MCPServerConfig]:
    if not config_path.exists():
        raise MCPError(f"MCP config file not found: {config_path}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    servers = payload.get("mcpServers")
    if not isinstance(servers, dict):
        raise MCPError(
            "Invalid MCP config: expected top-level 'mcpServers' object"
        )

    parsed: list[_MCPServerConfig] = []
    for name, spec in servers.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("disabled") is True:
            continue

        command = str(spec.get("command", "")).strip()
        if not command:
            continue

        args = [str(item) for item in spec.get("args", [])]
        env = {k: str(v) for k, v in spec.get("env", {}).items()}
        configured_stdio_mode = _normalize_stdio_mode(
            spec.get("stdio_mode") or spec.get("stdioMode")
        )
        stdio_mode = configured_stdio_mode or (
            STDIO_MODE_NDJSON
            if _looks_like_mcp_remote(command, args)
            else STDIO_MODE_CONTENT_LENGTH
        )

        cwd: Optional[Path] = None
        if spec.get("cwd"):
            cwd = Path(str(spec["cwd"])).expanduser()

        parsed.append(
            _MCPServerConfig(
                name=str(name),
                command=os.path.expandvars(command),
                args=[os.path.expandvars(item) for item in args],
                env={key: os.path.expandvars(value) for key, value in env.items()},
                cwd=cwd,
                stdio_mode=stdio_mode,
            )
        )

    return parsed


class _MCPStdioClient:
    def __init__(
        self,
        config: _MCPServerConfig,
        *,
        request_timeout_seconds: int,
        event_sink: Optional[Callable[[str], None]] = None,
    ):
        self.config = config
        self.request_timeout_seconds = request_timeout_seconds
        self.event_sink = event_sink

        self.process: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._stderr_task: Optional[asyncio.Task[None]] = None
        self._write_lock = asyncio.Lock()
        self._id_lock = asyncio.Lock()
        self._next_id = 1
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._closed = False

    def _emit(self, message: str) -> None:
        if self.event_sink:
            self.event_sink(message)

    async def start(self) -> None:
        if self.process is not None:
            return

        env = os.environ.copy()
        env.update(self.config.env)
        self.process = await asyncio.create_subprocess_exec(
            self.config.command,
            *self.config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.config.cwd) if self.config.cwd else None,
            env=env,
        )

        if self.process.stdin is None or self.process.stdout is None:
            raise MCPError(f"Failed to initialize stdio streams for server {self.config.name}")

        self._reader_task = asyncio.create_task(self._read_loop())
        if self.process.stderr is not None:
            self._stderr_task = asyncio.create_task(self._stderr_loop())

        try:
            await self._initialize()
        except Exception:
            await self.close()
            raise

    async def _initialize(self) -> None:
        result = await self.request(
            "initialize",
            {
                "protocolVersion": DEFAULT_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "mr-apple", "version": "0.2.0"},
            },
        )
        server_info = result.get("serverInfo", {}) if isinstance(result, dict) else {}
        self._emit(
            f"mcp> connected server={self.config.name} "
            f"info={server_info.get('name', self.config.name)}"
        )
        await self.notify("notifications/initialized", {})

    async def _stderr_loop(self) -> None:
        assert self.process is not None
        assert self.process.stderr is not None
        try:
            while True:
                line = await self.process.stderr.readline()
                if not line:
                    return
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    self._emit(f"mcp[{self.config.name}] stderr> {text}")
        except asyncio.CancelledError:
            return

    async def _read_loop(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None

        try:
            while True:
                message = await self._read_message()
                if message is None:
                    return
                await self._handle_message(message)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._emit(f"mcp> read loop failed server={self.config.name} error={exc}")
            self._fail_pending(exc)

    async def _read_message(self) -> Optional[dict[str, Any]]:
        if self.config.stdio_mode == STDIO_MODE_NDJSON:
            return await self._read_ndjson_message()
        return await self._read_content_length_message()

    async def _read_ndjson_message(self) -> Optional[dict[str, Any]]:
        assert self.process is not None
        assert self.process.stdout is not None
        reader = self.process.stdout

        while True:
            line = await reader.readline()
            if line == b"":
                return None
            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded:
                continue
            parsed = json.loads(decoded)
            if not isinstance(parsed, dict):
                raise MCPError("Expected JSON-RPC object")
            return parsed

    async def _read_content_length_message(self) -> Optional[dict[str, Any]]:
        assert self.process is not None
        assert self.process.stdout is not None
        reader = self.process.stdout

        headers: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if line == b"":
                return None
            if line in {b"\r\n", b"\n"}:
                break
            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded or ":" not in decoded:
                continue
            key, value = decoded.split(":", 1)
            headers[key.strip().lower()] = value.strip()

        content_length = int(headers.get("content-length", "0"))
        if content_length <= 0:
            raise MCPError(f"Invalid content-length from server {self.config.name}")

        body = await reader.readexactly(content_length)
        parsed = json.loads(body.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise MCPError("Expected JSON-RPC object")
        return parsed

    async def _handle_message(self, message: dict[str, Any]) -> None:
        if "id" in message and ("result" in message or "error" in message):
            response_id = message.get("id")
            if not isinstance(response_id, int):
                return
            future = self._pending.get(response_id)
            if not future or future.done():
                return
            if "error" in message:
                future.set_exception(MCPError(self._format_error(message["error"])))
            else:
                future.set_result(message.get("result"))
            return

        method = message.get("method")
        if not isinstance(method, str):
            return

        if "id" in message:
            request_id = message.get("id")
            if method == "ping":
                await self._send_message({"jsonrpc": "2.0", "id": request_id, "result": {}})
            else:
                await self._send_message(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Method not supported by client: {method}",
                        },
                    }
                )

    def _format_error(self, error: Any) -> str:
        if isinstance(error, dict):
            code = error.get("code")
            message = error.get("message")
            data = error.get("data")
            if data is not None:
                return f"{message} (code={code}, data={data})"
            return f"{message} (code={code})"
        return str(error)

    def _fail_pending(self, exc: Exception) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(exc)
        self._pending.clear()

    async def _next_request_id(self) -> int:
        async with self._id_lock:
            current = self._next_id
            self._next_id += 1
            return current

    async def _send_message(self, payload: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise MCPError(f"Server {self.config.name} is not running")

        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        if self.config.stdio_mode == STDIO_MODE_NDJSON:
            frame = data + b"\n"
        else:
            frame = f"Content-Length: {len(data)}\r\n\r\n".encode("ascii") + data
        async with self._write_lock:
            self.process.stdin.write(frame)
            await self.process.stdin.drain()

    async def notify(self, method: str, params: Optional[dict[str, Any]] = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        await self._send_message(payload)

    async def request(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
    ) -> Any:
        request_id = await self._next_request_id()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future

        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        try:
            await self._send_message(payload)
            return await asyncio.wait_for(
                future,
                timeout=self.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise MCPError(
                f"Timed out waiting for '{method}' response from server "
                f"{self.config.name} after {self.request_timeout_seconds}s"
            ) from exc
        finally:
            self._pending.pop(request_id, None)

    async def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor: Optional[str] = None

        while True:
            params: dict[str, Any] = {}
            if cursor:
                params["cursor"] = cursor

            result = await self.request("tools/list", params)
            if not isinstance(result, dict):
                raise MCPError(f"Invalid tools/list result from server {self.config.name}")

            for item in result.get("tools", []):
                if isinstance(item, dict):
                    tools.append(item)

            next_cursor = result.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                break
            cursor = next_cursor

        return tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self.request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments,
            },
        )
        if not isinstance(result, dict):
            raise MCPError(f"Invalid tools/call result for {tool_name}")
        return result

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except BaseException:
                pass
            self._reader_task = None

        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except BaseException:
                pass
            self._stderr_task = None

        process = self.process
        self.process = None
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()


class MCPManager:
    def __init__(
        self,
        config_path: Path,
        *,
        request_timeout_seconds: int = 60,
        event_sink: Optional[Callable[[str], None]] = None,
    ):
        self.config_path = config_path
        self.request_timeout_seconds = request_timeout_seconds
        self.event_sink = event_sink

        self._clients: dict[str, _MCPStdioClient] = {}
        self._tool_map: dict[str, tuple[_MCPStdioClient, MCPToolDescriptor]] = {}
        self._server_status: dict[str, MCPServerStatus] = {}

    def _emit(self, message: str) -> None:
        if self.event_sink:
            self.event_sink(message)

    @property
    def connected(self) -> bool:
        return bool(self._clients)

    def server_statuses(self) -> list[MCPServerStatus]:
        return list(self._server_status.values())

    def tool_descriptors(self) -> list[MCPToolDescriptor]:
        return [item[1] for item in self._tool_map.values()]

    async def start(self) -> None:
        await self.close()

        configs = _parse_mcp_config(self.config_path)
        if not configs:
            self._emit("mcp> no enabled servers in config")
            return

        taken_names: set[str] = set()

        for config in configs:
            client = _MCPStdioClient(
                config,
                request_timeout_seconds=self.request_timeout_seconds,
                event_sink=self.event_sink,
            )
            try:
                await client.start()
                raw_tools = await client.list_tools()

                self._clients[config.name] = client
                self._server_status[config.name] = MCPServerStatus(
                    name=config.name,
                    connected=True,
                    tool_count=len(raw_tools),
                    error=None,
                )

                for raw_tool in raw_tools:
                    tool_name = str(raw_tool.get("name", "")).strip()
                    if not tool_name:
                        continue
                    description = str(raw_tool.get("description", "")).strip()
                    input_schema = raw_tool.get("inputSchema") or raw_tool.get(
                        "input_schema"
                    )
                    if not isinstance(input_schema, dict):
                        input_schema = {}

                    base = (
                        f"mcp_{_sanitize_name(config.name)}__"
                        f"{_sanitize_name(tool_name)}"
                    )
                    full_name = base
                    suffix = 2
                    while full_name in taken_names:
                        full_name = f"{base}_{suffix}"
                        suffix += 1
                    taken_names.add(full_name)

                    descriptor = MCPToolDescriptor(
                        server_name=config.name,
                        tool_name=tool_name,
                        full_name=full_name,
                        description=description,
                        input_schema=input_schema,
                    )
                    self._tool_map[full_name] = (client, descriptor)
            except Exception as exc:
                self._server_status[config.name] = MCPServerStatus(
                    name=config.name,
                    connected=False,
                    tool_count=0,
                    error=str(exc),
                )
                self._emit(f"mcp> failed server={config.name} error={exc}")
                await client.close()

    async def reload(self) -> None:
        await self.start()

    async def call_tool(self, full_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        entry = self._tool_map.get(full_name)
        if not entry:
            raise MCPError(f"Unknown MCP tool: {full_name}")

        client, descriptor = entry
        return await client.call_tool(descriptor.tool_name, arguments)

    async def close(self) -> None:
        clients = list(self._clients.values())
        self._clients = {}
        self._tool_map = {}
        self._server_status = {}

        for client in clients:
            await client.close()
