from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

DEFAULT_INSTRUCTIONS = (
    "You are Mr.Apple, a pragmatic terminal coding assistant running locally on "
    "Apple Foundation Models. "
    "You can call tools to inspect files, modify files, search code, and run shell commands. "
    "When a tool returns an error, explain it and choose the next best action."
)
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_OUTPUT_CHAR_LIMIT = 12_000
DEFAULT_CONTEXT_WINDOW_CHARS = 120_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mr.Apple local terminal assistant for apple_fm_sdk."
    )
    parser.add_argument(
        "--ui",
        choices=["tui", "cli"],
        default="tui",
        help="Interface mode: textual TUI (default) or plain CLI.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root for file operations and shell commands.",
    )
    parser.add_argument(
        "--instructions",
        default=DEFAULT_INSTRUCTIONS,
        help="Base system instructions for the assistant.",
    )
    parser.add_argument(
        "--mode",
        choices=["chat", "agent"],
        default="chat",
        help="Initial interaction mode.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming responses by default.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Show progressive tool events by default.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Default timeout (seconds) for command execution.",
    )
    parser.add_argument(
        "--output-limit",
        type=int,
        default=DEFAULT_OUTPUT_CHAR_LIMIT,
        help="Max stdout/stderr/text characters returned by tools.",
    )
    parser.add_argument(
        "--context-window-chars",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW_CHARS,
        help="Approximate context window size used for status percentage.",
    )
    return parser.parse_args()


def _resolve_workspace(path: Path) -> Path:
    workspace_root = path.expanduser().resolve()
    if not workspace_root.exists():
        raise RuntimeError(f"workspace does not exist: {workspace_root}")
    if not workspace_root.is_dir():
        raise RuntimeError(f"workspace is not a directory: {workspace_root}")
    return workspace_root


def main() -> None:
    args = parse_args()
    try:
        from .core import MrAppleSession, create_context
        from .terminal_app import run_cli

        workspace_root = _resolve_workspace(args.workspace)
        context = create_context(
            workspace_root,
            timeout_seconds=args.timeout,
            output_char_limit=args.output_limit,
            context_window_chars=args.context_window_chars,
            trace=args.trace,
        )
        runtime = MrAppleSession(
            context=context,
            instructions=args.instructions,
            mode=args.mode,
            stream_mode=args.stream,
        )

        if args.ui == "cli":
            asyncio.run(run_cli(runtime))
            return

        try:
            from .textual_app import run_textual
        except ImportError as exc:
            raise RuntimeError(
                "Textual UI dependencies are missing. Install with: pip install textual rich"
            ) from exc
        run_textual(runtime)
    except KeyboardInterrupt:
        print()
    except Exception as exc:
        print(f"fatal> {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
