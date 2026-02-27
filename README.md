# Mr.Apple

`Mr.Apple` is a standalone terminal assistant built on top of Apple's `python-apple-fm-sdk`.

It supports:
- Interactive chat with the local Apple model
- `chat` mode and `agent` mode
- Automatic tool calls (shell + filesystem)
- Manual shell execution (`!<command>`)
- Parallel sub-agents (`spawn_subagents`)
- Standard MCP tool integration (`tools/list`, `tools/call` over stdio)
- Named sessions you can save and resume
- Textual TUI + plain CLI
- CLI tab auto-complete for commands and key subcommands

## Requirements

`Mr.Apple` depends on Apple Foundation Models through `apple-fm-sdk`:

- macOS 26.0+
- Xcode 26.0+ (and accepted Xcode/SDK agreement)
- Python 3.10+
- Apple Intelligence enabled on a compatible Mac

Optional but recommended:
- `rg` (ripgrep) for faster `search_files`

## Install

### 1) Install `apple-fm-sdk` first

If already installed and importable, skip this step.

```bash
git clone https://github.com/apple/python-apple-fm-sdk
cd python-apple-fm-sdk
uv venv
source .venv/bin/activate
uv sync
uv pip install -e .
```

### 2) Install `Mr.Apple`

```bash
git clone git@github.com:Hamid-K/Mr.Apple.git
cd Mr.Apple
uv pip install -e .
```

## Usage

### Launch TUI (default)

```bash
mr-apple --workspace /path/to/workspace
```

### Launch plain CLI

```bash
mr-apple --ui cli --workspace /path/to/workspace
```

### Start in agent mode

```bash
mr-apple --mode agent --workspace /path/to/workspace
```

### Start with a named session

```bash
mr-apple --session my-task --workspace /path/to/workspace
```

If `my-task` exists in the session store, it is resumed automatically.

## MCP Support (Standard Protocol)

`Mr.Apple` supports standard MCP stdio servers using a config file with top-level `mcpServers`.

Example `mcp_servers.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/you/work"
      ]
    },
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git"]
    }
  }
}
```

Run with MCP config:

```bash
mr-apple --workspace /path/to/workspace --mcp-config /path/to/mcp_servers.json
```

A template config is available at `examples/mcp_servers.example.json`.

If `--mcp-config` is omitted, `Mr.Apple` auto-loads:

- `<workspace>/.mr_apple/mcp_servers.json` (if present)

## Commands

- `/help` show help
- `/mode [chat|agent]` switch runtime behavior
- `/stream [on|off]` toggle streamed generation mode
- `/trace [on|off]` show/hide tool-event logs
- `/showcase "prompt"` run prompt in chat mode then agent mode
- `/tools` list model-callable tools
- `/cwd [path]` show or set working directory
- `/save [path]` save transcript JSON
- `/reset` start a fresh model session
- `/session list` list saved sessions
- `/session name <name>` set active session name
- `/session save [name]` save named session
- `/session load <name>` resume named session
- `/mcp status` show MCP server status
- `/mcp tools` list loaded MCP tools
- `/mcp reload` reload MCP config and rebuild tool set
- `/exit` quit
- `!<shell command>` execute a shell command directly

## Model Tools

Built-in model tools:
- `run_command`
- `read_file`
- `write_file`
- `list_files`
- `search_files`
- `spawn_subagents`

When MCP is enabled, additional model tools are exposed with names like:

- `mcp_<server>__<tool>`

All built-in filesystem tools are restricted to the configured workspace root.

## Session and Context

- Session context is preserved across prompts until `/reset`.
- Named sessions persist user/assistant history and user facts.
- Status bar tracks estimated context usage and marks overflow with `OVERFLOW`.

## Notes

- `stream=off` returns a single final assistant response.
- `trace=off` hides progressive tool logs.
- If context overflows, use `/reset` or reduce prompt/tool output size.
