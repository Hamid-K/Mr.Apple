# Mr.Apple

`Mr.Apple` is a standalone terminal assistant built on top of Apple's `python-apple-fm-sdk`.

It supports:
- Interactive chat with the local Apple model
- `chat` mode and `agent` mode
- Automatic tool calls from the model (shell + filesystem)
- Manual shell execution (`!<command>`)
- Parallel sub-agents via a model tool (`spawn_subagents`)
- Textual TUI with a live status bar

## Requirements

`Mr.Apple` depends on Apple Foundation Models through `apple-fm-sdk`, so host requirements come from Apple's SDK:

- macOS 26.0+
- Xcode 26.0+ (and accepted Xcode/SDK agreement)
- Python 3.10+
- Apple Intelligence enabled on a compatible Mac

Optional but recommended:
- `rg` (ripgrep) for faster `search_files`

## Install

### 1) Install apple-fm-sdk first

If you already have `apple-fm-sdk` installed and importable, skip this section.

```bash
git clone https://github.com/apple/python-apple-fm-sdk
cd python-apple-fm-sdk
uv venv
source .venv/bin/activate
uv sync
uv pip install -e .
```

### 2) Install Mr.Apple

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

### Start directly in agent mode

```bash
mr-apple --mode agent --workspace /path/to/workspace
```

## Commands

- `/help` show help
- `/mode [chat|agent]` switch runtime behavior
- `/stream [on|off]` toggle streamed generation mode
- `/trace [on|off]` show/hide tool-event logs
- `/showcase "prompt"` run the same prompt in chat mode then agent mode
- `/tools` list model-callable tools
- `/cwd [path]` show or set working directory
- `/save [path]` save transcript JSON
- `/reset` start a new model session
- `/exit` quit
- `!<shell command>` execute a shell command directly

## Model Tools

The model can call these tools automatically:
- `run_command`
- `read_file`
- `write_file`
- `list_files`
- `search_files`
- `spawn_subagents`

All tool file paths are restricted to the configured workspace root.

## Session and Context

- Session context is preserved across messages until `/reset`.
- User facts like `my name is ...` are stored in-session and injected into prompts.
- Status bar tracks estimated context usage and marks overflow with `OVERFLOW`.

## Notes

- `stream=off` means assistant output is returned as a single final response.
- `trace=off` hides tool event noise.
- If context overflow occurs, reset with `/reset` or reduce prompt/tool output size.
