test:
    uv run python3 -m unittest

format:
    uv run ruff format mcpx_py examples

check:
    uv run ruff check mcpx_py examples

chat:
    uv run python3 -m mcpx_py chat
