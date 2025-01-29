test:
    uv run python3 -m unittest

format:
    uv run ruff format mcpx_py tests examples

check:
    uv run ruff check mcpx_py tests examples
