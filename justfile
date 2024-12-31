test:
    uv run python3 -m unittest

format:
    uv run ruff format mcpx tests examples

check:
    uv run ruff check mcpx tests examples
