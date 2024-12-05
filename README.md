# mcpx-client

## Dependencies

- `nodejs`
- `mcpx` should be in your `$PATH`
- `uv`
- `ollama` (optional)

## Running

### Environment variables

- `ANTHROPIC_API_KEY`: used to configure API key when using the `claude` provider
- `XTP_APP_ID`: XTP app ID
- `XTP_TOKEN`: XTP auth token
- `XTP_GUEST_KEY`: XTP guest key
- `XTP_PLUGIN_CACHE_DIR`: XTP plugin cache dir
- `MCPX_PATH`: optional, specify an alternative `mcpx` executable

### Get usage 

```sh
uv run mcpx_client.py --help
```

### Chat with an LLM

```sh
uv run mcpx_client.py chat
```

### List tools

```sh
uv run mcpx_client.py list
```

### Call a tool

```sh
uv run mcpx_client.py tool eval_js '{"code": "2+2"}'
```
