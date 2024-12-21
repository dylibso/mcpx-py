# mcpx-client

A command line utility for working with `mcpx`

- List tools
- Execute tools directly
- Use tools with Claude, Ollama and ChatGPT
  - Llamafile is also supported using the `openai` provider, see https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#json-api-quickstart

## Dependencies

- `npx`
- `mcpx` should be in your `$PATH`
- `uv`
- `ollama` (optional)

## Installation

```sh
uv tool install git+https://github.com/dylibso/mcpx-client
```

Or from the root of the repo:

```sh
uv tool install .
```

### uvx

mcpx-client can also be executed without being installed using `uvx`:

```sh
uvx --from git+https://github.com/dylibso/mcpx-client mcpx-client
```

## Running

### Environment variables

- `ANTHROPIC_API_KEY`: used to configure API key when using the `claude` provider
- `OPENAI_API_KEY`: used to configure API key when using the `openai` provider
- `XTP_APP_ID`: XTP app ID
- `XTP_TOKEN`: XTP auth token
- `XTP_GUEST_KEY`: XTP guest key
- `XTP_PLUGIN_CACHE_DIR`: XTP plugin cache dir
- `MCPX_PATH`: optional, specify an alternative `mcpx` executable

### Get usage/help 

```sh
mcpx-client --help
```

### Chat with an LLM

```sh
mcpx-client chat
```

### List tools

```sh
mcpx-client list
```

### Call a tool

```sh
mcpx-client tool eval_js '{"code": "2+2"}'
```
