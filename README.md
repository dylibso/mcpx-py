# mcpx-client

A command line client for https://www.mcp.run. This tool enables seamless interaction with various AI models while providing access to a suite of powerful tools.

## Features

### Tool Management
- **List Tools**: Browse available tools and their capabilities
- **Direct Tool Execution**: Run tools with specific inputs without LLM interaction
- **Tool Integration**: Use tools seamlessly within AI chat conversations

### AI Provider Support
- **Claude**: Anthropic's Claude models via API
- **Ollama**: Local open-source LLMs
- **ChatGPT**: OpenAI's GPT models
- **Llamafile**: Local AI using Mozilla's Llamafile (via OpenAI-compatible API)
  - See [Llamafile JSON API setup](https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#json-api-quickstart)

### Interactive Features
- Real-time chat interface with AI models
- Tool suggestion and execution within conversations
- Support for both local and cloud-based AI providers

## Dependencies

- `npx`
- `mcpx` should be in your `$PATH`
- `uv`
- `ollama` (optional)

## Python Usage

### Installation

Using `uv`:

```bash
$ uv add git+https://github.com/dylibso/extism-py
```

Or `pip`:

```bash
$ pip install git+https://github.com/dylibso/extism-py
```

### Example code

```python
from mcpx import Client   # Import the mcp.run client

client = Client()         # Create the client, this will check the
                          # default location for the mcpx config or
                          # the `MCPX_SESSION_ID` env var can be used
                          # to specify a valid mcpx session id

# Call a tool with the given input
results = client.call("eval-js", {"code": "'Hello, world!'"})

# Iterate over the results
for content in results:
    print(content.text)
```

More examples can be found in the [examples/](https://github.com/dylibso/mcpx-py/tree/main/examples) directory

## Command Line Usage

### Installation

```sh
uv tool install git+https://github.com/dylibso/mcpx-py
```

Or from the root of the repo:

```sh
uv tool install .
```

#### uvx

mcpx-client can also be executed without being installed using `uvx`:

```sh
uvx --from git+https://github.com/dylibso/mcpx-py mcpx-client
```

### Configuration

#### Provider Setup

##### Claude
1. Sign up for an Anthropic API account at https://console.anthropic.com
2. Get your API key from the console
3. Set the environment variable: `ANTHROPIC_API_KEY=your_key_here`

##### OpenAI
1. Create an OpenAI account at https://platform.openai.com
2. Generate an API key in your account settings
3. Set the environment variable: `OPENAI_API_KEY=your_key_here`

##### Ollama
1. Install Ollama from https://ollama.ai
2. Pull your desired model: `ollama pull llama3.2`
3. No API key needed - runs locally

##### Llamafile
1. Download a Llamafile model from https://github.com/Mozilla-Ocho/llamafile/releases
2. Make the file executable: `chmod +x your-model.llamafile`
3. Run in JSON API mode: `./your-model.llamafile --json-api --host 127.0.0.1 --port 8080`
4. Use with the OpenAI provider pointing to `http://localhost:8080`

#### MCP Platform Setup
To use MCP tools, you'll need to configure:
- `MCPX_SESSION_ID`: Your mcp.run session ID

### Running

#### Environment variables

- `ANTHROPIC_API_KEY`: used to configure API key when using the `claude` provider
- `OPENAI_API_KEY`: used to configure API key when using the `openai` provider

#### Get usage/help

```sh
mcpx-client --help
```

#### Chat with an LLM

```sh
mcpx-client chat
```

#### List tools

```sh
mcpx-client list
```

#### Call a tool

```sh
mcpx-client tool eval-js '{"code": "2+2"}'
```
