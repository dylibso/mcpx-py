# mcpx-py
[![PyPI](https://img.shields.io/pypi/v/mcpx-py)](https://pypi.org/project/mcpx-py/)

A Python library for interacting with LLMs using mcp.run tools

## Features

### AI Provider Support
- **Ollama**: https://ollama.com/
- **Claude**: https://www.anthropic.com/api
- **OpenAI**: https://openai.com/api/
- **Gemini**: https://ai.google.dev/
- **Llamafile**: https://github.com/Mozilla-Ocho/llamafile
  - See [Llamafile JSON API setup](https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#json-api-quickstart)

### Interactive Features
- Real-time chat interface with AI models
- Tool suggestion and execution within conversations
- Support for both local and cloud-based AI providers

## Dependencies

- `uv`
- `npm`
- `ollama` (optional)

## mcp.run Setup

You will need to get an mcp.run session ID by running:

```bash
npx --yes -p @dylibso/mcpx gen-session --write
```

This will generate a new session and write the session ID to a configuration file that can be used
by `mcpx-py`.
 
If you need to store the session ID in  an environment variable you can run `gen-session`
without the `--write` flag:

```bash
npx --yes -p @dylibso/mcpx gen-session
```

which should output something like:

```
Login successful!
Session: kabA7w6qH58H7kKOQ5su4v3bX_CeFn4k.Y4l/s/9dQwkjv9r8t/xZFjsn2fkLzf+tkve89P1vKhQ
```

Then set the `MPC_RUN_SESSION_ID` environment variable:

```
$ export MCP_RUN_SESSION_ID=kabA7w6qH58H7kKOQ5su4v3bX_CeFn4k.Y4l/s/9dQwkjv9r8t/xZFjsn2fkLzf+tkve89P1vKhQ
```

## Python Usage

### Installation

Using `uv`:

```bash
uv add mcpx-py
```

Or `pip`:

```bash
pip install mcpx-py
```

### Example code

```python
from mcpx_py import Chat, Claude

llm = Chat(Claude)

# Or OpenAI
# from mcpx import OpenAI
# llm = Chat(OpenAI)

# Or Ollama
# from mcpx import Ollama
# llm = Chat(Ollama)

# Or Gemini
# from mcpx import Gemini
# llm = Chat(Gemini)

# Prompt claude and iterate over the results
async for response in llm.send_message(
    "summarize the contents of example.com"
):
    print(response)
```

More examples can be found in the [examples/](https://github.com/dylibso/mcpx-py/tree/main/examples) directory

## Command Line Usage

### Installation

```sh
uv tool install mcpx-py
```

From git:

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

### Running

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

### LLM Configuration

#### Provider Setup

##### Claude
1. Sign up for an Anthropic API account at https://console.anthropic.com
2. Get your API key from the console
3. Set the environment variable: `ANTHROPIC_API_KEY=your_key_here`

##### OpenAI
1. Create an OpenAI account at https://platform.openai.com
2. Generate an API key in your account settings
3. Set the environment variable: `OPENAI_API_KEY=your_key_here`

##### Gemini
1. Create an Gemini account at https://aistudio.google.com
2. Generate an API key in your account settings
3. Set the environment variable: `GEMINI_API_KEY=your_key_here`

##### Ollama
1. Install Ollama from https://ollama.ai
2. Pull your desired model: `ollama pull llama3.2`
3. No API key needed - runs locally

##### Llamafile
1. Download a Llamafile model from https://github.com/Mozilla-Ocho/llamafile/releases
2. Make the file executable: `chmod +x your-model.llamafile`
3. Run in JSON API mode: `./your-model.llamafile --json-api --host 127.0.0.1 --port 8080`
4. Use with the OpenAI provider pointing to `http://localhost:8080`
