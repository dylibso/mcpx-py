from ollama import ChatResponse as OllamaChatResponse, Client as OllamaClient
from anthropic import AsyncAnthropic
from openai import OpenAI as OpenAIClient

import traceback
import json
from dataclasses import dataclass
from typing import Optional, List, Iterator
import tempfile

from .client import Client, Tool


SYSTEM_PROMPT = """
- Do not come up with directions or indications.
- Always use the provided tools/functions when applicable, and share the results of
  tool calls with the user
- Invoke the tools upon requests you cannot fulfill on your own
  and parse the responses
- Always try to provide a well formatted, itemized summary
"""


@dataclass
class ChatConfig:
    """
    Stores configuration and session for chats
    """

    client: Client = Client()
    model: str | None = None
    max_tokens: int = 8192
    url: Optional[str] = None
    system: str = SYSTEM_PROMPT
    format: Optional[str] = None
    provider_client: object | None = None
    debug: bool = False


@dataclass
class ToolResponse:
    tool: str
    input: object
    _error: bool | None = None

    @property
    def is_error(self):
        return self.error or False


@dataclass
class ChatResponse:
    role: str
    content: str
    kind: str = "text"
    tool: ToolResponse | None = None


class ChatProvider:
    """
    Defines the interface for all chat provider implementations
    """

    messages: List[dict]
    tools: List[dict]

    def __init__(self, config: ChatConfig | None = None, print=print):
        self.messages = []
        self.tools = []
        self.config = config or ChatConfig()
        if self.config.model is None:
            self.config.model = self._default_model()
        self.print = print
        if self.config.provider_client is not None:
            self.provider_client = self.config.provider_client

    @staticmethod
    def _default_model() -> str:
        return ""

    def _convert_tool(self, tool: Tool):
        """
        Convert a tool from Tool to the expected format
        for the given provider
        """
        return tool

    def clear_history(self):
        """
        Clear the chat history
        """
        self.messages = []

    def append_message(self, msg: str, role: str = "user", tool: str | None = None):
        """
        Add a new message to the chat
        """
        self.messages.append(
            {
                "role": role,
                "content": msg
                if tool is None
                else f"Result of {tool} tool call:\n{msg}",
            }
        )

    async def chat(
        self, msg: str, tool: Optional[str] = None
    ) -> Iterator[ChatResponse]:
        """
        Handle chat message, if `tool` is set then the message is the result
        of a tool call
        """
        pass

    def get_tools(self):
        """
        Get all tools from the mcp server
        """
        tools = self.config.client.installs
        self.tools = []
        for name, t in tools.items():
            if t is None:
                continue
            for tool in t.tools.values():
                if self.config.debug:
                    self.print("FOUND TOOL:", tool.name)
                self.tools.append(self._convert_tool(tool))
        return self.tools


async def handle_tool_call(name, f, provider) -> Iterator[ChatResponse]:
    if isinstance(f, str):
        f = json.loads(f)
    if provider.config.debug:
        provider.print(">>", f"Calling tool: {name} with input: {f}")
    try:
        res = provider.config.client.call(tool=name, input=f)
        for c in res:
            if c.type == "text":
                yield ChatResponse(
                    role="tool", content=c.text, tool=ToolResponse(tool=name, input=f)
                )
                async for res in provider.chat(c.text, tool=name):
                    yield res
            elif c.type == "image":
                ext = ".jpg"
                if c.mime_type == "image/png":
                    ext = ".png"
                with tempfile.NamedTemporaryFile(
                    suffix=ext, delete=False, delete_on_close=False
                ) as tmp:
                    tmp.write(c._data)
                    yield ChatResponse(
                        role="tool",
                        content=tmp.name,
                        kind="image",
                        tool=ToolResponse(tool=name, input=f),
                    )
    except Exception:
        s = traceback.format_exc()
        yield ChatResponse(
            role="tool", content=s, tool=ToolResponse(tool=name, input=f, _error=True)
        )
        async for res in provider.chat(
            f"Encountered an error when calling tool \
                        {name}: {s}",
            tool=name,
        ):
            yield res


class Ollama(ChatProvider):
    """
    Chat using ollama
    """

    @staticmethod
    def _default_model() -> str:
        return "qwen2.5"

    def _convert_tool(self, tool):
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }

    async def chat(self, msg: str, tool: Optional[str] = None):
        if not hasattr(self, "provider_client"):
            self.provider_client = OllamaClient(host=self.config.url)
        if len(self.messages) == 0:
            self.append_message(self.config.system, role="system")
        self.append_message(msg, role="user" if tool is None else "tool", tool=tool)
        response: OllamaChatResponse = self.provider_client.chat(
            model=self.config.model,
            stream=False,
            tools=self.tools,
            messages=self.messages,
            format=self.config.format,
        )
        if self.config.debug:
            self.print(response)
        content = response.message.content
        if response.message.tool_calls is not None:
            for call in response.message.tool_calls:
                f = call.function.arguments
                async for res in handle_tool_call(call.function.name, f, self):
                    yield res
        if content is not None and content != "":
            self.append_message(content, role="assistant")
            yield ChatResponse(role="assistant", content=content)


class OpenAI(ChatProvider):
    """
    Chat using the OpenAI API
    """

    def _convert_tool(self, tool):
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }

    @staticmethod
    def _default_model() -> str:
        return "gpt-4o"

    async def chat(self, msg: str, tool: Optional[str] = None):
        if not hasattr(self, "provider_client"):
            self.provider_client = OpenAIClient(base_url=self.config.url)
        if len(self.messages) == 0:
            self.append_message(self.config.system, role="system")
        self.append_message(msg, tool=tool)
        r = self.provider_client.chat.completions.create(
            messages=self.messages, model=self.config.model, tools=self.tools
        )
        for response in r.choices:
            if self.config.debug:
                self.print(response)
            content = response.message.content
            if response.message.tool_calls is not None:
                for call in response.message.tool_calls:
                    f = call.function.arguments
                    async for res in handle_tool_call(call.function.name, f, self):
                        yield res
            if content is not None and content != "":
                self.append_message(content, role="assistant")
                yield ChatResponse(role="assistant", content=content)


class Claude(ChatProvider):
    """
    Chat using the Claude API
    """

    def _convert_tool(self, tool):
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }

    @staticmethod
    def _default_model():
        return "claude-3-5-sonnet-latest"

    async def chat(self, msg: str, tool: Optional[str] = None):
        if not hasattr(self, "provider_client"):
            self.provider_client = AsyncAnthropic(base_url=self.config.url)
        self.append_message(msg, tool=tool)
        res = await self.provider_client.messages.create(
            max_tokens=self.config.max_tokens,
            messages=self.messages,
            model=self.config.model,
            tools=self.tools,
            system=self.config.system,
        )
        for block in res.content:
            if self.config.debug:
                self.print(block)
            if block.type == "text":
                self.append_message(block.text, role="assistant")
                yield ChatResponse(role="assistant", content=block.text)
            elif block.type == "tool_use":
                async for res in handle_tool_call(block.name, block.input, self):
                    yield res


class Chat:
    provider: ChatProvider

    def __init__(self, provider=Claude, *args, **kw):
        config = None
        if len(args) > 0 and isinstance(args[0], ChatConfig):
            config = args[0]
        elif "config" in kw:
            config = kw["config"]
        else:
            config = ChatConfig(*args, **kw)
        self.provider = provider(config=config)

    async def chat(self, msg: str, tool: Optional[str] = None):
        self.provider.get_tools()
        async for res in self.provider.chat(msg, tool=tool):
            yield res
