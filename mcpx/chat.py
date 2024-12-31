from ollama import ChatResponse as OllamaChatResponse, Client as OllamaClient
from anthropic import AsyncAnthropic
from openai import OpenAI as OpenAIClient

import json
from dataclasses import dataclass
from typing import Optional, List, Iterator
import tempfile

from .client import Client, Tool

SYSTEM_PROMPT = """
- when evaluating a javascript function code don't print the result to stdout
  instead just call the generated javascript function on its own since the code
  will be executed using eval
- Do not come up with directions or indications.
- Always use the provided functions when applicable, and share the results of
  tool calls with the user
- Invoke the tools upon requests you cannot fulfill on your own
  and parse the responses
- Do not invoke the same tool multiple times in a row with the same
  arguments
- Always try to provide a well formatted, itemized summary
"""


@dataclass
class ChatConfig:
    """
    Stores configuration and session for chats
    """

    model: str
    client: Client = Client()
    url: Optional[str] = None
    system: str = SYSTEM_PROMPT
    format: Optional[str] = None
    debug: bool = False


@dataclass
class ChatResponse:
    role: str
    content: str
    kind: str = "text"
    tool: str | None = None


class ChatProvider:
    """
    Defines the interface for all chat provider implementations
    """

    messages: List[dict]
    tools: List[dict]

    def __init__(self, config: ChatConfig, print=print):
        self.messages = []
        self.tools = []
        self.config = config
        self.print = print

    def _convert_tool(self, tool: Tool):
        """
        Convert a tool from Tool to the expected format
        for the given provider
        """
        return tool

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
        provider.print(">>", f"Calling tool: {name}")
    try:
        res = provider.config.client.call(tool=name, input=f)
        for c in res:
            if c.type == "text":
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
                        role="tool", content=tmp.name, tool=name, kind="image"
                    )
    except Exception as exc:
        s = str(exc)
        r = provider.chat(
            f"Encountered an error when calling tool \
                        {name}: {s}",
            tool=name,
        )
        async for res in r:
            yield r


class Ollama(ChatProvider):
    """
    Chat using ollama
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

    async def chat(self, msg: str, tool: Optional[str] = None):
        if not hasattr(self, "client"):
            self.client = OllamaClient(host=self.config.url)
        if len(self.messages) == 0:
            self.messages.append({"role": "system", "content": self.config.system})
        self.messages.append(
            {
                "role": "user" if tool is None else "tool",
                "content": msg if tool is None else f"Result of {tool}:\n{msg}",
            }
        )
        if tool is not None:
            yield ChatResponse(role="tool", content=msg, tool=tool)
        response: OllamaChatResponse = self.client.chat(
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
            self.messages.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )
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

    async def chat(self, msg: str, tool: Optional[str] = None):
        if len(self.messages) == 0:
            self.messages.append({"role": "system", "content": self.config.system})
        if not hasattr(self, "client"):
            self.client = OpenAIClient(base_url=self.config.url)
        self.messages.append(
            {
                "role": "user",
                "content": msg
                if tool is None
                else f"Result of {tool} tool call:\n{msg}",
            }
        )
        if tool is not None:
            yield ChatResponse(role="tool", content=msg, tool=tool)
        r = self.client.chat.completions.create(
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
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )
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

    async def chat(self, msg: str, tool: Optional[str] = None):
        if not hasattr(self, "client"):
            self.client = AsyncAnthropic(base_url=self.config.url)
        self.messages.append(
            {
                "role": "user",
                "content": msg
                if tool is None
                else f"Result of {tool} tool call:\n{msg}",
            }
        )
        if tool is not None:
            yield ChatResponse(role="tool", content=msg, tool=tool)
        res = await self.client.messages.create(
            max_tokens=1024,
            messages=self.messages,
            model=self.config.model,
            tools=self.tools,
            system=self.config.system,
        )
        for block in res.content:
            if self.config.debug:
                self.print(block)
            if block.type == "text":
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": block.text,
                    }
                )
                yield ChatResponse(role="assistant", content=block.text)
            elif block.type == "tool_use":
                async for res in handle_tool_call(block.name, block.input, self):
                    yield res
