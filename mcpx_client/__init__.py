from ollama import Client as OllamaClient
from ollama import ChatResponse
from anthropic import AsyncAnthropic
from openai import OpenAI as OpenAIClient
from mcp import ClientSession, Tool

import json
from dataclasses import dataclass
from typing import Optional

SYSTEM_PROMPT = """
- when evaluating a javascript function code don't print the result to stdout
  instead just call the generated javascript function on its own since the code
  will be executed using eval
- Do not come up with directions or indications.
- Always use the provided functions when applicable, and share the results of
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

    session: ClientSession
    model: str
    url: Optional[str] = None
    system: str = SYSTEM_PROMPT
    format: Optional[str] = None
    debug: bool = False


class ChatProvider:
    """
    Defines the interface for all chat provider implementations
    """

    messages: list
    tools: list
    model: str

    def __init__(self, config: ChatConfig, print=print):
        self.messages = []
        self.tools = []
        self.config = config
        self.print = print

    def _convert_tool(self, tool: Tool):
        """
        Convert a tool from mcp.Tool to the expected format
        for the given provider
        """
        return tool

    async def chat(self, msg: str, tool: Optional[str] = None):
        """
        Handle chat message, if `tool` is set then the message is the result
        of a tool call
        """
        pass

    async def get_tools(self):
        """
        Get all tools from the mcp server
        """
        tools = await self.config.session.list_tools()
        self.tools = []
        for name, t in tools:
            if t is None:
                continue
            for tool in t:
                if self.config.debug:
                    self.print("FOUND TOOL:", tool.name)
                self.tools.append(self._convert_tool(tool))
        return self.tools


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
                "parameters": tool.inputSchema,
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
        response: ChatResponse = self.client.chat(
            model=self.config.model,
            stream=False,
            tools=self.tools,
            messages=self.messages,
            format=self.config.format,
        )
        if self.config.debug:
            self.print(response)
        content = response.message.content
        if content is not None and content != "":
            self.print(">>", content)
            self.messages.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )
        if response.message.tool_calls is not None:
            for call in response.message.tool_calls:
                f = call.function.arguments
                if isinstance(f, str):
                    f = json.loads(f)
                self.print(">>", f"Calling tool: {call.function.name}")
                try:
                    res = await self.config.session.call_tool(
                        call.function.name, arguments=f
                    )
                    for c in res.content:
                        if c.type == "text":
                            await self.chat(c.text, tool=call.function.name)
                except Exception as exc:
                    s = str(exc)
                    await self.chat(f"Encountered an error when calling tool \
                                    {tool.call.function.name}: {s}",
                                    tool=tool.call.function.name)


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
                "parameters": tool.inputSchema,
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
        r = self.client.chat.completions.create(
            messages=self.messages, model=self.config.model, tools=self.tools
        )
        for response in r.choices:
            if self.config.debug:
                self.print(response)
            content = response.message.content
            if content is not None and content != "":
                self.print(">>", content)
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )
            if response.message.tool_calls is not None:
                for call in response.message.tool_calls:
                    f = call.function.arguments
                    if isinstance(f, str):
                        f = json.loads(f)
                    self.print(">>", f"Calling tool: {call.function.name}")
                    res = await self.config.session.call_tool(
                        call.function.name, arguments=f
                    )
                    for c in res.content:
                        if c.type == "text":
                            await self.chat(c.text, tool=call.function.name)


class Claude(ChatProvider):
    """
    Chat using the Claude API
    """

    def _convert_tool(self, tool):
        if not hasattr(tool, "inputSchema"):
            return tool
        tool.input_schema = tool.inputSchema
        del tool.inputSchema
        return tool

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
                self.print(">>", block.text)
            elif block.type == "tool_use":
                self.print(">>", f"Calling tool: {block.name}")
                res = await self.config.session.call_tool(
                    block.name, arguments=block.input
                )
                for c in res.content:
                    if c.type == "text":
                        await self.chat(c.text, tool=block.name)
