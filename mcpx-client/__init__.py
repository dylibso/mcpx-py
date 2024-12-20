from ollama import chat
from ollama import ChatResponse
from anthropic import AsyncAnthropic
from openai import OpenAI as ChatGPT
from mcp import ClientSession, Tool

import json
from dataclasses import dataclass
from typing import Optional

SYSTEM_PROMPT = """
- when evaluating a javascript function code don't print the result to stdout
  instead just call the generated javascript function on its own since the code
  will be executed using eval
- only use tool calls when the prompt fits the use case exactly
- it is better to be asked for a tool to be used manually than
  to use a tool that isn't a perfect fit
"""


@dataclass
class ChatConfig:
    session: ClientSession
    model: str
    system: str = ""
    format: Optional[str] = None
    debug: bool = False


class ChatProvider:
    messages: list
    tools: list
    model: str

    def __init__(self, config: ChatConfig):
        self.messages = []
        self.tools = []
        self.config = config

    def _convert_tool(self, tool: Tool):
        return tool

    async def chat(self, msg: str, tool: Optional[str] = None):
        pass

    async def get_tools(self):
        tools = await self.config.session.list_tools()
        self.tools = []
        for name, t in tools:
            if t is None:
                continue
            for tool in t:
                self.tools.append(self._convert_tool(tool))
        return self.tools


class Ollama(ChatProvider):
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
        self.messages.append(
            {
                "role": "user",
                "content": msg if tool is None else f"Result of {tool}: {msg}",
            }
        )
        response: ChatResponse = chat(
            model=self.config.model,
            stream=False,
            tools=self.tools,
            messages=self.messages,
            format=self.config.format,
        )
        if self.config.debug:
            print(response)
        if response.message.content is not None and response.message.content != "":
            print(">>", response.message.content)
            self.messages.append(
                {
                    "role": "assistant",
                    "content": response.message.content,
                }
            )
        if response.message.tool_calls is not None:
            for call in response.message.tool_calls:
                f = call.function.arguments
                if isinstance(f, str):
                    f = json.loads(f)
                res = await self.config.session.call_tool(
                    call.function.name, arguments=f
                )
                for c in res.content:
                    if c.type == "text":
                        await self.chat(c.text, tool=call.function.name)
                        # print(">>", c.text)


class OpenAI(ChatProvider):
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
            self.client = ChatGPT()
        self.messages.append(
            {
                "role": "user",
                "content": msg if tool is None else f"Result of {tool}: {msg}",
            }
        )
        r = self.client.chat.completions.create(
            messages=self.messages, model=self.config.model, tools=self.tools
        )
        for response in r.choices:
            if self.config.debug:
                print(response)
            if response.message.content is not None and response.message.content != "":
                print(">>", response.message.content)
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": response.message.content,
                    }
                )
            if response.message.tool_calls is not None:
                for call in response.message.tool_calls:
                    f = call.function.arguments
                    if isinstance(f, str):
                        f = json.loads(f)
                    res = await self.config.session.call_tool(
                        call.function.name, arguments=f
                    )
                    for c in res.content:
                        if c.type == "text":
                            await self.chat(c.text, tool=call.function.name)


class Claude(ChatProvider):
    def _convert_tool(self, tool):
        if not hasattr(tool, "inputSchema"):
            return tool
        tool.input_schema = tool.inputSchema
        del tool.inputSchema
        return tool

    async def chat(self, msg: str, tool: Optional[str] = None):
        if not hasattr(self, "client"):
            self.client = AsyncAnthropic()
        self.messages.append(
            {
                "role": "user",
                "content": msg if tool is None else f"Result of {tool}: {msg}",
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
                print(block)
            if block.type == "text":
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": block.text,
                    }
                )
                print(">>", block.text)
            elif block.type == "tool_use":
                res = await self.config.session.call_tool(
                    block.name, arguments=block.input
                )
                for c in res.content:
                    if c.type == "text":
                        await self.chat(c.text, tool=block.name)
