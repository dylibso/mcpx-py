
from ollama import chat
from ollama import ChatResponse
from anthropic import AsyncAnthropic
from openai import OpenAI as ChatGPT

import json

SYSTEM_PROMPT = """
- when evaluating a javascript function code don't print the result to stdout
  instead just call the generated javascript function on its own since the code
  will be executed using eval
- only use tool calls when the prompt fits the use case exactly
- it is better to be asked for a tool to be used manually than
  to use a tool that isn't a perfect fit
"""


class ChatProvider:
    messages: list
    tools: list
    model: str

    def __init__(self, model=None):
        self.messages = []
        self.tools = []
        self.model = model

    def _convert_tool(self, tool):
        return tool

    async def chat(self, session, args, msg, tool=None):
        pass

    async def get_tools(self, session):
        tools = await session.list_tools()
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

    async def chat(self, session, args, msg, tool=None):
        if len(self.messages) == 0:
            self.messages.append({"role": "system", "content": args.system})
        self.messages.append(
            {
                "role": "user",
                "content": msg if tool is None else f"Result of {tool}: {msg}",
            }
        )
        response: ChatResponse = chat(
            model=args.model,
            stream=False,
            tools=self.tools,
            messages=self.messages,
            format=args.format,
        )
        if response.message.content is not None and response.message.content != "":
            print(">>", response.message.content)
            self.messages.append(
                {
                    "role": "assistant",
                    "content": response.message.content,
                }
            )
        if args.debug:
            print(response)
        if response.message.tool_calls is not None:
            for call in response.message.tool_calls:
                f = call.function.arguments
                if isinstance(f, str):
                    f = json.loads(f)
                res = await session.call_tool(
                    call.function.name, arguments=f
                )
                for c in res.content:
                    if c.type == "text":
                        await self.chat(
                            session, args, c.text, tool=call.function.name)
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

    async def chat(self, session, args, msg, tool=None):
        if len(self.messages) == 0:
            self.messages.append({"role": "system", "content": args.system})
        if not hasattr(self, 'client'):
            self.client = ChatGPT()
        self.messages.append(
            {
                "role": "user",
                "content": msg if tool is None else f"Result of {tool}: {msg}",
            }
        )
        r = self.client.chat.completions.create(
            messages=self.messages, model=args.model, tools=self.tools)
        for response in r.choices:
            if response.message.content is not None and response.message.content != "":
                print(">>", response.message.content)
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": response.message.content,
                    }
                )
            if args.debug:
                print(response)
            if response.message.tool_calls is not None:
                for call in response.message.tool_calls:
                    f = call.function.arguments
                    if isinstance(f, str):
                        f = json.loads(f)
                    res = await session.call_tool(
                        call.function.name, arguments=f
                    )
                    for c in res.content:
                        if c.type == "text":
                            await self.chat(
                                session, args, c.text, tool=call.function.name)


class Claude(ChatProvider):
    def _convert_tool(self, tool):
        if not hasattr(tool, "inputSchema"):
            return tool
        tool.input_schema = tool.inputSchema
        del tool.inputSchema
        return tool

    async def chat(self, session, args, msg, tool=None):
        if not hasattr(self, 'client'):
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
            model=self.model,
            tools=self.tools,
            system=args.system,
        )
        for block in res.content:
            if args.debug:
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
                res = await session.call_tool(block.name, arguments=block.input)
                for c in res.content:
                    if c.type == "text":
                        await self.chat(session, args, c.text, tool=block.name)
