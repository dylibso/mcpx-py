#!/usr/bin/env python3

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import chat
from ollama import ChatResponse
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

import os
import json
import argparse

try:
    import readline
    import atexit
except ImportError:
    pass

SYSTEM_PROMPT = """
- when evaluating a javascript function code don't print the result to stdout
  instead just call the generated javascript function on its own since the code
  will be executed using eval
- only use tool calls when the prompt fits the use case exactly
- it is better to be asked for a tool to be used manually than
  to use a tool that isn't a perfect fit
"""

load_dotenv()

mcpx_path = os.environ.get("MCPX_PATH", "mcpx")

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command=mcpx_path,
    args=[],
    env=os.environ,
)

# Disable node errors
server_params.env["NODE_NO_WARNINGS"] = "1"
server_params.env["MCPX_NO_REFRESH"] = "1"


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

    async def chat(self, session, args, msg):
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


async def list_cmd(args, session):
    tools = await ChatProvider().get_tools(session)
    for tool in tools:
        print()
        print(tool.name)
        print(tool.description)
        print("Input schema:")
        print(json.dumps(tool.inputSchema["properties"], indent=2))


async def tool_cmd(args, session):
    res = await session.call_tool(args.name, arguments=json.loads(args.input))
    for c in res.content:
        if c.type == "text":
            print(c.text)


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

    async def chat(self, session, args, msg):
        if len(self.messages) == 0:
            self.messages.append({"role": "system", "content": SYSTEM_PROMPT})
        await self.get_tools(session)
        self.messages.append(
            {
                "role": "user",
                "content": msg,
            }
        )
        response: ChatResponse = chat(
            model=args.model,
            stream=False,
            tools=self.tools,
            messages=self.messages,
            format=args.format,
        )
        if response.message.content != "":
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
                res = await session.call_tool(
                    call.function.name, arguments=call.function.arguments
                )
                for c in res.content:
                    if c.type == "text":
                        self.messages.append(
                            {
                                "role": "assistant",
                                "content": c.text,
                            }
                        )
                        print(">>", c.text)


class Claude(ChatProvider):
    def _convert_tool(self, tool):
        if not hasattr(tool, "inputSchema"):
            return tool
        tool.input_schema = tool.inputSchema
        del tool.inputSchema
        return tool

    async def chat(self, session, args, msg):
        # TODO: maybe avoid fetching tools for each prompt
        await self.get_tools(session)
        client = AsyncAnthropic()
        self.messages.append(
            {
                "role": "user",
                "content": msg,
            }
        )
        res = await client.messages.create(
            max_tokens=1024,
            messages=self.messages,
            model=self.model,
            tools=self.tools,
            system=SYSTEM_PROMPT,
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
                        self.messages.append(
                            {
                                "role": "assistant",
                                "content": c.text,
                            }
                        )
                        print(">>", c.text)


async def chat_cmd(args, session):
    if args.model is None:
        if args.provider == "ollama":
            args.model = "llama3.2"
        elif args.provider == "claude":
            args.model = "claude-3-5-sonnet-20241022"
    provider = None
    if args.provider == "ollama":
        provider = Ollama(args.model)
    elif args.provider == "claude":
        provider = Claude(args.model)
    while True:
        try:
            msg = input("> ")
            if msg == "exit" or msg == "quit":
                try:
                    readline.remove_history_item(
                        readline.get_current_history_length() - 1
                    )
                except:
                    pass
                break
            if msg == "":
                continue
            await provider.chat(session, args, msg)
        except Exception as exc:
            s = str(exc)
            if s != "":
                print("\nERROR>>", exc)
            else:
                print()
            continue


async def run(args):
    try:
        histfile = os.environ.get("XTP_PLUGIN_CACHE_DIR", ".") + "/.mcpx-client-history"
        atexit.register(readline.write_history_file, histfile)
        readline.set_history_length(1000)
        readline.read_history_file(histfile)
    except:
        pass

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            await args.func(args, session)


def main():
    import asyncio
    args = argparse.ArgumentParser(prog="mcpx-client")
    args.add_argument("--debug", action="store_true", help="Enable debug logging")
    sub = args.add_subparsers(title="subcommand", help="subcommands", required=True)

    # List subcommand
    list_parser = sub.add_parser("list")
    list_parser.set_defaults(func=list_cmd)

    # Tool subcommand
    tool_parser = sub.add_parser("tool")
    tool_parser.set_defaults(func=tool_cmd)
    tool_parser.add_argument("name", help="Tool name")
    tool_parser.add_argument("input", help="Tool input", nargs="?", default="")

    # Chat subcommand
    chat_parser = sub.add_parser("chat")
    chat_parser.set_defaults(func=chat_cmd)
    chat_parser.add_argument(
        "--provider",
        "-p",
        choices=["ollama", "claude"],
        default="claude",
        help="LLM provider",
    )
    chat_parser.add_argument("--model", default=None, help="model")
    chat_parser.add_argument("--format", default="", help="Output format")

    # Run
    asyncio.run(run(args.parse_args()))


if __name__ == '__main__':
    main()
