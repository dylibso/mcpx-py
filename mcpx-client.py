from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import chat
from ollama import ChatResponse
from anthropic import AsyncAnthropic

import os
import json
import argparse

SYSTEM_PROMPT = """
- when evaluating a javascript function code don't print the result to stdout
  instead just call the generated javascript function on its own since the code
  will be executed using eval
- only use tool calls when the prompt fits the use case exactly
- it is better to be asked for a tool to be used manually than
  to use a tool that isn't a perfect fit
"""


mcpx_path = os.environ.get("MCPX_PATH", "mcpx")

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command=mcpx_path,
    args=[],
    env=os.environ,
)

# Disable node errors
server_params.env['NODE_NO_WARNINGS'] = '1'


async def _get_tools(session):
    tools = await session.list_tools()
    out = []
    for name, t in tools:
        if t is None:
            continue
        for tool in t:
            out.append((name, tool))
    return out


async def list_cmd(args, session):
    tools = await _get_tools(session)
    for _, tool in tools:
        print()
        print(tool.name)
        print(tool.description)
        print("Input schema:")
        print(json.dumps(
            tool.inputSchema['properties'],
            indent=2
        ))


async def tool_cmd(args, session):
    res = await session.call_tool(
        args.name,
        arguments=json.loads(args.input)
    )
    for c in res.content:
        if c.type == 'text':
            print(c.text)


def _convert_tool_ollama(tool):
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        }
    }


def _convert_tool_claude(tool):
    if not hasattr(tool, 'inputSchema'):
        return tool
    tool.input_schema = tool.inputSchema
    del tool.inputSchema
    return tool


async def claude_provider(messages, msg, args, session, tools):
    tools = [_convert_tool_claude(t) for _, t in tools]
    client = AsyncAnthropic()
    messages.append({
        'role': 'user',
        'content': msg,
    })
    res = await client.messages.create(
        max_tokens=1024,
        messages=messages,
        model=args.model,
        tools=tools,
        system=SYSTEM_PROMPT,
    )
    for block in res.content:
        if args.debug:
            print(block)
        if block.type == 'text':
            messages.append({
                'role': 'assistant',
                'content': block.text,
            })
            print('>>', block.text)
        elif block.type == 'tool_use':
            res = await session.call_tool(
                block.name,
                arguments=block.input
            )
            for c in res.content:
                if c.type == 'text':
                    messages.append({
                        'role': 'assistant',
                        'content': c.text,
                    })
                    print(">>", c.text)


async def ollama_provider(messages, msg, args, session, tools):
    if len(messages) == 0:
        messages.append({
            'role': 'system',
            'content': SYSTEM_PROMPT
        })
    tools = [_convert_tool_ollama(t) for _, t in tools]
    messages.append({
        'role': 'user',
        'content': msg,
    })
    response: ChatResponse = chat(
        model=args.model,
        stream=False,
        tools=tools,
        messages=messages,
        format=args.format
    )
    if response.message.content != '':
        print(">>", response.message.content)
        messages.append({
            'role': 'assistant',
            'content': response.message.content,
        })
    if args.debug:
        print(response)
    if response.message.tool_calls is not None:
        for call in response.message.tool_calls:
            res = await session.call_tool(
                call.function.name,
                arguments=call.function.arguments
            )
            for c in res.content:
                if c.type == 'text':
                    messages.append({
                        'role': 'assistant',
                        'content': c.text,
                    })
                    print(">>", c.text)


async def chat_cmd(args, session):
    if args.model is None:
        if args.provider == "ollama":
            args.model = "llama3.1"
        elif args.provider == "claude":
            args.model = "claude-3-5-sonnet-20241022"
    tools = await _get_tools(session)
    messages = []
    while True:
        try:
            msg = input("> ")
            if msg == 'exit' or msg == 'quit':
                break
            if msg == '':
                continue
            if args.provider == 'ollama':
                await ollama_provider(messages, msg, args, session, tools)
            elif args.provider == 'claude':
                await claude_provider(messages, msg, args, session, tools)
            else:
                print("Invalid provider:", args.provider)
                exit(1)
        except Exception as exc:
            s = str(exc)
            if s != '':
                print("\nERROR>>", exc)
            else:
                print()
            continue


async def run(args):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            await args.func(args, session)

if __name__ == "__main__":
    import asyncio
    args = argparse.ArgumentParser(
        prog="mcpx-client"
    )
    args.add_argument("--debug", action='store_true', help="Enable debug logging")
    sub = args.add_subparsers(
        title="subcommand",
        help="subcommands",
        required=True
    )

    # List subcommand
    list_parser = sub.add_parser("list")
    list_parser.set_defaults(func=list_cmd)

    # Tool subcommand
    tool_parser = sub.add_parser("tool")
    tool_parser.set_defaults(func=tool_cmd)
    tool_parser.add_argument("name", help="Tool name")
    tool_parser.add_argument("input", help="Tool input", nargs='?', default='')

    # Chat subcommand
    chat_parser = sub.add_parser("chat")
    chat_parser.set_defaults(func=chat_cmd)
    chat_parser.add_argument("--provider", "-p", choices=["ollama", "claude"], default='claude', help="LLM provider")
    chat_parser.add_argument("--model", default=None, help="model")
    chat_parser.add_argument("--format", default="", help="Output format")

    # Run
    asyncio.run(run(args.parse_args()))
