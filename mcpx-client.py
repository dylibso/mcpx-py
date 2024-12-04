from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import chat
from ollama import ChatResponse

import os
import json
import argparse

mcpx_path = os.environ.get("MCPX_PATH", "../../mcp.run/mcpx/build/index.js")

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="node",  # Executable
    args=[mcpx_path],
    env=os.environ,
)


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


def _convert_tool(tool):
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        }
    }


async def chat_cmd(args, session):
    tools = await _get_tools(session)
    tools = [_convert_tool(t) for _, t in tools]
    messages = [{
        'role': 'system',
        'content': """
            - when evaluating code use the eval_js tool, if a prompt does not
              generate or contain code then do not use eval_js
            - the greet tool should only be used when saying hello
            - only use tool calls when the prompt fits the use case exactly
            - NEVER force tool use when it doesn't make complete sense
            - it is better to be asked for a tool to be used manually than
              to use a tool that isn't 100% right
        """,
    }]
    while True:
        try:
            msg = input("> ")
            if msg == 'exit' or msg == 'quit':
                break
            if msg == '':
                continue
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
    chat_parser.add_argument("--model", default="llama3.1", help="ollama model")
    chat_parser.add_argument("--format", default="", help="Output format")

    # Run
    asyncio.run(run(args.parse_args()))
