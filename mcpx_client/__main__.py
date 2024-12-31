#!/usr/bin/env python3
import os
import readline
import atexit
import argparse
import json
import psutil

from dotenv import load_dotenv

from . import Claude, OpenAI, Ollama, ChatConfig, Client
from .chat import SYSTEM_PROMPT

CHAT_HELP = """
Available commands:
  !help    - Show this help message
  !clear   - Clear chat history
  !exit    - Exit the chat
  !tools   - List available tools
  !sh      - Execute a shell command
"""


async def list_cmd(args):
    client = Client()
    for install in client.installs.values():
        for tool in install.tools.values():
            print()
            print(tool.name)
            print(tool.description)
            print("Input schema:")
            print(json.dumps(tool.input_schema, indent=2))


async def tool_cmd(args):
    try:
        client = Client()
        res = client.call(tool=args.name, input=json.loads(args.input))
        for c in res:
            if c.type == "text":
                print(c.text)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


async def chat_loop(provider):
    try:
        msg = input("> ").strip()

        # Handle special commands
        if msg.startswith("!") or msg == "exit" or msg == "quit":
            if msg == "!help":
                print(CHAT_HELP)
                return True
            elif msg == "!clear":
                provider.messages = []
                print("Chat history cleared")
                return True
            elif msg == "!tools":
                tools = provider.get_tools(provider.config.session)
                print("\nAvailable tools:")
                for tool in tools:
                    print(f"- {tool.name}")
                return True
            elif msg.startswith("!sh "):
                os.system(msg[4:])
                return True
            elif msg in ["!exit", "!quit", "exit", "quit"]:
                print("Goodbye!")
                return False
        if msg == "":
            return True
        # TODO: maybe avoid fetching tools for each prompt
        provider.get_tools()
        await provider.chat(msg)
    except Exception as exc:
        s = str(exc)
        if s != "":
            print("\nERROR>>", exc)
        else:
            print()
    return True


async def chat_cmd(args):
    client = Client()
    if args.model is None:
        if args.provider == "ollama":
            args.model = "llama3.2"
        elif args.provider == "claude":
            args.model = "claude-3-5-sonnet-20241022"
        elif args.provider == "openai":
            args.model = "gpt-4o"
    config = ChatConfig(
        client=client,
        model=args.model,
        url=args.url,
        system=args.system,
        format=args.format,
        debug=args.debug,
    )
    provider = None
    if args.provider == "ollama":
        provider = Ollama(config)
    elif args.provider == "claude":
        provider = Claude(config)
    elif args.provider == "openai":
        provider = OpenAI(config)
    while True:
        ok = await chat_loop(provider)
        if not ok:
            break


def killtree(pid):
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()


async def run(args):
    # Setup command history
    histfile = os.path.join(
        os.environ.get("XTP_PLUGIN_CACHE_DIR", "."), ".mcpx-client-history"
    )
    try:
        os.makedirs(os.path.dirname(histfile), exist_ok=True)
        readline.set_history_length(1000)

        # Try to read existing history
        try:
            readline.read_history_file(histfile)
        except FileNotFoundError:
            pass

        # Register history save on exit
        atexit.register(readline.write_history_file, histfile)
    except Exception as e:
        print(f"Warning: Could not setup command history: {str(e)}")

    # env = os.environ.copy()
    # # Disable node errors
    # env["NODE_NO_WARNINGS"] = "1"
    # env["MCP_RUN_ORIGIN"] = args.origin
    # if args.mcpx_debug:
    #     env["LOG_LEVEL"] = "debug"
    # else:
    #     env["LOG_LEVEL"] = "silent"

    # Create server parameters for stdio connection
    if args.debug:
        print(os.environ)
    await args.func(args)


def main():
    import asyncio

    args = argparse.ArgumentParser(prog="mcpx-client")
    args.add_argument("--debug", action="store_true", help="Enable debug logging")
    args.add_argument("--origin", default="https://www.mcp.run", help="mcpx server")
    sub = args.add_subparsers(title="subcommand", help="subcommands", required=True)

    # List subcommand
    list_parser = sub.add_parser("list")
    list_parser.set_defaults(func=list_cmd)

    # Tool subcommand
    tool_parser = sub.add_parser("tool")
    tool_parser.set_defaults(func=tool_cmd)
    tool_parser.add_argument("name", help="Install name name")
    tool_parser.add_argument("input", help="Tool input", nargs="?", default="{}")

    # Chat subcommand
    chat_parser = sub.add_parser("chat")
    chat_parser.set_defaults(func=chat_cmd)
    chat_parser.add_argument(
        "--provider",
        "-p",
        choices=["ollama", "claude", "openai"],
        default="claude",
        help="LLM provider",
    )
    chat_parser.add_argument("--url", "-u", default=None, help="Provider endpoint URL")
    chat_parser.add_argument("--model", default=None, help="Model name")
    chat_parser.add_argument("--system", default=SYSTEM_PROMPT, help="System prompt")
    chat_parser.add_argument("--format", default="", help="Output format")

    # Run
    asyncio.run(run(args.parse_args()))


if __name__ == "__main__":
    load_dotenv()
    main()
