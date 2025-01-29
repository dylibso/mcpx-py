#!/usr/bin/env python3
import os
import readline
import atexit
import argparse
import json
import psutil
import traceback
import logging

from dotenv import load_dotenv

from . import Claude, OpenAI, Ollama, ChatConfig, Client, Gemini, Chat, ClientConfig
from .chat import SYSTEM_PROMPT

CHAT_HELP = """
Available commands:
  !help    - Show this help message
  !clear   - Clear chat history
  !exit    - Exit the chat
  !tools   - List available tools
  !sh      - Execute a shell command
"""


async def list_cmd(client, args):
    for install in client.installs.values():
        for tool in install.tools.values():
            print()
            print(tool.name)
            print(tool.description)
            print("Input schema:")
            print(json.dumps(tool.input_schema, indent=2))


async def tool_cmd(client, args):
    try:
        res = client.call(tool=args.name, input=json.loads(args.input))
        for c in res:
            if c.type == "text":
                print(c.text)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


EXIT_COUNT = 0


async def chat_loop(chat):
    global EXIT_COUNT
    try:
        msg = input("> ").strip()
        EXIT_COUNT = 0
    except EOFError:
        if EXIT_COUNT == 0:
            print("\nPress Ctrl+D again to exit")
        else:
            EXIT_COUNT = 0
            return False
        EXIT_COUNT += 1
        return True

    try:
        # Handle special commands
        if msg.startswith("!") or msg == "exit" or msg == "quit":
            if msg == "!help":
                print(CHAT_HELP)
                return True
            elif msg == "!clear":
                chat.clear_history()
                print("Chat history cleared")
                return True
            elif msg == "!tools":
                tools = chat.provider.get_tools()
                print("\nAvailable tools:")
                for tool in tools:
                    print(f"- {tool['name']}")
                return True
            elif msg.startswith("!sh "):
                os.system(msg[4:])
                return True
            elif msg in ["!exit", "!quit", "exit", "quit"]:
                print("Goodbye!")
                return False
        if msg == "":
            return True
        async for res in chat.send_message(msg):
            if res.role == "assistant":
                print(">>", res.content)
            elif res.role == "tool":
                print(
                    ">>",
                    f"Calling {res.tool.name}",
                )
                print(
                    ">>",
                    f"Input: {res.tool.input}",
                )
                print(
                    ">>",
                    f"Result: {res.content}",
                )
    except Exception:
        print("\nERROR>>", traceback.format_exc())
    return True


async def chat_cmd(client, args):
    config = ChatConfig(
        client=client,
        model=args.model,
        base_url=args.url,
        system=args.system,
        format=args.format,
    )
    provider = None
    if args.provider == "ollama":
        provider = Ollama(config)
    elif args.provider == "claude":
        provider = Claude(config)
    elif args.provider == "openai":
        provider = OpenAI(config)
    elif args.provider == "gemini":
        provider = Gemini(config)

    if args.provider == "ollama" and args.pull:
        print(f">> Pulling {config.model}")
        provider.provider_client.pull(model=config.model, stream=False)

    chat = Chat(provider)

    while True:
        ok = await chat_loop(chat)
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

    client = Client(config=ClientConfig(base_url=args.base_url, profile=args.profile))
    if args.log_level is not None and args.log_level != "off":
        level = logging.getLevelName(args.log_level.upper())
        client.configure_logging(level=level)

    await args.func(client, args)


def main():
    import asyncio

    args = argparse.ArgumentParser(prog="mcpx-client")
    args.add_argument(
        "--log-level",
        choices=["off", "critical", "fatal", "warn", "info", "debug", "error"],
        help="Select log level",
    )
    args.add_argument("--base-url", default="https://www.mcp.run", help="nURL")
    args.add_argument("--profile", default="default", help="mcpx profile")
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
        choices=["ollama", "claude", "openai", "gemini"],
        default="claude",
        help="LLM provider",
    )
    chat_parser.add_argument("--url", "-u", default=None, help="Provider endpoint URL")
    chat_parser.add_argument("--model", default=None, help="Model name")
    chat_parser.add_argument("--system", default=SYSTEM_PROMPT, help="System prompt")
    chat_parser.add_argument("--format", default="", help="Output format")
    chat_parser.add_argument(
        "--pull", action="store_true", help="Pull the model before running, Ollama only"
    )

    # Run
    asyncio.run(run(args.parse_args()))


if __name__ == "__main__":
    load_dotenv()
    main()
