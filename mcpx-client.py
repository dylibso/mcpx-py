from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import chat
from ollama import ChatResponse

import sys
import os
import json
import argparse

# response: ChatResponse = chat(model='llama3.2', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])
# print(response['message']['content'])
# # or access fields directly from the response object
# print(response.message.content)

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="node",  # Executable
    args=["../../mcp.run/mcpx/build/index.js"],
    env=os.environ,
)


async def run(args):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            if args.command == 'list':
                tools = await session.list_tools()
                for _, tool in tools:
                    print(tool)
            elif args.command == 'tool':
                res = await session.call_tool(
                    sys.argv[1],
                    arguments=json.loads(sys.argv[2])
                )
                print(res.content)
            else:
                print("Invalid command: ", args.command)
                sys.exit(1)

if __name__ == "__main__":
    import asyncio
    args = argparse.ArgumentParser(
        prog="mcpx-client"
    )
    args.add_argument("command", required=True)
    asyncio.run(run(args.parse_args()))
