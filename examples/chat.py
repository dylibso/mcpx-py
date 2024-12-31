import asyncio
from mcpx import ChatConfig, Claude

# Configure the chat provider
config = ChatConfig(model="claude-3-5-sonnet-20241022")


# Using ChatProvider.chat requires async, so we'll wrap this in an
# async function
async def main():
    # Connect to the Claude API, this also creates an mcp.run client
    claude = Claude(config)

    # Fetch installted tools from mcp.run
    claude.get_tools()

    # Prompt claude and iterate over the results
    async for message in claude.chat("what is 2+2 using the eval-js tool"):
        print(message)


if __name__ == "__main__":
    asyncio.run(main())
