import asyncio
from mcpx import ChatConfig, Claude


async def main():
    config = ChatConfig(model="claude-3-5-sonnet-20241022")
    claude = Claude(config)
    claude.get_tools()
    async for message in claude.chat("what is 2+2 using the eval-js tool"):
        print(message)

if __name__ == '__main__':
    asyncio.run(main())
