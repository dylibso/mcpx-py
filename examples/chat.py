import asyncio
from mcpx_py import Chat, ChatConfig


# Using Chat.send_message requires async, so we'll wrap this in an
# async function
async def main():
    # Connect to the Claude API, this also creates an mcp.run client
    llm = Chat(ChatConfig(model="claude-3-5-sonnet-latest"))

    # Prompt and print the results
    response = await llm.send_message("summarize the contents of example.com")
    print(response.data)
    response = await llm.send_message("without refetching the contents, what are the first and last lines on that page?")
    print(response.data)



if __name__ == "__main__":
    asyncio.run(main())
