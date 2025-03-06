import asyncio
from mcpx_py import Chat, Claude


# Using ChatProvider.chat requires async, so we'll wrap this in an
# async function
async def main():
    # Connect to the Claude API, this also creates an mcp.run client
    llm = Chat(Claude)

    # Or OpenAI
    # from mcpx_py import OpenAI
    # llm = Chat(OpenAI)

    # Or Ollama
    # from mcpx_py import Ollama
    # llm = Chat(Ollama)

    # Or Gemini
    # from mcpx_py import Gemini
    # llm = Chat(Gemini)

    # Prompt claude and iterate over the results
    async for response in llm.send_message(
        "summarize the contents of http://example.com"
    ):
        if response.content:
            print(response.content)

    async for response in llm.send_message(
        "without fetching the webpage again, what is the first word on example.com?"
    ):
        if response.content:
            print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
