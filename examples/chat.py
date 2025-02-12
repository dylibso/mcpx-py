import asyncio
from mcpx_py import Chat, Claude


# Using ChatProvider.chat requires async, so we'll wrap this in an
# async function
async def main():
    # Connect to the Claude API, this also creates an mcp.run client
    llm = Chat(Claude)

    # Or OpenAI
    # from mcpx import OpenAI
    # llm = Chat(OpenAI)

    # Or Ollama
    # from mcpx import Ollama
    # llm = Chat(Ollama)

    # Or Gemini
    # from mcpx import Gemini
    # llm = Chat(Gemini)

    # Prompt claude and iterate over the results
    async for response in llm.send_message("summarize the contents of example.com"):
        print(response)


if __name__ == "__main__":
    asyncio.run(main())
