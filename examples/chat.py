import asyncio
from mcpx import Chat, Claude


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

    # Prompt claude and iterate over the results
    async for message in llm.chat("what is the area of a three foot cube"):
        print(message)


if __name__ == "__main__":
    asyncio.run(main())
