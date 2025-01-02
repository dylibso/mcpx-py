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

    # Or Gemini
    # from mcpx import Gemini
    # llm = Chat(Gemini)

    # Prompt claude and iterate over the results
    async for response in llm.send_message(
        "what is the total volume of nine three foot by 5 foot boxes"
    ):
        print(response)


if __name__ == "__main__":
    asyncio.run(main())
