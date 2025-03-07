import asyncio
from mcpx_py import Chat, openai_compatible_model


# Using Chat.send_message requires async, so we'll wrap this in an
# async function
async def main():
    llm = Chat("claude-3-5-sonnet-latest")

    # Or OpenAI
    # llm = Chat("gpt-4o")

    # Or Ollama
    # llm = Chat(openai_compatible_model("http://127.0.0.1:11434/v1", "qwen2.5"))

    # Or Gemini
    # llm = Chat("gemini-2.0-flash")

    # Prompt and print the results
    response = await llm.send_message("summarize the contents of http://example.com")
    print(response.data)
    response = await llm.send_message(
        "without refetching the contents, what are the first and last lines on that page?"
    )
    print(response.data)


if __name__ == "__main__":
    asyncio.run(main())
