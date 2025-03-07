import asyncio
from mcpx_py import Chat, BaseModel, Field
from typing import List


class Summary(BaseModel):
    """
    A summary of some longer text
    """

    source: str = Field("The source of the original_text")
    items: List[str] = Field("A list of summary points")
    first_line: str = Field("The first line from the original text")
    last_line: str = Field("The last line from the original text")


# Using Chat.send_message requires async, so we'll wrap this in an
# async function
async def main():
    # Prompt and print the results
    llm = Chat("claude-3-5-sonnet-latest", result_type=Summary)
    response = await llm.send_message("summarize the contents of example.com")
    print(response.data)


if __name__ == "__main__":
    asyncio.run(main())
