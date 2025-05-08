from mcpx_pydantic_ai import Agent, pydantic_ai


import asyncio


SYSTEM_PROMPT = """
- Do not come up with directions or indications.
- Always use the provided tools/functions when applicable, and share the
  results of tool calls with the user
- Invoke the tools upon requests you cannot fulfill on your own
  and parse the responses
- Always try to provide a well formatted, itemized summary
- If the user provides the result of a tool and no other action is needed just
  repeat it back to them
- Only perform verification of a computation at most once if absolutely needed,
  if a computation is performed using a tool then the results do not need to be
  re-verified
"""


class Chat:
    """
    LLM chat
    """

    agent: Agent
    history: list

    def __init__(
        self,
        *args,
        **kw,
    ):
        if "system_prompt" not in kw:
            kw["system_prompt"] = SYSTEM_PROMPT

        self.agent = Agent(
            *args,
            **kw,
        )
        self.history = []

    @property
    def client(self):
        """
        mcp.run client
        """
        return self.agent.client

    def clear_history(self):
        """
        Clear chat history
        """
        self.history = []

    async def send_message(self, msg: str, run_mcp_servers: bool = True, *args, **kw):
        """
        Send a chat message to the LLM
        """

        async def inner():
            with pydantic_ai.capture_run_messages() as messages:
                return (
                    await self.agent.run(
                        msg,
                        message_history=self.history,
                        *args,
                        **kw,
                    ),
                    messages,
                )

        if run_mcp_servers:
            async with self.agent.run_mcp_servers():
                res, messages = await inner()
        else:
            res, messages = await inner()
        self.history.extend(messages)
        return res

    def send_message_sync(self, *args, **kw):
        """
        Send a chat message to the LLM synchronously

        This creates a new event loop to run the async send_message method.
        """
        # Create a new event loop to avoid warnings about coroutines not being awaited
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.send_message(*args, **kw))
        finally:
            loop.close()

    async def iter(self, msg: str, run_mcp_servers: bool = True, *args, **kw):
        """
        Send a chat message to the LLM
        """

        async def inner():
            with pydantic_ai.capture_run_messages() as messages:
                async with self.agent.iter(
                    msg, message_history=self.history, *args, **kw
                ) as run:
                    async for node in run:
                        yield node
            self.history.extend(messages)

        if run_mcp_servers:
            async with self.agent.run_mcp_servers():
                async for msg in inner():
                    yield msg
        else:
            async for msg in inner():
                yield msg

    async def iter_content(self, msg: str, run_mcp_servers: bool = True, *args, **kw):
        """
        Send a chat message to the LLM
        """

        async def inner():
            with pydantic_ai.capture_run_messages() as messages:
                async with self.agent.iter(
                    msg, message_history=self.history, *args, **kw
                ) as run:
                    async for node in run:
                        if hasattr(node, "response"):
                            content = node.response
                        elif hasattr(node, "model_response"):
                            content = node.model_response
                        elif hasattr(node, "request"):
                            content = node.request
                        elif hasattr(node, "model_request"):
                            content = node.model_request
                        elif hasattr(node, "data"):
                            content = node.data
                        elif hasattr(node, "output"):
                            content = node
                        else:
                            continue
                        yield content
            self.history.extend(messages)

        f = inner()
        if run_mcp_servers:
            async with self.agent.run_mcp_servers():
                async for msg in f:
                    yield msg
        else:
            async for msg in f:
                yield msg

    async def inspect(self, msg: str, run_mcp_servers: bool = True, *args, **kw):
        """
        Send a chat message to the LLM
        """

        async def inner():
            with pydantic_ai.capture_run_messages() as messages:
                res = await self.agent.run(
                    msg,
                    message_history=self.history,
                    *args,
                    **kw,
                )
            return res, messages

        if run_mcp_servers:
            async with self.agent.run_mcp_servers():
                return await inner()
        else:
            return await inner()

    async def list_tools(self):
        async with self.client.mcp_sse(self.agent.client.profile).connect() as session:
            tools = await session.list_tools()
            for tool in tools.tools:
                yield tool
