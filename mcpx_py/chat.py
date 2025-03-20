from mcpx_pydantic_ai import Agent, pydantic_ai


from typing import TypedDict

from . import builtin_tools


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
        ignore_builtin_tools: bool = False,
        **kw,
    ):
        if "system_prompt" not in kw:
            kw["system_prompt"] = SYSTEM_PROMPT

        self.agent = Agent(
            *args,
            **kw,
        )
        if not ignore_builtin_tools:
            self._register_builtins()
        self.history = []

    def _register_builtins(self):
        for tool in builtin_tools.TOOLS:
            self.agent.register_tool(tool, getattr(self, "_tool_" + tool.name))

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

    async def send_message(self, msg: str, *args, **kw):
        """
        Send a chat message to the LLM
        """
        with pydantic_ai.capture_run_messages() as messages:
            res = await self.agent.run(
                msg,
                message_history=self.history,
                *args,
                **kw,
            )
        self.history.extend(messages)
        return res

    def send_message_sync(self, msg, *args, **kw):
        """
        Send a chat message to the LLM
        """
        with pydantic_ai.capture_run_messages() as messages:
            res = self.agent.run_sync(
                msg,
                message_history=self.history,
                *args,
                **kw,
            )
        self.history.extend(messages)
        return res

    async def iter(self, msg, *args, **kw):
        """
        Send a chat message to the LLM
        """
        with pydantic_ai.capture_run_messages() as messages:
            async with self.agent.iter(
                msg, message_history=self.history, *args, **kw
            ) as run:
                async for node in run:
                    yield node
        self.history.extend(messages)

    async def iter_content(self, msg, *args, **kw):
        """
        Send a chat message to the LLM
        """
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
                    else:
                        continue
                    yield content
        self.history.extend(messages)

    async def inspect(self, msg, *args, **kw):
        """
        Send a chat message to the LLM
        """
        with pydantic_ai.capture_run_messages() as messages:
            res = await self.send_message(msg, *args, **kw)
        return res, messages

    def _tool_mcp_run_search_servlets(
        self, input: TypedDict("SearchServlets", {"q": str})
    ):
        q = input.get("q", "")
        if q == "":
            return "ERROR: provide a query when searching"
        x = []
        for r in self.agent.client.search(input["q"]):
            x.append(
                {
                    "slug": r.slug,
                    "schema": {
                        "name": r.meta.get("name"),
                        "description": r.meta.get("description"),
                    },
                    "installation_count": r.installation_count,
                }
            )
        return x

    def _tool_mcp_run_get_profiles(self, input: TypedDict("GetProfile", {})):
        p = []
        for user, u in self.agent.client.profiles.items():
            if user == "~":
                continue
            for profile in u.values():
                p.append(
                    {
                        "name": f"{user}/{profile.slug}",  # Assume slug is string
                        "description": profile.description,
                    }
                )
        return p

    def _tool_mcp_run_set_profile(
        self, input: TypedDict("SetProfile", {"profile": str})
    ):
        profile = input["profile"]
        if "/" not in profile:
            profile = "~/" + profile
        self.agent.set_profile(profile)
        return f"Active profile set to {profile}"

    def _tool_mcp_run_current_profile(self, input: TypedDict("CurrentProfile", {})):
        """Get current profile name"""
        return self.agent.client.config.profile
