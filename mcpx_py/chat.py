from mcpx_pydantic_ai import Agent, pydantic_ai, mcp_run


from dataclasses import dataclass
from typing import List, TypedDict

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


@dataclass
class ChatConfig:
    """
    Stores configuration and session for chats
    """

    client: mcp_run.Client | None = None
    """
    mcp.run client
    """

    model: str | pydantic_ai.models.KnownModelName | None = None
    """
    Model name, if set to `None` then a default model will be selected
    """

    system: str = SYSTEM_PROMPT
    """
    System prompt
    """

    format: type = str
    """
    Output format
    """

    model_settings: pydantic_ai.agent.ModelSettings | None = None
    """
    Configure model-specifc settings
    """

    ignore_tools: List[str] | None = None
    """
    A list of tool names to ignore
    """


class Chat:
    """
    LLM chat
    """

    agent: Agent
    config: ChatConfig
    history: list

    def __init__(
        self,
        config: ChatConfig | str,
        *args,
        **kw,
    ):
        if not isinstance(config, ChatConfig):
            config = ChatConfig(model=config)
        self.config = config

        if "system_prompt" not in kw:
            kw["system_prompt"] = config.system

        if "result_type" not in kw:
            kw["result_type"] = config.format

        self.agent = Agent(
            config.model,
            client=config.client,
            ignore_tools=config.ignore_tools,
            *args,
            **kw,
        )
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
            return
        x = []
        for r in self.config.client.search(input["q"]):
            x.append(
                {
                    "slug": r.slug,
                    "meta": r.meta,
                    "installation_count": r.installation_count,
                }
            )
        return x

    def _tool_mcp_run_get_profiles(self, input: TypedDict("GetProfile", {})):
        p = []
        for user, u in self.config.client.profiles.items():
            if user == "~":
                continue
            for profile in u.values():
                p.append(
                    {
                        "name": f"{user}/{profile.slug.name}",
                        "description": profile.description,
                    }
                )
        return p

    def _tool_mcp_run_set_profile(
        self, input: TypedDict("SetProfile", {"profile": str})
    ):
        profile = input["profile"]
        self.agent.client.logger.info("Setting profile to {profile}")
        if "/" not in profile:
            profile = "~/" + profile
        self.agent.set_profile(profile)
        return f"Active profile set to {profile}"

    def _tool_mcp_run_current_profile(self, input):
        return self.client.config.profile
