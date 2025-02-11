from ollama import ChatResponse as OllamaChatResponse, Client as OllamaClient
from anthropic import AsyncAnthropic
from openai import OpenAI as OpenAIClient

import traceback
import json
from dataclasses import dataclass
from typing import Optional, List, Iterator, Callable
import tempfile
import os
import asyncio

from mcp_run import Client, Tool
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

    client: Client | None = None
    """
    mcp.run client
    """

    api_key: str | None = None
    """
    API key for the ChatProvider (Anthropic, OpenAI, ...). If set to `None` then
    then the default environment variable for that provider will be used.
    """

    model: str | None = None
    """
    Model name, if set to `None` then a default model will be selected
    """

    base_url: Optional[str] = None
    """
    Optionally specify an alternative API URL for a provider, this can be used
    to take advantage of APIs that offer compatibility layers with providers
    that are already supported
    """

    system: str = SYSTEM_PROMPT
    """
    System prompt
    """

    format: Optional[str] = None
    """
    Output format
    """

    provider_client: object | None = None
    """
    provider_client is a pre-initialized client for the selected ChatProvider.
    For example, when using OpenAI, this would be an `openai.OpenAI` object.

    Typically this can be set to `None`
    """

    max_tokens: int = 1024
    """
    Configure the number of tokens that can be generated by a chat
    """


@dataclass
class ToolCall:
    name: str
    """
    Tool name
    """

    input: object
    """
    The input that was passed to the tool
    """


@dataclass
class ChatResponse:
    """
    Response from LLM chats
    """

    role: str
    """
    The source of the message.

    "assistant" means the message is from an LLM
    "tool" means the message is the result of a tool call
    """

    content: str
    """
    Message content
    """

    kind: str = "text"
    """
    Message kind, for example "text" or "image"
    """

    tool: ToolCall | None = None
    """
    If role is "tool" then this contains additional information about the
    tool call
    """

    _error: bool | None = None
    """
    Set to True when the tool raised an error
    """

    @property
    def is_error(self):
        """
        Check if an error occured
        """
        return self.error or False


class ChatProvider:
    """
    Defines the interface for all chat provider implementations
    """

    messages: List[dict]
    """
    Message history
    """

    tools: List[dict]
    """
    A list of converted tool objects
    """

    config: ChatConfig
    """
    Chat configuration options
    """

    provider_client: object | None
    """
    API provider client
    """

    def __init__(self, config: ChatConfig | None = None):
        self.messages = []
        self.tools = []
        self.config = config or ChatConfig()
        if self.config.client is None:
            self.config.client = Client()
        if self.config.model is None:
            self.config.model = self._default_model()
        if self.config.provider_client is not None:
            self.provider_client = self.config.provider_client
        else:
            self.provider_client = self._default_provider_client(self.config)
        self.logger = config.client.logger.getChild("chat")

    @staticmethod
    def _default_provider_client(config):
        """
        Returns the default API provider client
        """
        pass

    @staticmethod
    def _default_model() -> str:
        """
        Specifies the default model for a chat provider
        """
        return ""

    @staticmethod
    def _convert_tool(tool: Tool):
        """
        Convert a tool from Tool to the expected format
        for the given provider
        """
        return tool

    def clear_history(self):
        """
        Clear the chat history
        """
        self.messages = []

    def append_message(self, msg: str, role: str = "user", tool: str | None = None):
        """
        Add a new message to the chat
        """
        self.messages.append(
            {
                "role": role,
                "content": msg if tool is None else f"{tool} output: {msg}",
            }
        )

    async def chat(
        self, msg: str, tool: Optional[str] = None
    ) -> Iterator[ChatResponse]:
        """
        Handle chat message, if `tool` is set then the message is the result
        of a tool call
        """
        pass

    def chat_sync(self, *args, **kw) -> Iterator[ChatResponse]:
        """
        Handle chat message, if `tool` is set then the message is the result
        of a tool call
        """
        for res in asyncio.run(self.chat(*args, **kw)):
            yield res

    def _builtin_tools(self) -> List[object]:
        return [
            self._convert_tool(builtin_tools.SEARCH),
            self._convert_tool(builtin_tools.GET_PROFILES),
            self._convert_tool(builtin_tools.SET_PROFILE),
        ]

    def get_tools(self) -> List[object]:
        """
        Get all tools from the mcp server
        """
        tools = self.config.client.installs
        self.tools = self._builtin_tools()
        for name, t in tools.items():
            if t is None:
                continue
            for tool in t.tools.values():
                self.logger.info(f"Tool: {tool.name}")
                self.tools.append(self._convert_tool(tool))
        return self.tools

    async def call_tool(self, name: str, input: object, **kw) -> Iterator[ChatResponse]:
        """
        Call a tool by name with the given input, the extra arguments are passed to
        `Client.call`
        """
        if isinstance(input, str):
            input = json.loads(input)
        self.logger.info(f"Calling tool: {name} with input: {input}")
        try:
            # Handle builtin tools
            if name in ["mcp_run_search_servlets"]:
                x = []
                for r in self.config.client.search(input["q"]):
                    x.append(
                        {
                            "slug": r.slug,
                            "meta": r.meta,
                            "installation_count": r.installation_count,
                        }
                    )
                c = json.dumps(x)
                yield ChatResponse(
                    role="tool",
                    content=c,
                    tool=ToolCall(name=name, input=input),
                )
                async for res in self.chat(c, tool=name):
                    yield res
                return
            elif name in ["mcp_run_get_profiles"]:
                p = []
                for user, u in self.config.client.profiles.items():
                    if user == '~':
                        continue
                    for profile in u.values():
                        p.append(
                            {
                                "name": f"{user}/{profile.name}",
                                "description": profile.description,
                            }
                        )
                c = json.dumps(p)
                yield ChatResponse(
                    role="tool",
                    content=c,
                    tool=ToolCall(name=name, input=input),
                )
                async for res in self.chat(c, tool=name):
                    yield res
                return
            elif name in ["mcp_run_set_profile"]:
                profile = input["profile"]
                c = f"Active profile set to {profile}"
                yield ChatResponse(
                    role="tool",
                    content=c,
                    tool=ToolCall(name=name, input=input),

                )
                async for res in self.chat(c, tool=name):
                    yield res
                return

            res = self.config.client.call(tool=name, input=input, **kw)
            for c in res.content:
                if c.type == "text":
                    yield ChatResponse(
                        role="tool",
                        content=c.text,
                        tool=ToolCall(name=name, input=input),
                    )
                    async for res in self.chat(c.text, tool=name):
                        yield res
                elif c.type == "image":
                    ext = ".jpg"
                    if c.mime_type == "image/png":
                        ext = ".png"
                    with tempfile.NamedTemporaryFile(
                        suffix=ext, delete=False, delete_on_close=False
                    ) as tmp:
                        tmp.write(c._data)
                        yield ChatResponse(
                            role="tool",
                            content=tmp.name,
                            kind="image",
                            tool=ToolCall(name=name, input=input),
                        )
        except Exception:
            s = traceback.format_exc()
            yield ChatResponse(
                role="tool",
                content=s,
                tool=ToolCall(name=name, input=input),
                _error=True,
            )
            async for res in self.chat(
                f"Encountered an error when calling tool \
                            {name}: {s}",
                tool=name,
            ):
                yield res


class Ollama(ChatProvider):
    """
    Chat using ollama
    """

    @staticmethod
    def _default_model() -> str:
        return "qwen2.5"

    @staticmethod
    def _convert_tool(tool):
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }

    @staticmethod
    def _default_provider_client(config):
        return OllamaClient(host=config.base_url)

    async def chat(
        self, msg: str, tool: Optional[str] = None
    ) -> Iterator[ChatResponse]:
        if len(self.messages) == 0:
            self.append_message(self.config.system, role="system")
        self.append_message(msg, role="user" if tool is None else "tool", tool=tool)
        response: OllamaChatResponse = self.provider_client.chat(
            model=self.config.model,
            stream=False,
            tools=self.tools,
            messages=self.messages,
            format=self.config.format,
        )
        self.logger.debug("Response:", response)
        content = response.message.content
        if content is not None and content != "":
            self.append_message(content, role="assistant")
            yield ChatResponse(role="assistant", content=content)
        if response.message.tool_calls is not None:
            for call in response.message.tool_calls:
                f = call.function.arguments
                async for res in self.call_tool(call.function.name, f):
                    yield res


class OpenAI(ChatProvider):
    """
    Chat using the OpenAI API
    """

    @staticmethod
    def _convert_tool(tool):
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }

    @staticmethod
    def _default_model() -> str:
        return "gpt-4o"

    @staticmethod
    def _default_provider_client(config):
        return OpenAIClient(base_url=config.base_url, api_key=config.api_key)

    async def chat(
        self, msg: str, tool: Optional[str] = None
    ) -> Iterator[ChatResponse]:
        if len(self.messages) == 0:
            self.append_message(self.config.system, role="system")
        self.append_message(msg, tool=tool)
        r = self.provider_client.chat.completions.create(
            messages=self.messages,
            model=self.config.model,
            tools=self.tools,
            max_completion_tokens=self.config.max_tokens,
        )
        for response in r.choices:
            self.logger.debug("Response", response)
            content = response.message.content
            if content is not None and content != "":
                self.append_message(content, role="assistant")
                yield ChatResponse(role="assistant", content=content)
            if (
                response.message.tool_calls is not None
                and response.finish_reason == "tool_calls"
            ):
                for call in response.message.tool_calls:
                    f = call.function.arguments
                    async for res in self.call_tool(call.function.name, f):
                        yield res


class Gemini(OpenAI):
    @staticmethod
    def _default_model() -> str:
        return "gemini-1.5-flash"

    @staticmethod
    def _default_provider_client(config):
        return OpenAIClient(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.environ.get("GEMINI_API_KEY", config.api_key),
        )

    async def chat(
        self, msg: str, tool: Optional[str] = None
    ) -> Iterator[ChatResponse]:
        if len(self.messages) == 0:
            self.append_message(self.config.system, role="system")
        self.append_message(msg, tool=tool)
        r = self.provider_client.chat.completions.create(
            messages=self.messages,
            model=self.config.model,
            tools=self.tools,
            max_tokens=self.config.max_tokens,
        )
        for response in r.choices:
            self.logger.debug(response)
            if response.message is None:
                continue
            content = response.message.content
            if content is not None and content != "":
                self.append_message(content, role="assistant")
                yield ChatResponse(role="assistant", content=content)

            if response.message.tool_calls is not None:
                for call in response.message.tool_calls:
                    f = call.function.arguments
                    async for res in self.call_tool(call.function.name, f):
                        yield res

            # TODO: remove this, checking `toolCalls` is only required for now because of a bug
            # in the Gemini OpenAI compatiblitiy
            # see https://discuss.ai.google.dev/t/two-tool-calling-bugs-i-found-in-openai-compatibility-beta/58174
            if (
                hasattr(response.message, "toolCalls")
                and response.message.toolCalls is not None
            ):
                for call in response.message.toolCalls:
                    f = call["function"]["arguments"]
                    async for res in self.call_tool(call["function"]["name"], f):
                        yield res


class Claude(ChatProvider):
    """
    Chat using the Claude API
    """

    @staticmethod
    def _convert_tool(tool):
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }

    @staticmethod
    def _default_model():
        return "claude-3-5-sonnet-latest"

    @staticmethod
    def _default_provider_client(config):
        return AsyncAnthropic(base_url=config.base_url, api_key=config.api_key)

    async def chat(
        self, msg: str, tool: Optional[str] = None
    ) -> Iterator[ChatResponse]:
        self.append_message(msg, tool=tool)
        res = await self.provider_client.messages.create(
            max_tokens=self.config.max_tokens,
            messages=self.messages,
            model=self.config.model,
            tools=self.tools,
            system=self.config.system,
        )
        for block in res.content:
            self.logger.debug(block)
            if block.type == "tool_use" and res.stop_reason == "tool_use":
                async for res in self.call_tool(block.name, block.input):
                    yield res
            elif block.type == "text":
                self.append_message(block.text, role="assistant")
                yield ChatResponse(role="assistant", content=block.text)


def _detect_provider() -> ChatProvider:
    if "ANTHROPIC_API_KEY" in os.environ:
        return Claude
    elif "OPENAI_API_KEY" in os.environ:
        return OpenAI
    elif "GEMINI_API_KEY" in os.environ:
        return Gemini
    else:
        return Ollama


class Chat:
    """
    A unified interface for chatting with ChatProviders
    """

    provider: ChatProvider

    def __init__(
        self,
        provider: Callable[..., ChatProvider] | ChatProvider = _detect_provider(),
        *args,
        **kw,
    ):
        if isinstance(provider, ChatProvider):
            self.provider = provider
        else:
            config = None
            if len(args) > 0 and isinstance(args[0], ChatConfig):
                config = args[0]
            elif "config" in kw and isinstance(args[0], ChatConfig):
                config = kw["config"]
            else:
                config = ChatConfig(*args, **kw)
            self.provider = provider(config=config)

    @property
    def client(self):
        """
        mcp.run client
        """
        return self.provider.config.client

    def clear_history(self):
        """
        Clear chat history
        """
        self.provider.clear_history()

    async def send_message(self, msg: str) -> Iterator[ChatResponse]:
        """
        Send a chat message to the LLM, returning an iterator of ChatResponse
        """
        self.provider.get_tools()
        async for res in self.provider.chat(msg):
            yield res

    def send_message_sync(self, msg) -> Iterator[ChatResponse]:
        """
        Send a chat message to the LLM, returning an iterator of ChatResponse
        """
        for res in asyncio.run(self.send_message(msg)):
            yield res
