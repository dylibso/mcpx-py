from dataclasses import dataclass
import os
import json
from pathlib import Path
from typing import Iterator, Dict, List, Tuple
from datetime import datetime, timedelta
import base64

import requests
import extism as ext


@dataclass
class Endpoints:
    """
    Manages mcp.run endpoints
    """

    base: str
    """
    mcp.run base URL
    """

    @property
    def installations(self):
        """
        List installations
        """
        return f"{self.base}/api/profiles/~/default/installations"

    def content(self, addr: str):
        """
        Get the data associated with a content address
        """
        return f"{self.base}/api/c/{addr}"


@dataclass
class Tool:
    """
    A tool definition
    """

    name: str
    """
    Name of the tool
    """

    description: str
    """
    Information about the tool and how to use it
    """

    input_schema: dict
    """
    Input parameter schema
    """

    servlet: 'Servlet'
    """
    The servlet the tool belongs to
    """


@dataclass
class Servlet:
    """
    An installed servlet
    """

    name: str
    """
    Servlet name
    """

    slug: str
    """
    Servlet slug
    """

    binding_id: str
    """
    Servlet binding ID
    """

    content_addr: str
    """
    Content address for WASM module
    """

    settings: dict
    """
    Servlet settings and permissions
    """

    tools: Dict[str, Tool]
    """
    All tools provided by the servlet
    """

    content: bytes | None = None
    """
    Cached WASM module data
    """


@dataclass
class Content:
    """
    The result of tool calls
    """

    type: str
    """
    The type of content, for example "text" or "image"
    """

    mime_type: str = "text/plain"
    """
    Content mime type
    """

    _data: bytes | None = None
    """
    Result message or data
    """

    @property
    def text(self):
        """
        Get the result message
        """
        return self.data.decode()

    @property
    def data(self):
        """
        Get the result as bytes
        """
        return self._data or b""


@dataclass
class CallResult:
    """
    Result of a tool call
    """

    content: List[Content]
    """
    Content returned from a call
    """


class InstalledPlugin:
    _install: Servlet
    _plugin: ext.Plugin

    def __init__(self, install, plugin):
        self._install = install
        self._plugin = plugin

    def call(self, tool: str | None = None, input: dict = {}) -> CallResult:
        """
        Call a tool with the given input
        """
        if tool is None:
            tool = self._install.name
        j = json.dumps({"params": {"arguments": input, "name": tool}})
        r = self._plugin.call("call", j)
        r = json.loads(r)

        out = []
        for c in r["content"]:
            ty = c["type"]
            if ty == "text":
                out.append(Content(type=ty, _data=c["text"].encode()))
            elif ty == "image":
                out.append(
                    Content(
                        type=ty,
                        _data=base64.b64decode(c["data"]),
                        mime_type=c["mimeType"],
                    )
                )
        return CallResult(content=out)


def _parse_mcpx_config(filename: str | Path) -> str | None:
    with open(filename) as f:
        j = json.loads(f.read())
        auth: str = j["authentication"][0][1]
        s = auth.split("=", maxsplit=1)
        return s[1]
    return None


def _default_session_id() -> str:
    # Allow session id to be specified using MCPX_SESSION_ID
    id = os.environ.get("MCPX_SESSION_ID")
    if id is not None:
        return id

    # Allow config file to be specified using MCPX_CONFIG
    path = os.environ.get("MCPX_CONFIG")
    if path is not None:
        return _parse_mcpx_config(path)

    # Try ~/.config/mcpx/config.json for Linux/macOS
    user = Path(os.path.expanduser("~"))
    dot_config = user / ".config" / "mcpx" / "config.json"
    if dot_config.exists():
        return _parse_mcpx_config(dot_config)

    # Try Windows paths
    windows_config = os.path.expandvars("%LOCALAPPDATA%/mcpx/config.json")
    if windows_config.exists():
        return _parse_mcpx_config(windows_config)

    windows_config = os.path.expandvars("%APPDATA%/mcpx/config.json")
    if windows_config.exists():
        return _parse_mcpx_config(windows_config)

    raise Exception("No mcpx session ID found")


@dataclass
class ClientConfig:
    """
    Configures an mcp.run Client
    """

    base_url: str = "https://www.mcp.run"
    """
    mcp.run base URL
    """

    tool_refresh_time: timedelta = timedelta(minutes=1)
    """
    Length of time to wait between checking for new tools
    """


class Cache[K, T]:
    items: Dict[K, T]
    duration: timedelta
    last_update: datetime | None = None

    def __init__(self, t: timedelta | None = None):
        self.items = {}
        self.last_update = None
        self.duration = t

    def add(self, key: K, item: T):
        self.items[key] = item

    def remove(self, key: K):
        del self.items[key]

    def clear(self):
        self.items = {}
        self.last_update = datetime.now()

    def needs_refresh(self) -> bool:
        if self.duration is None:
            return False
        if self.last_update is None:
            return True
        now = datetime.now()
        return now - self.last_update >= self.duration


class Client:
    """
    mcp.run API client
    """

    config: ClientConfig
    """
    Client configuration
    """

    session_id: str
    """
    mcp.run session ID
    """

    endpoints: Endpoints
    """
    mcp.run endpoint manager
    """

    install_cache: Cache[str, Servlet]
    """
    Cache of Installs
    """

    plugin_cache: Cache[str, InstalledPlugin]
    """
    Cache of InstalledPlugins
    """

    def __init__(
        self,
        session_id: str = _default_session_id(),
        config: ClientConfig | None = None,
    ):
        if config is None:
            config = ClientConfig()
        self.session_id = session_id
        self.endpoints = Endpoints(config.base_url)
        self.install_cache = Cache(config.tool_refresh_time)
        self.plugin_cache = Cache()

    def list_installs(self) -> Iterator[Servlet]:
        """
        List all installed servlets, this will make an HTTP
        request each time
        """
        url = self.endpoints.installations
        res = requests.get(
            url,
            cookies={
                "sessionId": self.session_id,
            },
        )
        data = res.json()
        for install in data["installs"]:
            binding = install["binding"]
            tools = install["servlet"]["meta"]["schema"]
            if "tools" in tools:
                tools = tools["tools"]
            else:
                tools = [tools]
            install = Servlet(
                binding_id=binding["id"],
                content_addr=binding["contentAddress"],
                name=install["name"],
                slug=install["servlet"]["slug"],
                settings=install["settings"],
                tools={},
            )
            for tool in tools:
                install.tools[tool["name"]] = Tool(
                    name=tool["name"],
                    description=tool["description"],
                    input_schema=tool["inputSchema"],
                    servlet=install,
                )
            yield install

    @property
    def installs(self) -> Dict[str, Servlet]:
        """
        Get all installed servlets, this will returned cached Installs if
        the cache timeout hasn't been reached
        """
        if self.install_cache.needs_refresh():
            self.install_cache.clear()
            self.plugin_cache.clear()
            for install in self.list_installs():
                self.install_cache.add(install.name, install)
        return self.install_cache.items

    @property
    def tools(self) -> Dict[str, Tool]:
        """
        Get all tools and their associated Install object
        """
        installs = self.installs
        tools = {}
        for install in installs.values():
            for tool in install.tools.values():
                tools[tool.name] = tool
        return tools

    def plugin(
        self,
        install: Servlet,
        wasi: bool = True,
        functions: List[ext.Function] | None = None,
    ) -> InstalledPlugin:
        """
        Instantiate an installed servlet, turning it into an InstalledPlugin
        """
        cache_name = f"{install.name}-{wasi}"
        if functions is not None:
            for func in functions:
                cache_name += "-"
                cache_name += hash(func.pointer)
        cache_name = str(hash(cache_name))
        if cache_name in self.plugin_cache.items:
            return self.plugin_cache.items[cache_name]
        if install.content is None:
            res = requests.get(
                self.endpoints.content(install.content_addr),
                cookies={
                    "sessionId": self.session_id,
                },
            )
            install.content = res.content
        perm = install.settings["permissions"]
        manifest = {
            "wasm": [{"data": install.content}],
            "allowed_paths": perm["filesystem"].get("volumes", {}),
            "allowed_hosts": perm["network"].get("domains", []),
            "config": install.settings.get("config", {}),
        }
        if functions is None:
            functions = []
        p = InstalledPlugin(
            install, ext.Plugin(manifest, wasi=wasi, functions=functions)
        )
        self.plugin_cache.add(install.name, p)
        return p

    def call(
        self,
        tool: str | Tool,
        input: dict = {},
        wasi: bool = True,
        functions: List[ext.Function] | None = None,
    ) -> CallResult:
        """
        Call a tool with the given input
        """
        if isinstance(tool, str):\
            tool = self.tools[tool]
        plugin = self.plugin(tool.servlet, wasi=wasi, functions=functions)
        return plugin.call(tool=tool.name, input=input)
