from dataclasses import dataclass
import os
import json
from pathlib import Path
from typing import Iterator, Dict, List, Optional
from datetime import datetime, timedelta
import base64
import logging
import urllib

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

    def installations(self, profile):
        """
        List installations
        """
        if "/" in profile:
            return f"{self.base}/api/profiles/{profile}/installations"
        return f"{self.base}/api/profiles/~/{profile}/installations"

    def search(self, query):
        """
        Search servlets
        """
        query = urllib.parse.quote_plus(query)
        return f"{self.base}/api/servlets?q={query}"

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

    servlet: Optional["Servlet"] = None
    """
    The servlet the tool belongs to
    """


@dataclass
class Servlet:
    """
    An mcpx servlet
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

    installed: bool
    """
    Marks whether the servlet is installed
    """

    tools: Dict[str, Tool]
    """
    All tools provided by the servlet
    """

    content: bytes | None = None
    """
    Cached WASM module data
    """

    def __eq__(self, other):
        if other is None:
            return False
        return (
            self.tools == other.tools
            and self.settings == other.settings
            and self.content_addr == other.content_addr
            and self.binding_id == other.binding_id
            and self.slug == other.slug
            and self.name == other.name
        )


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
    def json(self):
        """
        Get the result data as json
        """
        return json.loads(self.text)

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
    id = os.environ.get("MCP_RUN_SESSION_ID", os.environ.get("MCPX_SESSION_ID"))
    if id is not None:
        return id

    # Try ~/.config/mcpx/config.json for Linux/macOS
    user = Path(os.path.expanduser("~"))
    dot_config = user / ".config" / "mcpx" / "config.json"
    if dot_config.exists():
        return _parse_mcpx_config(dot_config)

    # Try Windows paths
    windows_config = Path(os.path.expandvars("%LOCALAPPDATA%/mcpx/config.json"))
    if windows_config.exists():
        return _parse_mcpx_config(windows_config)

    windows_config = Path(os.path.expandvars("%APPDATA%/mcpx/config.json"))
    if windows_config.exists():
        return _parse_mcpx_config(windows_config)

    raise Exception("No mcpx session ID found")


def _default_update_interval():
    ms = os.environ.get("MCPX_UPDATE_INTERVAL")
    if ms is None:
        return timedelta(minutes=1)
    else:
        return timedelta(milliseconds=int(ms))


@dataclass
class ClientConfig:
    """
    Configures an mcp.run Client
    """

    base_url: str = os.environ.get("MCP_RUN_ORIGIN", "https://www.mcp.run")
    """
    mcp.run base URL
    """

    tool_refresh_time: timedelta = _default_update_interval()
    """
    Length of time to wait between checking for new tools
    """

    logger: logging.Logger = logging.getLogger(__name__)
    """
    Python logger
    """

    profile: str = "default"
    """
    mcp.run profile name
    """

    def configure_logging(self, *args, **kw):
        """
        Configure logging using logging.basicConfig
        """
        return logging.basicConfig(*args, **kw)


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
        self.items.pop(key, None)

    def get(self, key: K) -> T | None:
        return self.items.get(key)

    def __contains__(self, key: K) -> bool:
        return key in self.items

    def clear(self):
        self.items = {}
        self.last_update = datetime.now()

    def set_last_update(self):
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

    logger: logging.Logger
    """
    Python logger
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

    last_installations_request: str | None = None
    """
    Date header from last installations request
    """

    def __init__(
        self,
        session_id: str | None = None,
        config: ClientConfig | None = None,
        log_level: int | None = None,
    ):
        if session_id is None:
            session_id = _default_session_id()
        if config is None:
            config = ClientConfig()
        self.session_id = session_id
        self.endpoints = Endpoints(config.base_url)
        self.install_cache = Cache(config.tool_refresh_time)
        self.plugin_cache = Cache()
        self.logger = config.logger
        self.config = config

        if log_level is not None:
            self.configure_logging(level=log_level)

    def configure_logging(self, *args, **kw):
        """
        Configure logging using logging.basicConfig
        """
        return logging.basicConfig(*args, **kw)

    def set_profile(self, profile: str):
        """
        Select a profile
        """
        self.config.profile = profile
        self.last_installations_request = None

    def list_installs(self) -> Iterator[Servlet]:
        """
        List all installed servlets, this will make an HTTP
        request each time
        """
        url = self.endpoints.installations(self.config.profile)
        self.logger.info(f"Listing installed mcp.run servlets from {url}")
        headers = {}
        if self.last_installations_request is not None:
            headers["if-modified-since"] = self.last_installations_request
        res = requests.get(
            url,
            headers=headers,
            cookies={
                "sessionId": self.session_id,
            },
        )
        res.raise_for_status()
        if res.status_code == 301:
            self.logger.debug(f"No changes since {self.last_installations_request}")
            for v in self.install_cache.items.values():
                yield v
            return
        self.last_installations_request = res.headers.get("Date")
        data = res.json()
        self.logger.debug(f"Got installed servlets from {url}: {data}")
        for install in data["installs"]:
            binding = install["binding"]
            tools = install["servlet"]["meta"]["schema"]
            if "tools" in tools:
                tools = tools["tools"]
            else:
                tools = [tools]
            install = Servlet(
                installed=True,
                binding_id=binding["id"],
                content_addr=binding["contentAddress"],
                name=install.get("name", ""),
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
            self.logger.info("Cache expired, fetching installs")
            visited = set()
            for install in self.list_installs():
                if install != self.install_cache.get(install.name):
                    self.install_cache.add(install.name, install)
                    self.plugin_cache.remove(install.name)
                visited.add(install.name)
            for install_name in self.install_cache.items:
                if install_name not in visited:
                    self.install_cache.remove(install_name)
                    self.plugin_cache.remove(install_name)
            self.install_cache.set_last_update()
        return self.install_cache.items

    @property
    def tools(self) -> Dict[str, Tool]:
        """
        Get all tools from all installed servlets
        """
        installs = self.installs
        tools = {}
        for install in installs.values():
            for tool in install.tools.values():
                tools[tool.name] = tool
        return tools

    def tool(self, name: str) -> Tool | None:
        """
        Get a tool by name
        """
        for install in self.installs.values():
            for tool in install.tools.values():
                if tool.name == name:
                    return tool
        return None

    def search(self, query: str) -> List[dict]:
        url = self.endpoints.search(query)
        res = requests.get(
            url,
            cookies={
                "sessionId": self.session_id,
            },
        )
        data = res.json()
        return data

    def plugin(
        self,
        install: Servlet,
        wasi: bool = True,
        functions: List[ext.Function] | None = None,
        wasm: List[Dict[str, bytes]] | None = None,
    ) -> InstalledPlugin:
        """
        Instantiate an installed servlet, turning it into an InstalledPlugin

        Args:
            install: The servlet to instantiate
            wasi: Whether to enable WASI
            functions: Optional list of Extism functions to include
            wasm: Optional list of additional WASM modules

        Returns:
            An InstalledPlugin instance
        """
        if not install.installed:
            raise Exception(f"Servlet {install.name} must be installed before use")
        cache_name = f"{install.name}-{wasi}"
        if functions is not None:
            for func in functions:
                cache_name += "-"
                cache_name += str(hash(func.pointer))
        cache_name = str(hash(cache_name))
        cached = self.plugin_cache.get(cache_name)
        if cached is not None:
            return cached
        if install.content is None:
            self.logger.info(
                f"Fetching servlet Wasm for {install.name}: {install.content_addr}"
            )
            res = requests.get(
                self.endpoints.content(install.content_addr),
                cookies={
                    "sessionId": self.session_id,
                },
            )
            install.content = res.content
        perm = install.settings["permissions"]
        wasm_modules = [{"data": install.content}]
        if wasm is not None:
            wasm_modules.extend(wasm)
        manifest = {
            "wasm": wasm_modules,
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
        wasm: List[Dict[str, bytes]] | None = None,
    ) -> CallResult:
        """
        Call a tool with the given input

        Args:
            tool: Name of the tool or Tool instance to call
            input: Dictionary of input parameters for the tool
            wasi: Whether to enable WASI
            functions: Optional list of Extism functions to include
            wasm: Optional list of additional WASM modules

        Returns:
            CallResult containing the tool's output
        """
        if isinstance(tool, str):
            found_tool = self.tool(tool)
            if found_tool is None:
                raise ValueError(f"Tool '{tool}' not found")
            tool = found_tool
        plugin = self.plugin(tool.servlet, wasi=wasi, functions=functions, wasm=wasm)
        return plugin.call(tool=tool.name, input=input)
