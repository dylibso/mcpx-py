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
    base: str

    @property
    def installations(self):
        return f"{self.base}/api/profiles/~/default/installations"

    def content(self, addr):
        return f"{self.base}/api/c/{addr}"


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict


@dataclass
class Install:
    name: str
    slug: str
    binding_id: str
    content_addr: str
    settings: dict
    tools: Dict[str, Tool]
    content: bytes | None = None


@dataclass
class Content:
    type: str
    mime_type: str = "text/plain"
    _data: str | None = None

    @property
    def text(self):
        return self._data or ""


class InstalledPlugin:
    _install: Install
    _plugin: ext.Plugin

    def __init__(self, install, plugin):
        self._install = install
        self._plugin = plugin

    def call(self, tool: str | None = None, input: dict = {}):
        if tool is None:
            tool = self._install.name
        print("XXX", input)
        j = json.dumps({"params": {"arguments": input, "name": tool}})
        r = self._plugin.call("call", j)
        print("XXX", r)
        r = json.loads(r)

        out = []
        for c in r["content"]:
            ty = c["type"]
            if ty == "text":
                out.append(Content(type=ty, _data=c["text"]))
            elif ty == "image":
                out.append(
                    Content(
                        type=ty,
                        _data=base64.b64decode(c["data"]),
                        mime_type=c["mimeType"],
                    )
                )
        return out


def _parse_mcpx_config(filename: str | Path) -> str | None:
    with open(filename) as f:
        j = json.loads(f.read())
        auth: str = j["authentication"][0][1]
        s = auth.split("=", maxsplit=1)
        return s[1]
    return None


def _default_session_id() -> str:
    id = os.environ.get("MCPX_SESSION_ID")
    if id is not None:
        return id

    user = Path(os.path.expanduser("~"))
    dot_config = user / ".config" / "mcpx" / "config.json"

    path = os.environ.get("MCPX_CONFIG")
    if path is not None:
        return _parse_mcpx_config(path)

    if dot_config.exists():
        return _parse_mcpx_config(dot_config)
    raise Exception("No mcpx session ID found")


@dataclass
class Config:
    base_url: str = "https://www.mcp.run"
    tool_refresh_time: timedelta = timedelta(minutes=1)


class Cache[K, T]:
    items: Dict[K, T]
    duration: timedelta
    last_update: datetime | None = None

    def __init__(self, t: timedelta):
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
        if self.last_update is None:
            return True
        now = datetime.now()
        return now - self.last_update >= self.duration


class Client:
    config: Config
    session_id: str
    endpoints: Endpoints
    install_cache: Cache[str, Install]

    def __init__(
        self, session_id: str = _default_session_id(), config: Config | None = None
    ):
        if config is None:
            config = Config()
        self.session_id = session_id
        self.endpoints = Endpoints(config.base_url)
        self.install_cache = Cache(config.tool_refresh_time)

    def list_installs(self) -> Iterator[Install]:
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
            t = {}
            for tool in tools:
                t[tool["name"]] = Tool(
                    name=tool["name"],
                    description=tool["description"],
                    input_schema=tool["inputSchema"],
                )
            install = Install(
                binding_id=binding["id"],
                content_addr=binding["contentAddress"],
                name=install["name"],
                slug=install["servlet"]["slug"],
                settings=install["settings"],
                tools=t,
            )
            yield install

    @property
    def installs(self) -> Dict[str, Install]:
        if self.install_cache.needs_refresh():
            for install in self.list_installs():
                self.install_cache.add(install.name, install)
        return self.install_cache.items

    @property
    def tools(self) -> Dict[str, Tuple[Install, Tool]]:
        installs = self.installs
        tools = {}
        for install in installs.values():
            for tool in install.tools.values():
                tools[tool.name] = (install, tool)
        return tools

    def plugin(
        self,
        install: Install,
        wasi: bool = True,
        functions: List[ext.Function] | None = None,
    ) -> InstalledPlugin:
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
        return InstalledPlugin(
            install, ext.Plugin(manifest, wasi=wasi, functions=functions)
        )

    def call(
        self,
        tool: str,
        input: dict = {},
        wasi: bool = True,
        functions: List[ext.Function] | None = None,
    ):
        install, t = self.tools[tool]
        plugin = self.plugin(install, wasi=wasi, functions=functions)
        return plugin.call(tool=tool, input=input)
