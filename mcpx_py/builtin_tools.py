from typing import Any
from mcpx_pydantic_ai import mcp_run


SEARCH = mcp_run.Tool(
    name="mcp_run_search_servlets",
    description="""
    Search for tools that might help solve the user's problem. Use single-word searches first (like "image" or "pdf"). If no results match, try one more related word. Never combine multiple terms in the first search. For each result found, tell the user to visit https://mcp.run/{owner}/{name} to install it.If no tools are found, suggest the user create one at mcp.run.
    """,
    input_schema={
        "type": "object",
        "properties": {
            "q": {
                "type": "string",
                "description": """
              The query of terms to search the mcp.run API for servlets.
              This query string supports:

                 * Regular word search: 'fetch markdown'  (finds documents containing both words)
                 * Phrase search: '"hello world"' (finds exact phrase)
                 * Prefix search: 'fetch*' (finds 'fetch', 'fetching', etc.)
                 * Mixed search: 'api "hello world"'
                 * Negation: '!javascript' (excludes documents with 'javascript')
              """,
            },
        },
        "required": [],
    },
)

GET_PROFILES = mcp_run.Tool(
    name="mcp_run_get_profiles",
    description="""
    List all profiles for the current user.
    """,
    input_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
)


SET_PROFILE = mcp_run.Tool(
    name="mcp_run_set_profile",
    description="""
    Set the active profile
    """,
    input_schema={
        "type": "object",
        "properties": {
            "profile": {
                "type": "string",
                "description": """The name of the profile to set as active""",
            },
        },
        "required": ["profile"],
    },
)


CURRENT_PROFILE = mcp_run.Tool(
    name="mcp_run_current_profile",
    description="""
    Get current profile name
    """,
    input_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
)

TOOLS = [SEARCH, GET_PROFILES, SET_PROFILE, CURRENT_PROFILE]
