from mcpx_pydantic_ai import mcp_run


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

TOOLS = [GET_PROFILES, SET_PROFILE, CURRENT_PROFILE]
