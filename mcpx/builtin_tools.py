from .client import Tool

SEARCH = Tool(
    name="mcp_run_search_servlets",
    description="""
    Search for tools that might help solve the user's problem. Use single-word searches first (like "image" or "pdf"). If no results match, try one more related word. Never combine multiple terms in the first search.

    For each result found, tell the user to visit https://mcp.run/{owner}/{name} to install it.

    If no tools are found, suggest the user create one at mcp.run.

    Search proactively - don't wait for users to ask about tools. Anytime you're helping with a task that external code could enhance, do a search.
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
