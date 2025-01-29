from mcpx_py import Client  # Import the mcp.run client

client = Client()  # Create the client, this will check the
# default location for the mcpx config or
# the `MCPX_SESSION_ID` env var can be used
# to specify a valid mcpx session id

results = client.search("fetch")  # Search for servlets that mention the
# word "fetch"

# Iterate through results and print the slug
for result in results:
    print(result["slug"])
