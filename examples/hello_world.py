from mcpx_py import Client  # Import the mcp.run client

client = Client()  # Create the client, this will check the
# default location for the mcpx config or
# the `MCPX_SESSION_ID` env var can be used
# to specify a valid mcpx session id

# Call a tool with the given input
results = client.call("eval-js", {"code": "'Hello, world!'"})

# Iterate over the results
for content in results.content:
    print(content.text)
