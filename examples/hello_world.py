from mcpx import Client

client = Client()
results = client.call("eval-js", {"code": "'Hello, world!'"})
for content in results:
    print(content.text)
