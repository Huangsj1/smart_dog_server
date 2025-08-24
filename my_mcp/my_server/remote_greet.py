from fastmcp import FastMCP

mcp = FastMCP(name="GreetServer")

@mcp.tool()
def greet():
    """
    Greets the user with a friendly message.

    This tool does not take any input and returns a simple greeting message.

    Example:
        >>> greet()
        "Hello, welcome to the Greet Server!"
    """
    return "Hello, welcome to the Greet Server!"

if __name__ == "__main__":
  # mcp.run()
  mcp.run(transport="http", host="127.0.0.1", port=9000)