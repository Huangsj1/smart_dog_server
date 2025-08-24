from fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP(name="CalculatorServer")

@mcp.tool(
    # name="add_numbers",
    # description="Add two integers and return the result.",
    tags={"math", "calculator", "basic"}
)
def add(
    x: int = Field(10, description="the first number to add", ge=1, le=100),
    y: int = Field(10, description="the second number to add", ge=1, le=100)
) -> int:
    """
    Adds two integer numbers and returns the result.

    This tool takes two integers as input, adds them together,
    and returns the computed sum.

    Example:
        >>> add(2, 3)
        5
    """
    return x + y

@mcp.tool(
    tags={"math", "calculator", "basic"}
)
def minus(
    x: int = Field(10, description="the first number to minus", ge=1, le=100),
    y: int = Field(10, description="the second number to minus", ge=1, le=100)
) -> int:
    """
    Minus two integer numbers and returns the result.

    This tool takes two integers as input, minus them,
    and returns the computed sum.

    Example:
        >>> minus(3, 2)
        1
    """
    return x - y

if __name__ == "__main__":
  mcp.run()
  # mcp.run(transport="http", host="127.0.0.1", port=9000)