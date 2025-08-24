import yaml
from fastmcp import Client

class MCPClientManager:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        # my_mcp client 实例（一个mcp client 对应 所有 my_mcp servers）
        self.server_configs = {
            "mcpServers": config.get("mcp_servers", {})
        }
        # 使用转换后的配置初始化 Client
        self.client = Client(self.server_configs)
