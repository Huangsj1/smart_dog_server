from openai import OpenAI

class OpenAIEngine:
    def __init__(self, llm_config: dict):
        self.api_key = llm_config["api_key"]
        self.base_url = llm_config["base_url"]
        # 大模型对话客户端
        self.llm_client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.model = llm_config["model"]
        self.max_tokens = llm_config["max_tokens"]

    async def chat(self, messages: list, tools=None, stream=False):
        """
        调用LLM的chat接口对话
        """
        params = {
            "model": self.model,
            "messages": messages,
            "tool_choice": "auto" if tools else "none",
            "stream": stream,
            "max_tokens": self.max_tokens
        }
        if tools:
            params["tools"] = tools

        return self.llm_client.chat.completions.create(**params)


    async def chat_stream(self, messages: list, tools=None):
        """
        调用LLM的chat接口流式对话
        """
        return await self.chat(messages=messages, tools=tools, stream=True)
