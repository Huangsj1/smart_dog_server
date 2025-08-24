import json
import yaml
from pathlib import Path

from chat_handler.chat_context_manager import ChatContextManager

class ChatHandler:
    def __init__(self, openai_engine, mcp_client, whitelist_path=None, max_context_tokens=64000):
        self.llm = openai_engine
        self.mcp_client = mcp_client

        # 工具列表
        self.tools = None
        # 工具白名单配置
        self.tools_whitelist = self._load_tools_whitelist(whitelist_path)

        # 上下文
        self.history = []
        # 上下文管理器
        self.context_manager = ChatContextManager(
            llm_engine=openai_engine,
            max_context_tokens=max_context_tokens
        )
        # 当前累积message的tokens数量（直接从本轮对话的response的prompt_token中读取）
        self.message_tokens = 0
        # 累积使用的tokens计数（将每轮对话的tokens都加起来）
        self.tokens_used = {"prompt_cached_tokens": 0, "prompt_miss_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

    async def initialize(self, system_role_path=None):
        """
        初始化聊天处理器，准备必要的工具和上下文。
        """
        # 准备工具列表
        self.tools = await self.prepare_tools()
        # 初始化系统提示
        self.init_system_prompt(system_role_path)

    def init_system_prompt(self, system_role_path: str):
        if system_role_path:
            # 加载 YAML 配置文件
            with open("chat_handler/system_role_prompt.yaml", "r") as file:
                config = yaml.safe_load(file)

            # 获取角色A的描述
            role_description = config['roles']['SpongeBob']
            tools_description = config['tools']['mcp_tools']
            total_description = f"{role_description}\n{tools_description}"
            print(total_description)
            self.history.append({"role": "system", "content": total_description})


    async def handle_chat(self, user_input: str, use_stream=False):
        """
        进行一轮对话，处理用户输入并返回LLM的最终回复。
        参数:
            user_input: 用户输入的文本
            use_stream: 是否使用流式输出
        """
        # 1. 将用户输入添加到历史记录
        self.history.append({"role": "user", "content": user_input})

        # 2. 获取工具列表（初始化的时候获取过一次，这里为了保险）
        if not self.tools:
            self.tools = await self.prepare_tools()

        while True:
            # 3. 调用LLM API (根据use_stream参数决定是否使用流式调用)
            if use_stream:
                response_message, finish_reason, tokens_used = await self._call_llm_stream()
            else:
                response_message, finish_reason, tokens_used = await self._call_llm_normal()

            # 更新token计数器
            if tokens_used:
                self.message_tokens = getattr(tokens_used, 'prompt_tokens', 0)
                self.tokens_used['prompt_cached_tokens'] += getattr(tokens_used, 'prompt_cache_hit_tokens', 0)
                self.tokens_used['prompt_miss_tokens'] += getattr(tokens_used, 'prompt_cache_miss_tokens', 0)
                self.tokens_used['prompt_tokens'] += getattr(tokens_used, 'prompt_tokens', 0)
                self.tokens_used['completion_tokens'] += getattr(tokens_used, 'completion_tokens', 0)

            # 输出token统计信息
            if tokens_used:
                print(
                    f"\nCurrent tokens used: [prompt_cached_tokens]: {getattr(tokens_used, 'prompt_cache_hit_tokens', 0)}"
                    f" - [prompt_miss_tokens]: {getattr(tokens_used, 'prompt_cache_miss_tokens', 0)}"
                    f" - [prompt_tokens]: {getattr(tokens_used, 'prompt_tokens', 0)}"
                    f" - [completion_tokens]: {getattr(tokens_used, 'completion_tokens', 0)}")
                print(f"Total tokens used: [prompt_cached_tokens]: {self.tokens_used['prompt_cached_tokens']}"
                      f" - [prompt_miss_tokens]: {self.tokens_used['prompt_miss_tokens']}"
                      f" - [prompt_tokens]: {self.tokens_used['prompt_tokens']}"
                      f" - [completion_tokens]: {self.tokens_used['completion_tokens']}")

            # 3.1 如果LLM没有工具调用，则直接返回结果
            if finish_reason != "tool_calls":
                self.history.append({
                    "role": "assistant",
                    "content": response_message.content,
                })
                return response_message

            # 3.2 否则，LLM请求工具调用
            print("🤖 LLM requested tool calls...")
            # i. 将LLM的回复添加到历史记录
            self.history.append({
                "role": "assistant",
                "content": response_message.content,
                # 这里一定要有 tool_calls 字段，里面包含了LLM请求的所有工具调用的详细信息，
                #  否则下面调用完工具放入message 的tool字段中会出错
                "tool_calls": response_message.tool_calls
            })

            # ii. 执行所有工具调用
            await self._process_tool_calls(response_message.tool_calls)

            # 带着工具调用的结果再次请求LLM进行总结，循环继续
            print("🔄 Sending tool results back to LLM for final response...")

    async def _call_llm_normal(self):
        """
        调用LLM API的标准方式
        """
        response = await self.llm.chat(self.history, self.tools)
        response_message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        return response_message, finish_reason, response.usage


    async def _call_llm_stream(self):
        """
        调用LLM API的流式方式
        """
        stream = await self.llm.chat_stream(self.history, self.tools)

        # 用于累积完整响应
        response_content = ""
        tool_calls = []
        finish_reason = None
        tokens_used = None

        # 处理流式响应
        print("LLM: ", end="", flush=True)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta

                # 收集内容片段
                if delta.content:
                    response_content += delta.content
                    print(delta.content, end="", flush=True)

                # 收集工具调用信息
                if delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        # 查找或创建工具调用
                        existing_call = next((tc for tc in tool_calls if tc.index == tool_call_delta.index), None)
                        if existing_call is None:
                            tool_calls.append(tool_call_delta)
                        else:
                            # 更新现有工具调用
                            if tool_call_delta.function and tool_call_delta.function.name:
                                existing_call.function.name = tool_call_delta.function.name
                            if tool_call_delta.function and tool_call_delta.function.arguments:
                                existing_call.function.arguments = (
                                                                               existing_call.function.arguments or "") + tool_call_delta.function.arguments
                            if tool_call_delta.id:
                                existing_call.id = tool_call_delta.id

            # 检查完成原因
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
                tokens_used = chunk.usage

        print()  # 换行

        # 构造响应消息
        response_message = type('obj', (object,), {
            'content': response_content,
            'tool_calls': tool_calls if tool_calls else None
        })

        return response_message, finish_reason, tokens_used

    async def _process_tool_calls(self, tool_calls):
        """
        处理工具调用
        """
        for tool_call in tool_calls:
            print(f"  - Calling tool: {tool_call.function.name}")
            try:
                # 使用MCP管理器执行调用
                tool_result = await self.mcp_client.client.call_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )

                print(f"✅ Tool call successful: {tool_call.function.name}, arguments: {tool_call.function.arguments},  Result: {tool_result.content[0].text}")

                # 将工具调用的结果添加回历史记录
                self.history.append({
                    "role": "tool",
                    "content": tool_result.content[0].text,
                    "tool_call_id": tool_call.id
                })
            except Exception as e:
                # 捕获异常并记录错误
                error_message = f"工具调用失败: {tool_call.function.name}, 错误: {str(e)}"
                print(f"❌ {error_message}")

                # 将错误信息作为工具响应添加到历史记录中
                self.history.append({
                    "role": "tool",
                    "content": error_message,
                    "tool_call_id": tool_call.id
                })

    async def prepare_tools(self):
        """
        准备工具列表，将MCP客户端的工具转换为OpenAI API所需的格式
        并根据白名单过滤工具
        """
        tools = await self.mcp_client.client.list_tools()
        # 根据白名单过滤
        filtered_tools = [
            tool for tool in tools if self._is_tool_allowed(tool.name)
        ]
        print(f"总工具数: {len(tools)}, 过滤后工具数: {len(filtered_tools)}")

        structured_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            for tool in filtered_tools
        ]

        for tool in filtered_tools:
            print(f"tool: {tool}")

        return structured_tools

    def _load_tools_whitelist(self, whitelist_path):
        """加载工具白名单配置文件"""
        if not whitelist_path or not Path(whitelist_path).exists():
            return None

        with open(whitelist_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _is_tool_allowed(self, tool_name: str):
        """检查工具是否在白名单中"""
        if not self.tools_whitelist:
            return True  # 如果没有白名单，则允许所有工具

        # 提取服务器名称（工具名称格式为 "server-service_tool_name"）
        #  根据第一个 '_' 来分割
        server_name, function_name = tool_name.split('_', maxsplit=1)

        # 检查服务器是否在白名单中且已启用
        if (server_name in self.tools_whitelist.get('mcp_servers', {}) and
                self.tools_whitelist['mcp_servers'][server_name].get('enabled', False)):

            # 如果允许所有
            if self.tools_whitelist['mcp_servers'][server_name].get('allow_all', False):
                return True

            # 检查具体工具是否在白名单中
            allowed_tools = self.tools_whitelist['mcp_servers'][server_name].get('tools', [])
            return function_name in allowed_tools

        return False

    async def loop(self, use_stream=False):
        """
        处理对话循环，等待用户输入并进行对话；
        一轮对话结束后，自动精简上下文以保持上下文的有效性和简洁性。
        参数:
            use_stream: 是否使用流式输出
        """
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat...")
                break

            response = await self.handle_chat(user_input, use_stream=use_stream)

            # 如果不是流式模式，需要打印输出
            if not use_stream:
                print(f"LLM: {response.content}")

            # 本轮对话结束后，精简上下文
            self.history = await self.context_manager.manage_context(self.history, self.message_tokens)


    # 为了向后兼容，保留原来的方法
    async def handle_chat_stream(self, user_input: str):
        """
        进行一轮对话，处理用户输入并返回LLM的最终回复(流式处理)。
        """
        return await self.handle_chat(user_input, use_stream=True)

    async def loop_stream(self):
        """
        处理对话循环，等待用户输入并进行对话，使用流式输出。
        """
        await self.loop(use_stream=True)