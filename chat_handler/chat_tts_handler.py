import json
import yaml
from pathlib import Path
import threading
import queue
import re
from typing import List
import asyncio
from my_tts.audio_player import AudioPlayer
from my_asr.audio_record import AudioRecord
from chat_handler.chat_context_manager import ChatContextManager


class ChatTTSHandler:
    def __init__(self, openai_engine, mcp_client, tts_engine, asr_engine,
                 whitelist_path=None, max_context_tokens=64000, system_role="ai_assistant"):
        # LLM相关组件
        self.llm = openai_engine
        self.mcp_client = mcp_client
        self.system_role = system_role

        # TTS相关组件
        self.tts_engine = tts_engine
        self.audio_player = AudioPlayer()

        # ASR相关组件
        self.asr_engine = asr_engine
        self.audio_recorder = AudioRecord()

        # 线程通信队列
        #  用户输入input；llm读取input并输出到message；tts读取messages中的句子并转换为音频输出到audio
        self.input_queue = queue.Queue()
        self.message_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        # 线程控制
        self.llm_thread = None
        self.audio_thread = None
        self.should_stop = threading.Event()

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
        self.tokens_used = {"prompt_cached_tokens": 0, "prompt_miss_tokens": 0, "prompt_tokens": 0,
                            "completion_tokens": 0}

        # websocket：用于发送音频到客户端
        self.websocket = None

    # 在 ChatHandler 类中更新 initialize 方法
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

            # 获取角色的描述
            role_description = config['roles'][self.system_role]
            tools_description = config['tools']['mcp_tools']
            format_description = config['format']['Chinese']
            total_description = f"{role_description}\n{tools_description}\n{format_description}"
            print(total_description)
            self.history = [{"role": "system", "content": total_description}]

    def split_sentences(self, text: str) -> (List[str], str):
        """
        将文本拆分为句子，返回完整的句子列表和可能的未完成句子
        """
        # 使用正则表达式匹配句子结束标记
        sentence_ends = re.finditer(r'[.!?。！？\n]+', text)

        last_end = 0
        sentences = []

        for match in sentence_ends:
            end = match.end()
            sentence = text[last_end:end].strip()
            if sentence:
                sentences.append(sentence)
            last_end = end

        # 检查是否有未完成的句子
        remaining = text[last_end:].strip()

        return sentences, remaining

    # llm线程-------------------------------------------------------------------------------------------
    async def llm_worker(self):
        """
        LLM处理线程的工作函数
        """
        print("[LLM Thread] 启动")
        while not self.should_stop.is_set():
            try:
                # 非阻塞获取输入，超时后继续循环检查 should_stop 标志
                user_input = self.input_queue.get(timeout=0.5)

                # 将用户输入添加到历史记录
                self.history.append({"role": "user", "content": user_input})

                # llm循环处理当前输入，直到没有工具调用为止
                while True:
                    # 调用LLM API
                    response_message, finish_reason, tokens_used = await self._call_llm_stream()

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

                    # 3.1 如果LLM没有工具调用，标记当前对话消息流完成，并且精简消息
                    if finish_reason != "tool_calls":
                        self.history.append({
                            "role": "assistant",
                            "content": response_message.content,
                        })

                        # 标记消息流已完成（主线程就会结束此轮对话）
                        self.message_queue.put(None)
                        # 标记输入队列任务完成
                        self.input_queue.task_done()

                        # 精简消息
                        self.history = await self.context_manager.manage_context(self.history, self.message_tokens)
                        break

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

            except queue.Empty:
                # 队列为空，继续循环
                continue
            except Exception as e:
                print(f"[LLM Thread] 错误: {e}")
                self.message_queue.put(f"处理错误: {str(e)}")
                self.message_queue.put(None)  # 标记完成
                self.input_queue.task_done()

        print("[LLM Thread] 关闭")

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

        # 处理流式响应（不需要print输出，主线程会输出）
        # print("LLM: ", end="", flush=True)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta

                # 收集内容片段
                if delta.content:
                    response_content += delta.content
                    # 流式输出
                    # print(delta.content, end="", flush=True)
                    # 同时还将内容片段放入到消息队列
                    self.message_queue.put(delta.content)

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

        # print()  # 换行

        # 构造响应消息
        response_message = type('obj', (object,), {
            'content': response_content,
            'tool_calls': tool_calls if tool_calls else None
        })

        return response_message, finish_reason, tokens_used


    # 音频播放线程--------------------------------------------------------------------------------------
    async def audio_worker(self):
        """
        音频播放线程的工作函数
        如果对话不停止，该线程一直存在，即一直处理对话
        """
        print("[Audio Thread] 启动")
        while not self.should_stop.is_set():
            try:
                # 非阻塞获取音频数据
                audio_data = self.audio_queue.get(timeout=0.5)

                # 播放音频
                self.audio_player.play_audio(audio_data)

                # 标记音频队列任务完成
                self.audio_queue.task_done()
            except queue.Empty:
                # 队列为空，继续循环
                continue
            except Exception as e:
                print(f"[Audio Thread] 错误: {e}")

        print("[Audio Thread] 关闭")


    async def start(self, system_role_path=None, websocket=None):
        """启动处理器，初始化并创建工作线程"""
        # 设置websocket
        self.websocket = websocket

        # 初始化
        await self.initialize(system_role_path)

        # 创建并启动线程
        self.should_stop.clear()

        # 创建LLM处理线程
        self.llm_thread = threading.Thread(target=lambda: asyncio.run(self.llm_worker()))
        self.llm_thread.daemon = True   # 守护线程：主线程退出时自动结束
        self.llm_thread.start()

        # 创建音频播放线程（如果没有websocket的话）；如果有websocket就会在直接将音频发送给客户端
        if not self.websocket:
            self.audio_thread = threading.Thread(target=lambda: asyncio.run(self.audio_worker()))
            self.audio_thread.daemon = True  # 守护线程：主线程退出时自动结束
            self.audio_thread.start()

        print("ChatHandler 已启动")

    async def stop(self):
        """停止处理器和相关线程"""
        self.should_stop.set()

        # 等待线程结束
        if self.llm_thread and self.llm_thread.is_alive():
            self.llm_thread.join(timeout=5)

        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=5)

        print("ChatHandler 已停止")


    async def chat_with_tts(self, user_input: str):
        """
        对话的逻辑函数：
        1.将用户输入放入输入队列
        2.循环从消息队列读取LLM的流式回复（命令行输出 + 供音频输出）
        3.构建完整的句子，将其转换为音频并放入音频队列，供音频播放线程处理；或者通过websocket发送给客户端
        """
        if not self.tts_engine:
            raise ValueError("TTS模型未初始化")

        print("LLM: ", end="", flush=True)

        # 将用户输入放入输入队列
        self.input_queue.put(user_input)

        # 从消息队列读取LLM的流式回复
        message_buffer = ""

        while True:
            try:
                # 获取一个消息片段
                chunk = self.message_queue.get(timeout=10)

                # 如果收到None，表示流结束（当llm_worker在处理完此轮对话后会发送None）
                if chunk is None:
                    # 处理最后剩余的文本
                    if message_buffer:
                        audio_data = await self.tts_engine.text_to_speech(message_buffer)
                        await self._handle_audio_data(audio_data)

                    self.message_queue.task_done()
                    break

                # 累积消息
                message_buffer += chunk
                print(chunk, end="", flush=True)

                # 尝试拆分句子
                sentences, remaining = self.split_sentences(message_buffer)

                # 处理完整的句子
                if sentences:
                    # 为每个完整的句子生成语音
                    for sentence in sentences:
                        audio_data = await self.tts_engine.text_to_speech(sentence)
                        await self._handle_audio_data(audio_data)

                # 保存剩余的不完整句子
                message_buffer = remaining

                self.message_queue.task_done()

            except queue.Empty:
                print("\n[警告] 等待LLM响应超时")
                break

        print()  # 换行，保持输出整洁

        # 等待所有音频播放完毕（即使一直为空也能join）
        self.audio_queue.join()

    async def _handle_audio_data(self, audio_data: bytes):
        """
        处理音频数据，将其发送到WebSocket或放入音频队列
        如果有WebSocket连接，则直接发送音频数据到客户端
        否则，将音频数据放入音频队列供音频播放线程处理
        """
        if audio_data:
            if self.websocket:
                # 如果有WebSocket连接，直接发送音频数据
                await self.websocket.send_bytes(audio_data)
            else:
                # 否则，将音频数据放入音频队列
                self.audio_queue.put(audio_data)

    # 交互式对话循环-----------------------------------------------------------------------------------
    async def interactive_loop_with_tts(self):
        """交互式对话循环，带TTS功能"""
        try:
            while True:
                user_input = input("\nYou: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("退出对话...")
                    break

                # await self.chat_with_tts(user_input)
                await self.chat_with_tts(user_input)
        finally:
            await self.stop()

    async def interactive_loop_with_tts_asr(self):
        """
        交互式对话循环，带ASR、TTS功能
        """
        try:
            while True:
                # 使用ASR录音并转换为文本
                audio_file_path = self.audio_recorder.record_audio()
                print("--start audio -> text--")
                user_input = self.asr_engine.audio_to_text(audio_file_path)
                print(f"\nYou: {user_input}")
                # 下面操作对语音输入没用，对键盘输入有用
                if user_input.lower() in ["exit", "quit"]:
                    print("退出对话...")
                    break

                await self.chat_with_tts(user_input)

                # 需要清除临时文件
                self.audio_recorder.cleanup()
        finally:
            await self.stop()

    # 交互式单次对话-----------------------------------------------------------------------------------
    async def interactive_with_audio_input(self, audio_file_path: str):
        """
        单次交互式对话，带音频输入，用于和客户端交互
        最后不需要stop中止，而是等到websocket断开后才中止
        param:
            audio_file_path: 音频文件路径
        """
        # 使用ASR将音频转换为文本
        user_input = self.asr_engine.audio_to_text(audio_file_path)
        print(f"\nYou: {user_input}")

        await self.chat_with_tts(user_input)

    async def interactive_with_text_input(self, input_text: str):
        """
        单次交互式对话，带音频输入，用于和客户端交互
        最后不需要stop中止，而是等到websocket断开后才中止
        param:
            input_text: 输入文本
        """
        # 使用ASR将音频转换为文本
        print(f"\nYou: {input_text}")

        await self.chat_with_tts(input_text)

        print("对话已完成，等待下一次输入...")



    # 工具相关------------------------------------------------------------------------------------------
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
