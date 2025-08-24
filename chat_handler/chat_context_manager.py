class ChatContextManager:
    """
    聊天上下文管理器：负责管理对话历史，自动精简冗长对话，优化token使用
    """
    def __init__(self, llm_engine, max_context_tokens=64000,
                 summarize_threshold=0.5, keep_chat_rounds=5, system_prompt_maxnum=10):
        """
        初始化上下文管理器

        参数:
            llm_engine: 用于生成摘要的语言模型引擎
            max_context_tokens: 模型的最大上下文长度
            summarize_threshold: 触发精简的阈值比例
            keep_chat_rounds: 保留的最近对话轮数
            system_prompt_maxnum: 最大保存系统提示的数量，用于精简
        """
        self.llm = llm_engine
        self.max_context_tokens = max_context_tokens
        self.summarize_threshold = summarize_threshold
        self.keep_chat_rounds = keep_chat_rounds
        self.system_prompt_maxnum = system_prompt_maxnum

    async def manage_context(self, history: list, current_tokens: int) -> list:
        """
        管理对话上下文，当达到阈值时自动精简历史记录，精简后作为system_prompt保存

        参数:
            history: 完整的对话历史列表
            current_tokens: 当前上下文的tokens数量

        返回:
            经过管理的历史记录列表
        """
        # 如果当前tokens数量超过最大上下文长度的阈值，执行精简
        if current_tokens > self.max_context_tokens * self.summarize_threshold:
            print(f"⚠️ 上下文大小({current_tokens}tokens)超过阈值，开始精简历史记录...")

            # 分离系统提示、需要保留的最近消息和需要精简的旧消息
            system_prompts = [msg for msg in history if msg.get("role") == "system"]
            non_system_messages = [msg for msg in history if msg.get("role") != "system"]

            # 保留最近的几轮消息
            keep_chat_messages = self._get_keep_chat_messages(non_system_messages, self.keep_chat_rounds)
            recent_messages = non_system_messages[-keep_chat_messages:] if len(non_system_messages) > keep_chat_messages else non_system_messages
            # 需要精简的旧消息
            old_messages = non_system_messages[:-keep_chat_messages] if len(non_system_messages) > keep_chat_messages else []

            # 如果有需要精简的消息
            if old_messages:
                # 将旧消息发送给LLM进行精简
                summary = await self._summarize_messages(old_messages)

                # 重构历史记录：系统提示 + 精简摘要 + 最近消息
                new_history = system_prompts + [{"role": "system", "content": f"新增的对话摘要：\n{summary}"}] + recent_messages

                print(f"✅ 历史记录精简完成")

                print(f"message size: {len(new_history)}")
                for msg in new_history:
                    print(f"role: {msg['role']}, content: {msg['content']}")

                # 如果精简的系统提示超过最大数量，继续精简
                currrent_sysprompts_len = len(system_prompts)
                if currrent_sysprompts_len >= self.system_prompt_maxnum:
                    system_summary = await self._summarize_system_prompts(new_history[1:currrent_sysprompts_len + 1])
                    new_history = [new_history[0], {"role": "system", "content": f"新增的对话摘要：\n{system_summary}"}] + recent_messages

                    print(f"✅ 系统提示精简完成")
                    print(f"message size: {len(new_history)}")
                    for msg in new_history:
                        print(f"role: {msg['role']}, content: {msg['content']}")

                return new_history

        # 如果不需要精简，直接返回原历史记录
        return history

    async def _summarize_messages(self, messages: list) -> str:
        """使用LLM对历史消息进行摘要"""

        # 构建摘要提示
        messages.append({
            "role": "user",
            "content": (
                "请你总结上述对话内容，你应该使用简洁准确的语言尽可能概括出所有重要的用户提问与助手回答，对每轮对话都严格按照如下格式进行整理：\n\n"
                "用户提到：...；助手回答：...。\n"
            )
        })

        # 调用LLM生成摘要
        response = await self.llm.chat(messages, [])  # 不需要提供工具
        summary = response.choices[0].message.content

        return summary

    async def _summarize_system_prompts(self, messages: list) -> str:
        """使用LLM对系统提示进行摘要"""
        # 构建摘要提示
        messages.append({
            "role": "user",
            "content": (
                "请你总结上述所有系统摘要中用户对话内容，使用简洁准确的语言提取出最重要的部分，合并重复的部分，丢弃不重要的部分，并严格按照如下格式进行整理：\n\n"
                "用户提到：...；助手回答：...。\n"
            )
        })

        # 调用LLM生成摘要
        response = await self.llm.chat(messages, [])
        return response.choices[0].message.content

    def _get_keep_chat_messages(self, messages: list, rounds: int) -> int:
        """
        获取需要保留的最近消息数量

        参数:
            messages: 用户的对话消息列表
            rounds: 保留的最近消息轮数

        返回:
            需要保留的消息数量
        """
        save_len = 0
        messages_len = len(messages)
        while rounds > 0 and save_len < messages_len:
            if messages[messages_len - save_len - 1].get("role") == "user":
                rounds -= 1
            save_len += 1
        return save_len
