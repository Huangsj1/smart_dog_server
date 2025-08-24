from chat_handler.chat_tts_handler import ChatTTSHandler
from my_asr.sensevoice_engine import SenseVoiceEngine
from my_llm.openai_engine import OpenAIEngine
from my_mcp.mcp_client import MCPClientManager
from my_tts.cosy_voice_engine import CosyVoiceEngine
from my_tts.gpt_sovits_engine import GPTSoVTISEngine
import asyncio
import yaml

async def main():
    config_file = None
    # 获取各种配置文件路径
    with open("config.yaml", 'r', encoding='utf-8') as f:
         config_file = yaml.safe_load(f)

    system_role = config_file.get("system_role", "ai_assistant")
    llm_provider = config_file.get("llm_provider", "siliconflow")
    llm_config = config_file["llm"][llm_provider]

    tts_provider = config_file.get("tts_provider", "cosy_voice")
    tts_remote = config_file.get("tts_remote", True)
    tts_config = config_file["tts"][tts_provider]

    asr_remote = config_file.get("asr_remote", True)
    asr_location = "remote" if asr_remote else "local"
    asr_provider = config_file.get("asr_provider", "sensevoice_small")
    asr_config = config_file["asr"][asr_location][asr_provider]

    # 初始化MCP客户端
    mcp_client = MCPClientManager()
    # 初始化LLM引擎
    llm_engine = OpenAIEngine(llm_config)
    # 初始化TTS引擎
    if system_role == "ai_assistant":
        tts_engine = CosyVoiceEngine(tts_config)
    else:
        tts_engine = GPTSoVTISEngine(tts_config, remote=tts_remote, role=system_role)
    # 初始化ASR引擎
    asr_engine = SenseVoiceEngine(asr_config, asr_remote)

    async with mcp_client.client:
        # 创建聊天处理器实例
        chat_tts_handler = ChatTTSHandler(llm_engine, mcp_client, tts_engine, asr_engine,
                                          whitelist_path=config_file["config_paths"]["whitelist_path"],
                                          max_context_tokens=llm_config["max_context_tokens"], system_role=system_role)
        await chat_tts_handler.start(system_role_path=config_file["config_paths"]["system_role_path"])

        # 进入对话循环
        await chat_tts_handler.interactive_loop_with_tts_asr()



if __name__ == "__main__":
    asyncio.run(main())