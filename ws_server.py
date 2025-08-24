import wave

from fastapi import FastAPI, WebSocket
from chat_handler.chat_tts_handler import ChatTTSHandler
from my_asr.sensevoice_engine import SenseVoiceEngine
from my_llm.openai_engine import OpenAIEngine
from my_mcp.mcp_client import MCPClientManager
from my_tts.cosy_voice_engine import CosyVoiceEngine
from my_tts.gpt_sovits_engine import GPTSoVTISEngine
from my_vad.webrtc_vad import WebRTCVAD
from starlette.websockets import WebSocketDisconnect
from scipy.io.wavfile import write
import tempfile
import yaml
import os

app = FastAPI()

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

# 创建聊天处理器实例
chat_tts_handler = ChatTTSHandler(llm_engine, mcp_client, tts_engine, asr_engine,
                                  whitelist_path=config_file["config_paths"]["whitelist_path"],
                                  max_context_tokens=llm_config["max_context_tokens"], system_role=system_role)

# VAD（语音活动检测）引擎
vad = WebRTCVAD()


# 播放音频测试
import numpy as np
import noisereduce as nr # 引入降噪库

# --- 文件名定义 ---
SAMPLE_WIDTH = 2  # 每个采样点 2字节（int16）
PCM_FILE_16K = "audio_16khz.pcm"
WAV_FILE_16K = "audio_16khz.wav"
PCM_FILE_8K = "audio_8khz.pcm"
WAV_FILE_8K = "audio_8khz.wav"
# 为降噪后的文件定义新名称
PCM_FILE_16K_DENOISED = "audio_16khz_denoised.pcm"
WAV_FILE_16K_DENOISED = "audio_16khz_denoised.wav"
PCM_FILE_8K_DENOISED = "audio_8khz_denoised.pcm"
WAV_FILE_8K_DENOISED = "audio_8khz_denoised.wav"


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("客户端已连接")

    # 使用 bytearray 作为内存中的缓冲区，比反复读写文件更高效
    raw_buffer = bytearray()

    try:
        async for message in websocket.iter_bytes():
            # 将接收到的原始数据直接追加到缓冲区
            # 数据格式为 [S1, S1, S2, S2, ...]
            raw_buffer.extend(message)

    except WebSocketDisconnect:
        print("客户端正常断开连接")
    except Exception as e:
        print(f"连接异常：{e}")
    finally:
        print("数据接收完毕，开始处理并保存文件...")

        if not raw_buffer:
            print("未接收到任何数据，程序退出。")
            return

        try:
            # --- 1. 处理并保存 16kHz 原始文件 ---
            # 直接将缓冲区内容存为 16kHz PCM 文件
            with open(PCM_FILE_16K, "wb") as f:
                f.write(raw_buffer)
            print(f"已保存 16kHz PCM 文件: {PCM_FILE_16K}")

            # 将 16kHz PCM 数据存为 WAV 文件
            with wave.open(WAV_FILE_16K, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(16000)
                wf.writeframes(raw_buffer)
            print(f"已保存 16kHz WAV 文件: {WAV_FILE_16K}")

            # --- 2. 处理并保存 8kHz 清理后文件 ---
            # 使用 numpy 解析原始数据流
            raw_samples = np.frombuffer(raw_buffer, dtype=np.int16)

            # 提取不重复的采样点 (得到原始的8kHz数据)
            clean_samples = raw_samples[::2]

            # 将干净的数据转换回字节
            clean_bytes = clean_samples.tobytes()

            # 将清理后的字节存为 8kHz PCM 文件
            with open(PCM_FILE_8K, "wb") as f:
                f.write(clean_bytes)
            print(f"已保存 8kHz PCM 文件: {PCM_FILE_8K}")

            # 将 8kHz PCM 数据存为 WAV 文件
            with wave.open(WAV_FILE_8K, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(8000)
                wf.writeframes(clean_bytes)
            print(f"已保存 8kHz WAV 文件: {WAV_FILE_8K}")

            # --- 3. 对 16kHz 音频进行简单降噪并保存 ---
            print("正在对 16kHz 音频进行简单降噪处理...")
            # 使用 noisereduce 库对 16kHz 音频进行降噪
            denoised_16k_samples = nr.reduce_noise(y=raw_samples, sr=16000, prop_decrease=0.8)
            denoised_16k_bytes = denoised_16k_samples.tobytes()
            # 保存降噪后的 16kHz PCM 文件
            with open("audio_16khz_denoised.pcm", "wb") as f:
                f.write(denoised_16k_bytes)
            print(f"已保存降噪后的 16kHz PCM 文件: audio_16khz_denoised.pcm")
            # 保存降噪后的 16kHz WAV 文件
            with wave.open(WAV_FILE_16K_DENOISED, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(16000)
                wf.writeframes(denoised_16k_bytes)
            print(f"已保存降噪后的 16kHz WAV 文件: {WAV_FILE_16K_DENOISED}")

            # --- 4. 对 8kHz 音频进行AI降噪并保存 ---
            print("正在对 8kHz 音频进行降噪处理...")
            # 调用降噪函数。参数 prop_decrease 控制降噪强度（0到1之间）
            # 注意：clean_samples 是一个 numpy 数组，可以直接处理
            denoised_samples = nr.reduce_noise(y=clean_samples, sr=8000, prop_decrease=0.8)

            # 将降噪后的 numpy 数组转换回字节
            denoised_bytes = denoised_samples.tobytes()

            # 保存降噪后的 PCM 文件
            with open(PCM_FILE_8K_DENOISED, "wb") as f:
                f.write(denoised_bytes)
            print(f"已保存降噪后的 8kHz PCM 文件: {PCM_FILE_8K_DENOISED}")

            # 保存降噪后的 WAV 文件
            with wave.open(WAV_FILE_8K_DENOISED, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(8000)
                wf.writeframes(denoised_bytes)
            print(f"已保存降噪后的 8kHz WAV 文件: {WAV_FILE_8K_DENOISED}")


        except Exception as e:
            print(f"保存文件时出错：{e}")



# --- 端点二：将处理好的音频发送给客户端 ---
@app.websocket("/ws_send_audio")
async def websocket_send_audio(websocket: WebSocket):
    await websocket.accept()
    print(f"发送端：客户端已连接，准备发送音频文件: {PCM_FILE_8K}")

    chunk_size = 1024  # 每次发送1024字节

    try:
        if not os.path.exists(PCM_FILE_8K):
            not_found_msg = f"错误: 音频文件 '{PCM_FILE_8K}' 不存在。请先通过 /ws 端点接收音频。"
            print(f"发送端：{not_found_msg}")
            await websocket.send_text(not_found_msg)
            return

        with open(PCM_FILE_8K, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break  # 文件读取完毕
                await websocket.send_bytes(chunk)

        print(f"发送端：文件 {PCM_FILE_8K} 发送完毕。")

    except WebSocketDisconnect:
        print("发送端：客户端在传输过程中断开连接。")
    except Exception as e:
        print(f"发送端：发送音频时发生异常: {e}")
    finally:
        await websocket.close()
        print("发送端：连接已关闭。")


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     # 接受WebSocket连接（握手）
#     await websocket.accept()
#
#     print("--------------WebSocket连接已建立--------------")
#
#     async with mcp_client.client:
#         # 启动聊天处理器（目前每次启动都是新的开始，没有保存和加载用户上下文）
#         await chat_tts_handler.start(system_role_path=config_file["config_paths"]["system_role_path"],
#                                      websocket=websocket)
#         print("对话窗口长度：", len(chat_tts_handler.history))
#
#         try:
#             while True:
#                 try:
#                     record_audio = await vad.detect_voice_from_ws(websocket)
#                     if record_audio:
#                         print("检测到语音活动，开始处理...")
#                         # 1.将音频保存成临时的wav文件
#                         temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#                         write(temp_audio_file.name, vad.sample_rate, record_audio)
#
#                         # 2.传递给chat_tts_handler进行处理
#                         await chat_tts_handler.interactive_with_audio_input(temp_audio_file.name)
#
#                         # 3.处理完毕后删除临时文件
#                         if temp_audio_file.name:
#                             print(f"删除临时音频文件: {temp_audio_file.name}")
#                             os.remove(temp_audio_file.name)
#                     else:
#                         print("没有检测到语音活动，等待下一次输入...")
#                 except WebSocketDisconnect as e:
#                     print(f"客户端断开连接: {e.code}, 原因: {e.reason}")
#                     break
#         finally:
#             # 确保在退出时停止聊天处理器
#             await chat_tts_handler.stop()
#
#     print("----------------WebSocket连接已关闭----------------")
