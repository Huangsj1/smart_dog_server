from openai import OpenAI

class CosyVoiceEngine:
    """文本转语音处理器"""

    def __init__(self, tts_config: dict):
        self.api_key = tts_config["api_key"]
        self.base_url = tts_config["base_url"]
        # TTS客户端
        self.tts_client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.model = tts_config["model"]
        self.voice = tts_config["voice"]
        self.response_format = tts_config["response_format"]

    async def text_to_speech(self, text: str) -> bytes:
        """
        将文本转换为语音数据，返回可直接播放的音频字节数据
        Args:
            text: 要转换为语音的文本
        Returns:
            bytes: 音频字节数据，格式为在配置中指定的response_format（如MP3、WAV等）
        """
        params = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "response_format": self.response_format
        }

        try:
            with self.tts_client.audio.speech.with_streaming_response.create(
                    **params
            ) as response:
                # 读取所有音频数据
                audio_data = response.read()

                print(f"[CosyVoiceEngine] 音频数据长度: {len(audio_data)} bytes")

                return audio_data
        except Exception as e:
            print(f"TTS API调用失败: {e}")
            raise
