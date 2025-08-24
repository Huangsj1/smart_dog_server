import requests
from my_tts.audio_player import AudioPlayer

class GPTSoVTISEngine:
    """
    GPT-SoVITS引擎
    """

    def __init__(self, tts_config: dict, remote=True, role="SpongeBob"):
        """
        初始化GPT-SoVITS引擎
        """
        location = "remote" if remote else "local"
        self.gpt_sovits_config = tts_config.get(location, {})
        # url
        self.base_url = self.gpt_sovits_config.get("base_url", {})

        self.role_config = self.gpt_sovits_config.get(role, {})
        # 设置模型
        gpt_model_path = self.role_config.get("gpt_model_path", "")
        sovits_model_path = self.role_config.get("sovits_model_path", "")
        self.switch_role_audio(gpt_model_path, sovits_model_path)
        # 角色、参考音频相关
        self.version = self.role_config.get("version", {})
        self.prompt_lang = self.role_config.get("prompt_lang", {})
        self.ref_audio_emotion_config = self.role_config.get("ref_audio_emotion", {})

    async def text_to_speech(self, text: str, text_lang="zh", emotion="normal") -> bytes:
        """
        将文本转换为语音数据，返回可直接播放的音频字节数据
        Args:
            text: 要转换为语音的文本
            text_lang: 文本语言，默认为中文（zh）
            emotion: 参考音频情感，默认为"normal"，可选值包括"normal", "happy", "angry",不同角色拥有的情绪不同
        Returns:
            bytes: 音频字节数据
        """
        # 如果有当前情绪的参考音频就使用，否则使用“normal”默认的
        ref_audio_config = self.ref_audio_emotion_config.get(emotion, self.ref_audio_emotion_config.get("normal", {}))

        data = {
            "text": text,
            "text_lang": text_lang,
            "prompt_lang": self.prompt_lang,
            "ref_audio_path": ref_audio_config.get("ref_audio_path", ""),
            "prompt_text": ref_audio_config.get("prompt_text", ""),
            "sample_steps": self.role_config.get("sample_steps", 16),
        }

        # 使用v4版本的GPT-SoVITS API（v2暂时有问题，还没有改）
        response = requests.post(f"{self.base_url}/tts", json=data)

        if response.status_code != 200:
            raise Exception(f"请求GPT-SoVITS出错: {response.text}")

        return response.content

    def switch_role_audio(self, gpt_model_path: str, sovits_model_path: str):
        """
        切换角色音频模型
        Args:
            gpt_model_path: GPT模型路径
            sovits_model_path: SoVITS模型路径
        """
        if not gpt_model_path or not sovits_model_path:
            raise ValueError("GPT模型路径和SoVITS模型路径不能为空")

        # 发送GET请求切换模型
        gpt_response = requests.get(f"{self.base_url}/set_gpt_weights", params={
            "weights_path": gpt_model_path
        })
        if gpt_response.status_code != 200:
            raise Exception(f"切换GPT模型失败: {gpt_response.text}")

        sovits_response = requests.get(f"{self.base_url}/set_sovits_weights", params={
            "weights_path": sovits_model_path
        })
        if sovits_response.status_code != 200:
            raise Exception(f"切换SoVITS模型失败: {sovits_response.text}")

        print(f"成功切换角色音频模型: GPT={gpt_model_path}, SoVITS={sovits_model_path}")


if __name__ == "__main__":
    data = {
        "text": "你好，我的朋友。我们去抓水母吧。",
        "text_lang": "zh",
        # 下面的参考音频可以不指定（不指定用启动模型时默认的）
        "ref_audio_path": "output/denoise_opt/精神饱满地迎接新的一天，小蜗.wav_0000000000_0000116480.wav",
        "prompt_text": "精神饱满地迎接新的一天，小蜗。",
        "prompt_lang": "zh"
    }

    response = requests.post("http://127.0.0.1:9880/tts", json=data)

    print(response)

    # 失败时返回错误信息
    if response.status_code == 400:
        raise Exception(f"请求GPT-VoTIS出错:{response.message}")

    # # 成功时返回wav音频流
    # with open("success.wav", 'wb') as f:
    #     f.write(response.content)

    ap = AudioPlayer()
    ap.play_audio(response.content)
