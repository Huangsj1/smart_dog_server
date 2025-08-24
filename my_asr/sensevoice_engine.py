import requests
from gradio_client import Client, handle_file

class SenseVoiceEngine:
    def __init__(self, asr_config: dict, remote=True):
        # 是否使用远程调用
        self.remote = remote
        self.api_key = asr_config.get("api_key", "")
        self.base_url = asr_config.get("base_url", "")
        self.model = asr_config.get("model", "")

    def audio_to_text(self, file_path: str, file_lang: str="auto") -> str:
        """
        将音频文件通过 API 调用得到文本结果
        """
        if self.remote:
            return self._remote_audio_to_text(file_path)
        else:
            # 下面这个api调用会很慢，暂时找不到解决方法
            # return self._local_audio_to_text(file_path, file_lang)
            # 而开启webui来调用api就很快
            return self._local_audio_to_text_webui(file_path, file_lang)

    def _remote_audio_to_text(self, file_path: str) -> str:
        """
        通过远程API将音频文件转换为文本
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        files = {
            "file": open(file_path, "rb"),
        }

        data = {
            "model": self.model
        }

        print("📤 正在发送语音转文字请求...")
        response = requests.post(self.base_url, headers=headers, files=files, data=data)

        if response.status_code == 200:
            return response.json().get("text", "无结果")
        else:
            raise Exception(f"❌ 出错：{response.status_code} {response.text}")

    def _local_audio_to_text(self, file_path: str, file_lang: str) -> str:
        """
        本地处理音频文件转换为文本
        """
        # Headers：期望收到json数据格式
        headers = {
            "accept": "application/json",
        }

        # 上传文件（可以上传多个文件，需要按照下面格式）
        files = [
            ("files", (file_path, open(file_path, "rb"), "audio/wav")),
            # ("files", ("zero_shot_0.wav", open("zero_shot_0.wav", "rb"), "audio/wav"))
        ]
        data = {
            # 音频文件名，用逗号分开
            "keys": "my_audio_file",
            # 音频语言类型
            "lang": file_lang
        }

        # 请求数据
        response = requests.post(self.base_url, headers=headers, files=files, data=data)

        # P返回数据
        if response.status_code == 200:
            # 返回解析到的数据
            return response.json().get('result')[0].get('text')
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print("响应内容:", response.text)
            return response.text

    def _local_audio_to_text_webui(self, file_path: str, file_lang: str) -> str:
        """
        使用WebUI处理音频文件转换为文本
        """
        client = Client(self.base_url)
        result = client.predict(
            input_wav=handle_file(file_path),
            language=file_lang,
            api_name="/model_inference"
        )
        return result


if __name__ == "__main__":
    # 下面是调用webui的api
    client = Client("http://127.0.0.1:9999/")
    print("start---")
    result = client.predict(
        input_wav=handle_file('/Users/geer/音频/海绵宝宝/海绵宝宝/不，派大星，你为什么！抓住它，抓住他派大星，不！抓住它！.wav'),
        language="auto",
        api_name="/model_inference"
    )
    print("end---")
    print(result)