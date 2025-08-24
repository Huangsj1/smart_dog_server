import sounddevice as sd
import soundfile as sf
import io


class AudioPlayer:
    """音频播放器"""

    def __init__(self, sample_rate=24000):
        """
        初始化音频播放器

        Args:
            sample_rate: 采样率，默认24000Hz (OpenAI TTS的默认采样率)
        """
        self.sample_rate = sample_rate

    def play_audio(self, audio_data: bytes):
        """
        播放音频数据

        Args:
            audio_data: WAV格式的音频字节数据
        """
        try:
            # 将字节数据转换为numpy数组
            with io.BytesIO(audio_data) as buf:
                data, fs = sf.read(buf)
                # 播放音频 (阻塞式)
                sd.play(data, fs)
                sd.wait()  # 等待音频播放完成

            print("[AudioPlayer] 音频播放完成")
        except Exception as e:
            print(f"[AudioPlayer] 播放出错: {e}")