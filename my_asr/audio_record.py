from pynput import keyboard
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os


class AudioRecord:
    SAMPLE_RATE = 16000
    CHANNELS = 1

    def __init__(self):
        self.recording = None
        self.file_path = None
        self.is_recording = False

    def record_audio(self) -> str:
        """
        按住键开始录音，松开键停止录音
        返回录音文件的路径
        """
        print("🎙️ 按住 'a' 键开始录音，松开停止录音...")

        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char == 'a' and not self.is_recording:
                    print("🎙️ 正在录音...")
                    self.is_recording = True
                    self.recording = sd.rec(int(60 * self.SAMPLE_RATE), samplerate=self.SAMPLE_RATE, channels=self.CHANNELS, dtype='int16')
            except AttributeError:
                pass

        def on_release(key):
            try:
                # 检查 key 是否有 char 属性
                if hasattr(key, 'char') and key.char == 'a' and self.is_recording:
                    self.is_recording = False
                    sd.stop()
                    self.file_path = self.save_audio(self.recording)
                    print("✅ 录音完成")
                    # 打印self.recording的形状和类型（numpy.ndarray类型，每个元素为int16类型，范围是 -32768 到 32767）
                    print(f"录音数据形状: {self.recording.shape}, 类型: {type(self.recording)}，每个元素类型: {self.recording[0][0].dtype}")
                    return False  # 停止监听
            except AttributeError:
                pass

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

        return self.file_path

    def save_audio(self, audio):
        """
        保存录音到临时文件
        """
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write(temp.name, self.SAMPLE_RATE, audio)
        return temp.name

    def cleanup(self):
        """
        删除临时音频文件
        """
        if self.file_path and os.path.exists(self.file_path):
            os.remove(self.file_path)


if __name__ == "__main__":
    recorder = AudioRecord()
    audio_file = recorder.record_audio()
    if audio_file:
        print(f"录音文件已保存: {audio_file}")
    else:
        print("没有录音文件被保存。")

    # 清理临时文件
    recorder.cleanup()