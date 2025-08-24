import webrtcvad
import os


class WebRTCVAD:
    def __init__(self, mode=3, sample_rate=16000, frame_duration_ms=30, max_silence_ms=2000):
        """
        初始化VAD
        :param mode: VAD模式，0-3，值越高越敏感
        :param sample_rate: 音频采样率（客户端那边默认16khz）
        :param frame_duration_ms: 每帧的时长（毫秒）（只能选10、20、30）
        """
        self.vad = webrtcvad.Vad(mode)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        # 每帧采样数（16位音频）（16000*30/1000=480，客户端那边每次发送ws数据的chunksize为512）
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        # 每帧的字节数（16位音频，每个样本2字节）
        self.frame_bytes = self.frame_size * 2
        # 最大静音时长（毫秒），超过这个时间没有语音则认为是静音
        self.max_silence_ms = max_silence_ms

    def is_speech(self, frame, sample_rate=None):
        """
        检测帧是否包含语音
        :param frame: 音频帧
        :param sample_rate: 采样率
        :return: True表示语音，False表示静音
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        return self.vad.is_speech(frame, sample_rate)

    async def detect_voice_from_ws(self, websocket, sample_rate=None):
        """
        从WebSocket接收音频数据并检测语音活动
        :param websocket: WebSocket连接对象
        :param sample_rate: 采样率
        :return: 检测到的语音数据
        """
        audio_buffer = b""
        record_audio = b""
        is_speaking = False
        silence_time_ms = 0
        while True:
            try:
                # 接收音频数据（每次传输音频的大小chunksize为512，字节数为1024）
                audio_data = await websocket.receive_bytes()
                if not audio_data:
                    break  # 连接关闭或没有数据

                audio_buffer += audio_data
                # 将音频数据分成帧进行处理
                for i in range(0, len(audio_buffer), self.frame_bytes):
                    frame = audio_buffer[i:i + self.frame_bytes]
                    # 如果帧长度小于预期的帧大小，跳过
                    if len(frame) < self.frame_bytes:
                        audio_buffer = frame
                        break
                    # 检测语音活动
                    if self.is_speech(frame, sample_rate):
                        if not is_speaking:
                            print("检测到语音活动，开始记录音频...")
                            is_speaking = True
                            record_audio = b""  # 清空之前的记录
                        # 重置静音计时器
                        silence_time_ms = 0
                    else:
                        if is_speaking:
                            silence_time_ms += self.frame_duration_ms
                            if silence_time_ms >= self.max_silence_ms:
                                print("检测到静音，结束记录音频。")
                                is_speaking = False
                                break
                    # 只要正在说话，就将音频帧添加到记录中（及时当前帧是沉默的）
                    if is_speaking:
                        record_audio += frame
                if not is_speaking and record_audio:
                    print("语音活动结束，返回记录的音频数据。")
                    break

            except Exception as e:
                print(f"Error receiving data: {e}")
                break

        return record_audio

    async def detect_voice_from_file(self, file_path, sample_rate=None):
        """
        从本地音频文件进行语音活动检测
        每次读取 frame_bytes 大小的数据，调用 VAD 检测
        :param file_path: 音频文件路径 (必须是16kHz, 16bit PCM 格式)
        :param sample_rate: 采样率
        :return: 检测到的语音数据（bytes）
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"音频文件不存在: {file_path}")

        record_audio = b""
        is_speaking = False
        silence_time_ms = 0

        with open(file_path, "rb") as f:
            while True:
                frame = f.read(self.frame_bytes)
                if len(frame) < self.frame_bytes:
                    break  # 文件读完

                if self.is_speech(frame, sample_rate):
                    if not is_speaking:
                        print("检测到语音活动，开始记录音频...")
                        is_speaking = True
                        record_audio = b""
                    silence_time_ms = 0
                else:
                    if is_speaking:
                        silence_time_ms += self.frame_duration_ms
                        if silence_time_ms >= self.max_silence_ms:
                            print("检测到静音，结束记录音频。")
                            is_speaking = False
                            break

                if is_speaking:
                    record_audio += frame

        if record_audio:
            print("语音活动结束，返回记录的音频数据。")
        else:
            print("未检测到语音活动。")

        return record_audio


if __name__ == '__main__':
    import asyncio
    import wave

    vad = WebRTCVAD()

    async def test_vad():
        file_path = "../audio_16khz.wav"
        save_path = "../audio_16khz_detected.wav"
        sample_rate = 16000
        audio_data = await vad.detect_voice_from_file(file_path)
        if audio_data:
            print(f"检测到语音数据，长度: {len(audio_data)} bytes")
            # 将字节保存成wav文件，方便播放测试
            with wave.open(save_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)
            print(f"已保存检测到的语音数据到: {save_path}")
        else:
            print("未检测到语音数据")

    asyncio.run(test_vad())