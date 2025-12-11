import sounddevice as sd
import numpy as np
import queue
import time
from config_fixed import SAMPLE_RATE, CHANNELS, BLOCK_SIZE, BLOCK_DURATION_MS

class AudioCapture:
    def __init__(self, sample_rate=SAMPLE_RATE, channels=CHANNELS, block_size=BLOCK_SIZE):
        """
        初始化音频采集器
        :param sample_rate: 采样率
        :param channels: 声道数
        :param block_size: 每个块的样本数
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.audio_queue = queue.Queue(maxsize=100)
        self.stream = None
        self.is_recording = False

    def _audio_callback(self, indata, frames, time_info, status):
        """SoundDevice 回调函数，将音频数据放入队列"""
        if status:
            print(f"音频流状态: {status}")
        # indata 形状为 (frames, channels)，转换为单声道并展平
        audio_data = indata[:, 0] if self.channels > 1 else indata.flatten()
        if audio_data.shape[0] == self.block_size:
            try:
                self.audio_queue.put_nowait(audio_data.astype(np.float32))
            except queue.Full:
                # 队列已满，丢弃最旧的数据
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio_data.astype(np.float32))
                except queue.Full:
                    pass  # 忽略

    def start(self):
        """开始音频采集"""
        if self.is_recording:
            return
        print(f"开始音频采集，采样率: {self.sample_rate} Hz, 块大小: {self.block_size} 样本 ({BLOCK_DURATION_MS} ms)")
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.block_size,
            callback=self._audio_callback,
            dtype=np.float32
        )
        self.stream.start()
        self.is_recording = True
        print("音频采集已启动")

    def stop(self):
        """停止音频采集"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_recording = False
        print("音频采集已停止")

    def read_block(self, timeout=1.0):
        """
        从队列中读取一个音频块
        :param timeout: 超时时间（秒）
        :return: numpy array 形状 (block_size,)，如果超时返回 None
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_blocks(self, num_blocks, timeout=1.0):
        """
        读取多个音频块并拼接
        :param num_blocks: 块数
        :param timeout: 每个块的超时时间
        :return: 拼接后的音频数据
        """
        blocks = []
        for _ in range(num_blocks):
            block = self.read_block(timeout)
            if block is None:
                break
            blocks.append(block)
        if blocks:
            return np.concatenate(blocks)
        else:
            return np.array([], dtype=np.float32)

if __name__ == "__main__":
    # 测试音频采集
    import sys
    capture = AudioCapture()
    try:
        capture.start()
        print("采集 5 个音频块...")
        for i in range(5):
            block = capture.read_block(timeout=2.0)
            if block is not None:
                print(f"块 {i+1}: 形状 {block.shape}, 均值 {np.mean(block):.4f}")
            else:
                print(f"块 {i+1}: 超时")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("中断")
    finally:
        capture.stop()
