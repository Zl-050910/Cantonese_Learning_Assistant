# 修复版配置参数
import os

# 音频采集 - 匹配VAD帧大小
SAMPLE_RATE = 16000  # Whisper 标准采样率
CHANNELS = 1
BLOCK_SIZE = 480  # 30 ms @ 16000 Hz (30 * 16)，匹配VAD帧大小
BLOCK_DURATION_MS = 30  # 毫秒

# VAD 参数 - 优化准确性
VAD_AGGRESSIVENESS = 2  # 0-3, 2 为中等敏感度，平衡误识别和漏识别
VAD_FRAME_DURATION_MS = 30  # webrtcvad 支持的帧时长（10, 20, 30），使用30ms提高准确性
VAD_SILENCE_DURATION_MS = 1000  # 静音持续时间（毫秒）后认为语音段结束，适中值

# 音量阈值 - 优化
VOLUME_THRESHOLD_DB = -50  # 音量阈值（dB），-50 dB允许正常语音通过
MIN_SPEECH_DURATION_MS = 300  # 最小语音时长（毫秒），过滤过短语音

# Whisper 模型
MODEL_PATH = os.path.join("whisper_models", "cantonese_model_epoch_20.pt")
MODEL_DEVICE = "cpu"  # 使用CPU确保兼容性
MODEL_COMPUTE_TYPE = "float32"  # 使用float32保证精度
BEAM_SIZE = 1  # 减小beam size以降低计算量，提高速度

# 语言设置
LANGUAGE = "zh"  # 中文识别

# 实时处理
MAX_QUEUE_SIZE = 100  # 队列大小
