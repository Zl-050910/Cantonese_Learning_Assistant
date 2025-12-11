import whisper
import numpy as np
import torch
import time
import zhconv
from config_fixed import MODEL_PATH, SAMPLE_RATE, LANGUAGE, MODEL_COMPUTE_TYPE, BEAM_SIZE

class WhisperRecognizer:
    def __init__(self, model_path=MODEL_PATH, device=None):
        """
        初始化 Whisper 识别器
        :param model_path: 模型路径（.pt 文件）
        :param device: "cuda" 或 "cpu"，None 表示自动选择
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"加载 Whisper 模型: {model_path}, 设备: {device}, 计算类型: {MODEL_COMPUTE_TYPE}")

        # 加载whisper官方模型方法
        # self.model = whisper.load_model(model_path, device=device)

        # 加载自定义完全模型方法
        self.model = torch.load(
                        model_path,
                        map_location=device,
                        weights_only=False
                    )

        # 加载自定义参数模型方法
        # base_model = whisper.load_model(
        #                     "base",
        #                     device=device,
        #                     download_root=".\whisper_models"
        #                 ) # 加载对应尺寸的基础模型
        # state_dict = torch.load(
        #                     model_path,
        #                     map_location=device,
        #                     weights_only=True  # state_dict可以安全加载
        #                 ) # 加载state_dict
        # base_model.load_state_dict(state_dict) # 加载微调后的权重
        # self.model = base_model

        self.sample_rate = SAMPLE_RATE
        print("模型加载完成")

    def _convert_to_simplified(self, text):
        """
        将文本转换为简体中文
        :param text: 输入文本
        :return: 简体中文文本
        """
        if not text:
            return text
        
        # 使用zhconv将繁体转换为简体
        try:
            simplified_text = zhconv.convert(text, 'zh-cn')
            return simplified_text
        except Exception as e:
            print(f"繁体转简体转换错误: {e}")
            return text

    def transcribe(self, audio_data, language=None, beam_size=None):
        """
        转录音频数据
        :param audio_data: numpy array，int16 格式，单声道
        :param language: 语言代码，如 "zh", "en"，None 为自动检测（默认使用 config.LANGUAGE）
        :param beam_size: beam search 大小，None 为使用 config.BEAM_SIZE
        :return: 识别文本和分段信息
        """
        if len(audio_data) == 0:
            return "", []
        
        # 设置默认语言
        if language is None:
            language = LANGUAGE
        
        # 设置默认 beam_size
        if beam_size is None:
            beam_size = BEAM_SIZE
        
        # 确保音频为 float32，并归一化到 [-1, 1]
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / 32768.0
        
        start_time = time.time()
        audio_duration = len(audio_data) / self.sample_rate
        
        # 根据 MODEL_COMPUTE_TYPE 设置 fp16
        if MODEL_COMPUTE_TYPE.lower() == "float16" and torch.cuda.is_available() and self.model.device == "cuda":
            fp16 = True
        else:
            fp16 = False
        
        # 转录
        result = self.model.transcribe(
            audio_data,
            language=language,
            beam_size=beam_size,
            fp16=fp16
        )
        
        # 转换主文本为简体中文
        original_text = result["text"].strip()
        simplified_text = self._convert_to_simplified(original_text)
        
        # 转换segment文本为简体中文
        segments_list = []
        for segment in result["segments"]:
            seg_text = segment["text"].strip()
            simplified_seg_text = self._convert_to_simplified(seg_text)
            segments_list.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": simplified_seg_text
            })
        
        elapsed = time.time() - start_time
        real_time_ratio = elapsed / audio_duration if audio_duration > 0 else 0
        print(f"转录完成: 音频时长 {audio_duration:.2f}s, 处理时间 {elapsed:.2f}s, 实时比 {real_time_ratio:.2f}")
        
        return simplified_text, segments_list

    def transcribe_stream(self, audio_generator, language=None, beam_size=None):
        """
        流式转录（实验性）
        :param audio_generator: 生成音频块的生成器
        :param language: 语言代码
        :param beam_size: beam search 大小
        :return: 生成 (text, is_final) 的生成器
        """
        # 设置默认 beam_size
        if beam_size is None:
            beam_size = BEAM_SIZE
            
        # 简单缓冲实现
        buffer = []
        min_duration = 1.0  # 最小缓冲时长（秒）
        
        for audio_chunk in audio_generator:
            buffer.append(audio_chunk)
            buffer_duration = sum(len(chunk) for chunk in buffer) / self.sample_rate
            
            if buffer_duration >= min_duration:
                # 有足够的数据进行转录
                audio_data = np.concatenate(buffer)
                text, _ = self.transcribe(audio_data, language, beam_size)
                yield text, False  # 中间结果
                buffer = []  # 清空缓冲
        
        # 处理剩余数据
        if buffer:
            audio_data = np.concatenate(buffer)
            text, _ = self.transcribe(audio_data, language, beam_size)
            yield text, True  # 最终结果

if __name__ == "__main__":
    # 测试 Whisper 识别
    recognizer = WhisperRecognizer()
    
    # 测试转录（生成测试音频）
    print("生成测试音频...")
    duration = 3.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    test_audio = (np.sin(2 * np.pi * 440 * t) * 0.5 * 32767).astype(np.int16)
    
    print("开始转录测试...")
    text, segments = recognizer.transcribe(test_audio, language=LANGUAGE)
    print(f"识别结果: {text}")
    for seg in segments:
        print(f"  [{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}")
    
    # 测试繁体转简体功能
    print("\n测试繁体转简体功能...")
    test_traditional = "這是一個繁體中文測試，香港和台灣使用繁體字"
    simplified = zhconv.convert(test_traditional, 'zh-cn')
    print(f"繁体: {test_traditional}")
    print(f"简体: {simplified}")
