# import torch
# import soundfile as sf
# import numpy as np
# import librosa
# import pyaudio
# from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

# # ====================== 配置 ======================
# MODEL_PATH = r"E:\School\learning\语音信号处理\大作业\best_model_chinese-hubert-base"
# id2label = {
#     0: "happy",      # 开心
#     1: "neutral",    # 中性
#     2: "surprised",  # 惊讶
#     3: "fearful",    # 恐惧
#     4: "disgusted",  # 厌恶
#     5: "sad",        # 悲伤
#     6: "other",      # 其他
#     7: "angry"       # 愤怒
# }
# SR = 16000
# MAX_LENGTH = SR * 5


# # ====================== 加载模型 ======================
# def load_model():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"✅ 模型加载到设备：{device}")
    
#     model = HubertForSequenceClassification.from_pretrained(
#         MODEL_PATH,
#         dtype=torch.float32 if device.type == "cpu" else torch.float16
#     ).to(device)
#     model.eval()
    
#     feature_extractor = Wav2Vec2FeatureExtractor(
#         sampling_rate=SR,
#         do_normalize=True,
#         return_attention_mask=False,
#         padding_value=0.0,
#         truncation=True,
#         max_length=MAX_LENGTH
#     )
#     return model, feature_extractor, device


# # ====================== 音频预处理 ======================
# def preprocess_audio(audio_data, sample_rate):
#     if isinstance(audio_data, str):
#         waveform, sample_rate = sf.read(audio_data)
#     else:
#         waveform = audio_data
    
#     if len(waveform.shape) > 1:
#         waveform = waveform.mean(axis=1)
#     if sample_rate != SR:
#         waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SR)
#     if len(waveform) < MAX_LENGTH:
#         waveform = np.pad(waveform, (0, MAX_LENGTH - len(waveform)), mode="constant")
#     else:
#         waveform = waveform[:MAX_LENGTH]
    
#     inputs = feature_extractor(
#         waveform,
#         sampling_rate=SR,
#         return_tensors="pt",
#         padding=False,
#         truncation=False
#     )
#     return inputs["input_values"]


# # ====================== 情感预测（核心修改：输出置信度） ======================
# def predict_emotion(model, feature_extractor, device, audio_input):
#     """
#     输入音频，输出：
#     - 预测的情感标签
#     - 该标签的置信度（0~1）
#     - 所有情感的概率分布
#     """
#     # 预处理音频
#     if isinstance(audio_input, str):
#         input_tensor = preprocess_audio(audio_input, sample_rate=None)
#     else:
#         input_tensor = preprocess_audio(audio_input, sample_rate=SR)
    
#     # 推理并计算概率
#     input_tensor = input_tensor.to(device)
#     with torch.no_grad():
#         outputs = model(input_values=input_tensor)
#         # 将logits转为概率（softmax）
#         probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
#         # 取概率最大的标签ID和置信度
#         pred_id = np.argmax(probs)
#         pred_label = id2label[pred_id]
#         pred_confidence = round(probs[pred_id], 4)  # 保留4位小数
#         # 生成所有标签的概率分布
#         all_probs = {id2label[i]: round(probs[i], 4) for i in range(len(probs))}
    
#     return pred_label, pred_confidence, all_probs


# # ====================== 实时录制 ======================
# def record_audio_realtime(duration=5):
#     p = pyaudio.PyAudio()
#     stream = p.open(
#         format=pyaudio.paFloat32,
#         channels=1,
#         rate=SR,
#         input=True,
#         frames_per_buffer=1024
#     )
    
#     print(f"\n🎤 开始录制{duration}秒语音...")
#     frames = []
#     for _ in range(0, int(SR / 1024 * duration)):
#         data = stream.read(1024)
#         frames.append(np.frombuffer(data, dtype=np.float32))
    
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#     print("🎙️ 录制完成！")
#     return np.concatenate(frames)


# # ====================== 交互入口（修改输出逻辑） ======================
# if __name__ == "__main__":
#     try:
#         model, feature_extractor, device = load_model()
#     except Exception as e:
#         print(f"❌ 模型加载失败：{e}")
#         exit(1)
    
#     while True:
#         print("\n" + "="*60)
#         print("🎯 语音情感识别系统（支持8类情感）")
#         print("="*60)
#         print("请选择输入方式：")
#         print("1. 输入WAV文件路径（如E:\\test.wav）")
#         print("2. 实时麦克风录制（5秒）")
#         print("3. 退出程序")
#         choice = input("\n输入选项（1/2/3）：").strip()
        
#         if choice == "1":
#             wav_path = input("请输入WAV文件路径：").strip()
#             if not wav_path.endswith(".wav"):
#                 print("❌ 错误：请输入WAV格式的音频文件！")
#                 continue
#             try:
#                 # 获取预测结果+置信度+所有概率
#                 pred_label, pred_conf, all_probs = predict_emotion(model, feature_extractor, device, wav_path)
#                 print(f"\n✅ 情感识别结果：")
#                 print(f"   最终标签：{pred_label}")
#                 print(f"   置信度：{pred_conf}（{pred_conf*100:.2f}%）")
#                 # 可选：输出所有情感的概率分布（按概率降序排列）
#                 print("\n📊 所有情感概率分布（降序）：")
#                 sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
#                 for label, prob in sorted_probs:
#                     print(f"   {label}: {prob}（{prob*100:.2f}%）")
#             except Exception as e:
#                 print(f"❌ 识别失败：{e}（请检查文件路径/格式）")
        
#         elif choice == "2":
#             try:
#                 audio_array = record_audio_realtime()
#                 pred_label, pred_conf, all_probs = predict_emotion(model, feature_extractor, device, audio_array)
#                 print(f"\n✅ 情感识别结果：")
#                 print(f"   最终标签：{pred_label}")
#                 print(f"   置信度：{pred_conf}（{pred_conf*100:.2f}%）")
#                 print("\n📊 所有情感概率分布（降序）：")
#                 sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
#                 for label, prob in sorted_probs:
#                     print(f"   {label}: {prob}（{prob*100:.2f}%）")
#             except Exception as e:
#                 print(f"❌ 识别失败：{e}（请检查麦克风是否正常）")
        
#         elif choice == "3":
#             print("👋 退出程序，再见！")
#             break
        
#         else:
#             print("❌ 无效选项，请输入1/2/3！")

import torch
import soundfile as sf
import numpy as np
import librosa
import pyaudio
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

# ====================== 配置（可自定义录制时长） ======================
MODEL_PATH = r"E:\School\learning\语音信号处理\大作业\best_model_chinese-hubert-base"
id2label = {
    0: "happy",      # 开心
    1: "neutral",    # 中性
    2: "surprised",  # 惊讶
    3: "fearful",    # 恐惧
    4: "disgusted",  # 厌恶
    5: "sad",        # 悲伤
    6: "other",      # 其他
    7: "angry"       # 愤怒
}
SR = 16000  # 固定采样率（必须和训练时一致）
# 核心修改：自定义录制时长（默认10秒，可改）
RECORD_DURATION_DEFAULT = 10  # 默认录制10秒，可改为20/30等
MAX_LENGTH = SR * RECORD_DURATION_DEFAULT  # 适配录制时长的最大长度


# ====================== 加载模型 ======================
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 模型加载到设备：{device}")
    
    model = HubertForSequenceClassification.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32 if device.type == "cpu" else torch.float16
    ).to(device)
    model.eval()
    
    # 特征提取器适配自定义时长
    feature_extractor = Wav2Vec2FeatureExtractor(
        sampling_rate=SR,
        do_normalize=True,
        return_attention_mask=False,
        padding_value=0.0,
        truncation=True,  # 若音频超过自定义时长，截断；若不足，补零
        max_length=MAX_LENGTH
    )
    return model, feature_extractor, device


# ====================== 音频预处理（适配任意长度） ======================
def preprocess_audio(audio_data, sample_rate):
    """支持任意长度音频：不足补零，超过截断"""
    if isinstance(audio_data, str):
        waveform, sample_rate = sf.read(audio_data)
    else:
        waveform = audio_data
    
    # 转单声道
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)
    # 转16kHz采样率
    if sample_rate != SR:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SR)
    # 适配自定义时长（不足补零，超过截断）
    if len(waveform) < MAX_LENGTH:
        waveform = np.pad(waveform, (0, MAX_LENGTH - len(waveform)), mode="constant")
    else:
        waveform = waveform[:MAX_LENGTH]
    
    inputs = feature_extractor(
        waveform,
        sampling_rate=SR,
        return_tensors="pt",
        padding=False,
        truncation=False
    )
    return inputs["input_values"]


# ====================== 情感预测（保留置信度输出） ======================
def predict_emotion(model, feature_extractor, device, audio_input):
    if isinstance(audio_input, str):
        input_tensor = preprocess_audio(audio_input, sample_rate=None)
    else:
        input_tensor = preprocess_audio(audio_input, sample_rate=SR)
    
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        outputs = model(input_values=input_tensor)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_id = np.argmax(probs)
        pred_label = id2label[pred_id]
        pred_confidence = round(probs[pred_id], 4)
        all_probs = {id2label[i]: round(probs[i], 4) for i in range(len(probs))}
    
    return pred_label, pred_confidence, all_probs


# ====================== 实时录制（支持自定义时长） ======================
def record_audio_realtime(duration):
    """
    录制指定时长的音频
    :param duration: 录制时长（秒），由用户输入
    """
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SR,
        input=True,
        frames_per_buffer=1024
    )
    
    print(f"\n🎤 开始录制{duration}秒语音...（请说话）")
    frames = []
    # 按指定时长录制
    for _ in range(0, int(SR / 1024 * duration)):
        data = stream.read(1024)
        frames.append(np.frombuffer(data, dtype=np.float32))
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    print(f"🎙️ {duration}秒音频录制完成！")
    return np.concatenate(frames)


# ====================== 交互入口（支持自定义录制时长） ======================
if __name__ == "__main__":
    try:
        model, feature_extractor, device = load_model()
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        exit(1)
    
    while True:
        print("\n" + "="*60)
        print("🎯 语音情感识别系统（支持自定义录制时长）")
        print("="*60)
        print("请选择输入方式：")
        print("1. 输入WAV文件路径（支持任意长度）")
        print("2. 实时麦克风录制（自定义时长）")
        print("3. 退出程序")
        choice = input("\n输入选项（1/2/3）：").strip()
        
        if choice == "1":
            wav_path = input("请输入WAV文件路径：").strip()
            if not wav_path.endswith(".wav"):
                print("❌ 错误：请输入WAV格式的音频文件！")
                continue
            try:
                pred_label, pred_conf, all_probs = predict_emotion(model, feature_extractor, device, wav_path)
                print(f"\n✅ 情感识别结果：")
                print(f"   最终标签：{pred_label}")
                print(f"   置信度：{pred_conf}（{pred_conf*100:.2f}%）")
                print("\n📊 所有情感概率分布（降序）：")
                sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                for label, prob in sorted_probs:
                    print(f"   {label}: {prob}（{prob*100:.2f}%）")
            except Exception as e:
                print(f"❌ 识别失败：{e}（请检查文件路径/格式）")
        
        elif choice == "2":
            # 让用户输入录制时长（默认10秒）
            try:
                duration_input = input(f"请输入录制时长（秒，默认{RECORD_DURATION_DEFAULT}）：").strip()
                # 处理空输入（用默认时长）
                if not duration_input:
                    duration = RECORD_DURATION_DEFAULT
                else:
                    duration = int(duration_input)
                    # 校验时长合理性（至少1秒，最多60秒）
                    if duration < 1 or duration > 60:
                        print(f"❌ 时长无效，默认使用{RECORD_DURATION_DEFAULT}秒")
                        duration = RECORD_DURATION_DEFAULT
            except ValueError:
                print(f"❌ 输入非数字，默认使用{RECORD_DURATION_DEFAULT}秒")
                duration = RECORD_DURATION_DEFAULT
            
            # 录制并识别
            try:
                audio_array = record_audio_realtime(duration)
                pred_label, pred_conf, all_probs = predict_emotion(model, feature_extractor, device, audio_array)
                print(f"\n✅ 情感识别结果（{duration}秒音频）：")
                print(f"   最终标签：{pred_label}")
                print(f"   置信度：{pred_conf}（{pred_conf*100:.2f}%）")
                print("\n📊 所有情感概率分布（降序）：")
                sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                for label, prob in sorted_probs:
                    print(f"   {label}: {prob}（{prob*100:.2f}%）")
            except Exception as e:
                print(f"❌ 识别失败：{e}（请检查麦克风是否正常）")
        
        elif choice == "3":
            print("👋 退出程序，再见！")
            break
        
        else:
            print("❌ 无效选项，请输入1/2/3！")