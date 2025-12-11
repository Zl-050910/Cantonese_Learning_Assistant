# 实时语音识别系统最终版

## 系统概述

这是一个基于Whisper模型的实时语音识别系统，具有以下特性：

- **实时语音识别**：麦克风输入，自动语音检测（VAD）
- **繁体转简体**：自动将识别结果转换为简体中文
- **人机互动**：支持简单对话处理
- **多进程架构**：音频采集、VAD检测、语音识别分离处理

## 文件结构

```
real_time_asr_final_package/
├── real_time_asr_final.py     # 主程序
├── config_fixed.py            # 配置文件（音量阈值、VAD参数等）
├── audio_capture.py           # 音频采集模块
├── whisper_recognizer.py      # 语音识别模块（包含繁体转简体）
├── requirements.txt           # Python依赖包
└── whisper_models/
    └── cantonese_model_epoch_20.pt  # Whisper粤语模型
```

## 安装与运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行系统

```bash
python real_time_asr_final.py
```

### 3. 使用方法

1. 运行程序后，系统会显示配置参数
2. 开始说话，系统会自动检测语音
3. 识别结果会显示为简体中文
4. 说"结束"可以退出系统，或按 Ctrl+C

## 配置参数（config_fixed.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| SAMPLE_RATE | 16000 Hz | 音频采样率 |
| VOLUME_THRESHOLD_DB | -60 dB | 音量阈值 |
| VAD_AGGRESSIVENESS | 2 | VAD敏感度（0-3） |
| MIN_SPEECH_DURATION_MS | 300 ms | 最小语音时长 |
| MODEL_PATH | whisper_models/cantonese_model_epoch_20.pt | 模型路径 |
| LANGUAGE | "zh" | 识别语言（中文） |

## 主要特性说明

### 1. 音频处理
- 采样率：16000 Hz
- 帧大小：480样本（30ms）
- VAD检测：WebRTC VAD库
- 音量阈值：-60 dB（0.001线性值）

### 2. 语音识别
- 模型：Whisper粤语微调模型
- 自动繁体转简体：使用zhconv库
- 实时比：显示处理速度与音频时长的比例

### 3. 系统交互
- 自动语音检测和暂停
- 简单对话处理（回显用户输入）
- "结束"指令退出

## 故障排除

### 常见问题

1. **音量阈值过低**
   - 症状：VAD检测到语音但音量过低被过滤
   - 解决：调整`config_fixed.py`中的`VOLUME_THRESHOLD_DB`值

2. **模型加载失败**
   - 症状：无法加载模型文件
   - 解决：检查`whisper_models/cantonese_model_epoch_20.pt`是否存在

3. **音频设备问题**
   - 症状：无法打开麦克风
   - 解决：检查系统音频设置，确保麦克风权限

### 性能优化

1. **降低延迟**
   - 减小`MIN_SPEECH_DURATION_MS`（例如200ms）
   - 调整`VAD_SILENCE_DURATION_MS`（例如300ms）

2. **提高准确性**
   - 增加`VAD_AGGRESSIVENESS`（例如3）
   - 增大`MIN_SPEECH_DURATION_MS`（例如500ms）

## 依赖说明

### 主要依赖
- **openai-whisper**：语音识别核心
- **webrtcvad**：语音活动检测
- **zhconv**：繁体转简体转换
- **sounddevice**：音频采集
- **torch**：PyTorch深度学习框架

### 完整依赖列表
参见`requirements.txt`

## 扩展与定制

### 1. 更换模型
修改`config_fixed.py`中的`MODEL_PATH`指向其他Whisper模型

### 2. 支持其他语言
修改`config_fixed.py`中的`LANGUAGE`参数

### 3. 自定义处理逻辑
修改`real_time_asr_final.py`中的`process_user_input`方法
