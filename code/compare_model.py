import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import io
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, cohen_kappa_score, matthews_corrcoef, log_loss,
    classification_report
)
from transformers import (
    Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor,
    HubertForSequenceClassification,
    TrainingArguments, Trainer
)
from datasets import Dataset as HFDataset
import warnings
import os

# 屏蔽无关警告
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None")
warnings.filterwarnings("ignore", message="Passing `gradient_checkpointing` to a config initialization is deprecated")

# ====================== 全局配置 ======================
# 本地模型路径
LOCAL_WAV2VEC2_PATH = "/root/autodl-tmp/大作业/model/wav2vec2-base"
LOCAL_HUBERT_PATH = "/root/autodl-tmp/大作业/model/chinese-hubert-base"

# 基础配置
TEST_MODE = "sample"
# TEST_MODE = "full"
SAMPLE_SIZE = 100
EPOCHS = 50
BATCH_SIZE = 128
SR = 16000  # 音频采样率，需与模型训练时一致
raw_labels = []
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 打印配置信息
print(f"===== 运行配置 =====")
print(f"使用设备: {DEVICE}")
print(f"本地Wav2Vec2路径: {LOCAL_WAV2VEC2_PATH}")
print(f"本地HuBERT路径: {LOCAL_HUBERT_PATH}")
print(f"执行顺序: 本地Wav2Vec2 → 本地Chinese-HuBERT → E-CNN\n")

# ====================== 1. 数据加载与预处理 ======================
# 读取数据集
# df = pd.read_parquet("/root/autodl-tmp/大作业/data/train-00000-of-00045.parquet")
df = pd.read_parquet("/root/autodl-tmp/大作业/data/final_balanced_data.parquet")

# 提取音频二进制数据（过滤无效样本）
df['audio_bytes'] = df['audio_file'].apply(
    lambda x: x['bytes'] if isinstance(x, dict) and 'bytes' in x else None
)
df = df[df['audio_bytes'].notna()].reset_index(drop=True)

# 标签映射
raw_labels = df['label'].unique().tolist()
label2id = {label: idx for idx, label in enumerate(raw_labels)}
id2label = {idx: label for label, idx in label2id.items()}
df['label_id'] = df['label'].map(label2id)
num_classes = len(raw_labels)
print(f"数据集信息：样本数={len(df)} | 类别数={num_classes} | 标签={raw_labels}\n")

# 划分训练/验证集
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label_id']
)

# ====================== 2. 数据格式转换 ======================
# 2.1 E-CNN特征提取函数（最后执行）
def extract_melspec(audio_bytes):
    audio_io = io.BytesIO(audio_bytes)
    y, _ = sf.read(audio_io)
    if len(y.shape) > 1:
        y = y.mean(axis=1)  # 转单声道
    # 提取梅尔频谱
    mel_spec = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=40, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # 统一长度为100帧
    if mel_spec_db.shape[1] < 100:
        mel_spec_db = np.pad(mel_spec_db, ((0,0), (0, 100-mel_spec_db.shape[1])), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :100]
    return mel_spec_db[np.newaxis, ...]  # [1, 40, 100]

# 预处理E-CNN数据（提前处理，最后使用）
train_df['melspec'] = train_df['audio_bytes'].apply(extract_melspec)
test_df['melspec'] = test_df['audio_bytes'].apply(extract_melspec)
X_train_cnn = np.array(train_df['melspec'].tolist(), dtype=np.float32)
y_train_cnn = np.array(train_df['label_id'].tolist(), dtype=np.int64)
X_test_cnn = np.array(test_df['melspec'].tolist(), dtype=np.float32)
y_test_cnn = np.array(test_df['label_id'].tolist(), dtype=np.int64)

# 2.2 预训练模型音频加载函数
def load_audio_waveform(audio_bytes):
    audio_io = io.BytesIO(audio_bytes)
    y, _ = sf.read(audio_io)
    if len(y.shape) > 1:
        y = y.mean(axis=1)  # 转单声道
    return y

# 转换为HuggingFace Dataset格式（预训练模型用）
hf_train = HFDataset.from_pandas(train_df[['audio_bytes', 'label_id']])
hf_test = HFDataset.from_pandas(test_df[['audio_bytes', 'label_id']])
hf_train = hf_train.map(lambda x: {"audio": load_audio_waveform(x['audio_bytes'])})
hf_test = hf_test.map(lambda x: {"audio": load_audio_waveform(x['audio_bytes'])})

# ====================== 3. 通用函数 ======================
# 3.1 通用评估指标函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # 区分E-CNN（直接传preds）和预训练模型（传logits）
    if len(logits.shape) == 1:
        preds = logits
        preds_probs = np.zeros((len(preds), num_classes))
        preds_probs[np.arange(len(preds)), preds.astype(int)] = 1.0
    else:
        preds = np.argmax(logits, axis=-1)
        preds_probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    
    # 计算核心指标
    accuracy = accuracy_score(labels, preds)
    precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
    recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    kappa = cohen_kappa_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    
    # 计算Log Loss
    try:
        logloss = log_loss(labels, preds_probs, labels=np.arange(num_classes))
    except Exception as e:
        logloss = np.nan
        print(f"计算Log Loss时出错: {e}")
    
    # 打印混淆矩阵
    cm = confusion_matrix(labels, preds)
    print("\n混淆矩阵（行=真实标签，列=预测标签）：")
    print(cm)
    
    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_micro": f1_micro,
        "kappa": kappa,
        "mcc": mcc,
        "log_loss": logloss
    }

# 3.2 通用验证函数（修复维度问题）
def evaluate_model(model_name, model, data, true_labels, process_func=None):
    # 抽样验证
    if TEST_MODE == "sample":
        sample_indices = np.random.choice(len(data), min(SAMPLE_SIZE, len(data)), replace=False)
        data_sample = [data[i] for i in sample_indices]
        true_labels_sample = [true_labels[i] for i in sample_indices]
        print(f"\n===== {model_name} 抽样验证（{len(sample_indices)}个样本） =====")
    else:
        data_sample = data
        true_labels_sample = true_labels
        print(f"\n===== {model_name} 全量验证（{len(data)}个样本） =====")

    model.eval()
    y_pred = []

    with torch.no_grad():
        for i in range(len(data_sample)):

            # =============== 1. 处理 E-CNN ===============
            if model_name == "E-CNN":
                # data_sample[i] 为 [1,40,100]，这里扩成 batch
                input_data = torch.from_numpy(
                    data_sample[i][np.newaxis, ...]
                ).to(DEVICE)             # [1,1,40,100]
                outputs = model(input_data)
                pred_id = torch.argmax(outputs, dim=-1).item()
                y_pred.append(pred_id)
                continue

            # =============== 2. 处理 Wav2Vec2/HuBERT ===============
            processed = process_func(data_sample[i])  # 得到 input_values:[1,80000] 或 [80000]

            input_values = processed["input_values"]

            # ---- 修复所有可能的错误维度 ----
            # 去掉多余维度（例如 [1,1,80000] → [80000]）
            input_values = input_values.squeeze()

            # 保证至少是 1D
            if input_values.dim() == 0:
                raise ValueError("input_values 最终为 0D，不合法")

            # 如果为 1D → 加 batch dim
            if input_values.dim() == 1:
                input_values = input_values.unsqueeze(0)   # [1, seq]

            # 如果为 2D → OK（符合 Wav2Vec2 要求）
            # 如果为 3D（例如 [1,1,seq]）→ 去掉中间 channel dim
            if input_values.dim() == 3 and input_values.size(1) == 1:
                input_values = input_values.squeeze(1)     # [1, seq]

            # 最终保证 input_values 为 [batch, seq_len]
            input_values = input_values.to(DEVICE).float()

            # ----------- 推理 -----------
            outputs = model(input_values=input_values)
            pred_id = torch.argmax(outputs.logits, dim=-1).item()
            y_pred.append(pred_id)

    # 计算指标
    eval_pred = (np.array(y_pred), np.array(true_labels_sample))
    metrics = compute_metrics(eval_pred)

    # 打印分类报告
    print(f"\n{model_name} 逐类分类报告：")
    # print(classification_report(
    #     true_labels_sample, y_pred,
    #     target_names=raw_labels,
    #     zero_division=0
    # ))
    # 获取本次出现过的标签
    unique_labels = sorted(set(true_labels_sample))

    # 动态生成 target_names（只拿出现过的部分）
    target_names_dynamic = [raw_labels[i] for i in unique_labels]

    print(classification_report(
        true_labels_sample,
        y_pred,
        labels=unique_labels,           # ← 新增！
        target_names=target_names_dynamic,
        zero_division=0
    ))

    # 打印核心指标
    print(f"\n{model_name} 核心指标汇总：")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics


# ====================== 4. 预训练模型训练函数 ======================
# 4.1 本地Wav2Vec2模型训练
def train_wav2vec2():
    # 加载本地特征提取器
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        LOCAL_WAV2VEC2_PATH,
        local_files_only=True
    )
    
    # 预处理函数（修复维度：只去掉batch维度，保留sequence_length）
    def preprocess(examples):
        inputs = feature_extractor(
            examples["audio"], sampling_rate=SR, padding="max_length",
            max_length=SR*5, truncation=True, return_tensors="pt"
        )
        # 关键修复：squeeze(0) 只去掉batch维度，输出[80000]
        return {"input_values": inputs["input_values"].squeeze(0)}
    
    # 处理数据集
    hf_train_proc = hf_train.map(preprocess, remove_columns=["audio_bytes", "audio"])
    hf_test_proc = hf_test.map(preprocess, remove_columns=["audio_bytes", "audio"])
    hf_train_proc = hf_train_proc.rename_column("label_id", "label")
    hf_test_proc = hf_test_proc.rename_column("label_id", "label")
    
    # 加载本地模型（权重初始化警告是正常的）
    print("\n⚠️ 权重初始化警告：本地模型无分类头，自动创建适配当前任务的分类层")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        LOCAL_WAV2VEC2_PATH,
        num_labels=num_classes,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        local_files_only=True
    ).to(DEVICE)
    
    # 冻结特征提取层
    model.wav2vec2.feature_extractor._freeze_parameters()
    
    # 训练参数（修复eval_strategy，移除废弃参数）
    training_args = TrainingArguments(
        output_dir="./wav2vec2_results",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",          # 替换evaluation_strategy
        logging_strategy="epoch",
        save_strategy="no",
        bf16=True if DEVICE.type == "cuda" else False,
        disable_tqdm=False,
        dataloader_num_workers=8,
        learning_rate=5e-5,
        weight_decay=0.01
    )
    
    # 构建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train_proc,
        eval_dataset=hf_test_proc,
        compute_metrics=compute_metrics,
    )
    
    # 开始训练
    print("\n===== 【第一步】开始本地Wav2Vec2训练 =====")
    trainer.train()
    return model, feature_extractor

# 4.2 本地Chinese-HuBERT模型训练
def train_hubert():
    # 加载本地特征提取器（HuBERT与Wav2Vec2特征提取器兼容）
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        LOCAL_HUBERT_PATH,
        local_files_only=True
    )
    
    # 预处理函数（修复维度）
    def preprocess(examples):
        inputs = feature_extractor(
            examples["audio"], sampling_rate=SR, padding="max_length",
            max_length=SR*5, truncation=True, return_tensors="pt"
        )
        return {"input_values": inputs["input_values"].squeeze(0)}
    
    # 处理数据集
    hf_train_proc = hf_train.map(preprocess, remove_columns=["audio_bytes", "audio"])
    hf_test_proc = hf_test.map(preprocess, remove_columns=["audio_bytes", "audio"])
    hf_train_proc = hf_train_proc.rename_column("label_id", "label")
    hf_test_proc = hf_test_proc.rename_column("label_id", "label")
    
    # 加载本地模型
    print("\n⚠️ 权重初始化警告：本地模型无分类头，自动创建适配当前任务的分类层")
    model = HubertForSequenceClassification.from_pretrained(
        LOCAL_HUBERT_PATH,
        num_labels=num_classes,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        local_files_only=True
    ).to(DEVICE)
    
    # 冻结特征提取层
    model.hubert.feature_extractor._freeze_parameters()
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./hubert_results",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        bf16=True if DEVICE.type == "cuda" else False,
        disable_tqdm=False,
        dataloader_num_workers=8,
        learning_rate=5e-5,
        weight_decay=0.01
    )
    
    # 构建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train_proc,
        eval_dataset=hf_test_proc,
        compute_metrics=compute_metrics,
    )
    
    # 开始训练
    print("\n===== 【第二步】开始本地Chinese-HuBERT训练 =====")
    trainer.train()
    return model, feature_extractor

# ====================== 5. E-CNN模型 ======================
class E_CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (3,3), padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, (3,3), padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*10*25, 128)  # 40/2/2=10, 100/2/2=25
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)

class CNNAudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

def train_ecnn():
    # 初始化模型
    model = E_CNN(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 数据加载器
    train_loader = DataLoader(
        CNNAudioDataset(X_train_cnn, y_train_cnn),
        batch_size=32, shuffle=True, num_workers=8, pin_memory=True
    )
    test_loader = DataLoader(
        CNNAudioDataset(X_test_cnn, y_test_cnn),
        batch_size=32, shuffle=False, num_workers=8, pin_memory=True
    )
    
    # 开始训练
    print("\n===== 【第三步】开始E-CNN训练 =====")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                y_true.extend(batch_y.cpu().numpy())
        
        # 计算指标
        ecnn_metrics = compute_metrics((np.array(y_pred), np.array(y_true)))
        print(f"\nE-CNN Epoch {epoch+1} 详细指标：")
        for k, v in ecnn_metrics.items():
            print(f"{k}: {v:.4f}")
    
    return model, DEVICE

# ====================== 6. 主流程 ======================
if __name__ == "__main__":
    # ---------------------- 第一步：Wav2Vec2训练+验证 ----------------------
    wav2vec2_model, wav2vec2_proc = train_wav2vec2()
    # 准备验证数据
    wav2vec2_data = [load_audio_waveform(b) for b in test_df['audio_bytes'].tolist()]
    pretrained_labels = test_df['label_id'].tolist()
    # 预处理函数
    def wav2vec2_preprocess(audio):
        return wav2vec2_proc(
            audio, sampling_rate=SR, padding="max_length",
            max_length=SR*5, truncation=True, return_tensors="pt"
        )
    # 验证Wav2Vec2
    wav2vec2_metrics = evaluate_model(
        "wav2vec2-base(本地)", wav2vec2_model, wav2vec2_data, pretrained_labels,
        process_func=wav2vec2_preprocess
    )

    # ---------------------- 第二步：HuBERT训练+验证 ----------------------
    hubert_model, hubert_proc = train_hubert()
    # 预处理函数
    def hubert_preprocess(audio):
        return hubert_proc(
            audio, sampling_rate=SR, padding="max_length",
            max_length=SR*5, truncation=True, return_tensors="pt"
        )
    # 验证HuBERT
    hubert_metrics = evaluate_model(
        "chinese-hubert-base(本地)", hubert_model, wav2vec2_data, pretrained_labels,
        process_func=hubert_preprocess
    )

    # ---------------------- 第三步：E-CNN训练+验证 ----------------------
    ecnn_model, ecnn_device = train_ecnn()
    # 验证E-CNN
    ecnn_metrics = evaluate_model(
        "E-CNN", ecnn_model, X_test_cnn, y_test_cnn
    )

    # ---------------------- 最终汇总 ----------------------
    print("\n" + "="*120)
    print("最终汇总：三个模型性能对比")
    print("="*120)
    
    results = [
        {"model": "wav2vec2-base(本地)", "metrics": wav2vec2_metrics},
        {"model": "chinese-hubert-base(本地)", "metrics": hubert_metrics},
        {"model": "E-CNN", "metrics": ecnn_metrics}
    ]
    
    for res in results:
        print(f"\n【{res['model']}】")
        for k, v in res['metrics'].items():
            print(f"{k:<15}: {v:.4f}")
    print("="*120)

    # 保存最优模型权重（可选）
    # 先找出加权F1最高的模型
    best_model_name = ""
    best_f1 = 0.0
    best_model = None
    for res in results:
        if res['metrics']['f1_weighted'] > best_f1:
            best_f1 = res['metrics']['f1_weighted']
            best_model_name = res['model']
            if best_model_name == "wav2vec2-base(本地)":
                best_model = wav2vec2_model
            elif best_model_name == "chinese-hubert-base(本地)":
                best_model = hubert_model
            else:
                best_model = ecnn_model
    
    # 保存最优模型
    if best_model_name in ["wav2vec2-base(本地)", "chinese-hubert-base(本地)"]:
        best_model.save_pretrained(f"./best_model_{best_model_name.replace('(本地)', '').strip()}")
    else:
        torch.save(best_model.state_dict(), "./best_model_E-CNN.pth")
    
    print(f"\n✅ 最优模型：{best_model_name}（加权F1={best_f1:.4f}），已保存到本地")
