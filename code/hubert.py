import pandas as pd
import numpy as np
import torch
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, cohen_kappa_score, matthews_corrcoef, log_loss,
    classification_report
)
from transformers import (
    HubertForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset as HFDataset
import soundfile as sf
import io

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16000
LOCAL_HUBERT_PATH = "/root/autodl-tmp/大作业/model/chinese-hubert-base"
EPOCHS = 5
BATCH_SIZE = 64
SAMPLE_SIZE = 100
TEST_MODE = "sample"

print(f"使用设备: {DEVICE}")
print(f"本地HuBERT路径: {LOCAL_HUBERT_PATH}")

# ====================== 加载数据 ======================
df = pd.read_parquet("/root/autodl-tmp/大作业/data/final_balanced_data.parquet")
df['audio_bytes'] = df['audio_file'].apply(
    lambda x: x['bytes'] if isinstance(x, dict) and 'bytes' in x else None
)
df = df[df['audio_bytes'].notna()].reset_index(drop=True)

raw_labels = df['label'].unique().tolist()
label2id = {l: i for i, l in enumerate(raw_labels)}
id2label = {i: l for l, i in label2id.items()}
df['label_id'] = df['label'].map(label2id)
num_classes = len(raw_labels)

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label_id']
)


# ====================== 处理音频 ======================
def load_audio_waveform(audio_bytes):
    audio_io = io.BytesIO(audio_bytes)
    y, _ = sf.read(audio_io)
    if len(y.shape) > 1:
        y = y.mean(axis=1)
    return y


hf_train = HFDataset.from_pandas(train_df[['audio_bytes', 'label_id']])
hf_test = HFDataset.from_pandas(test_df[['audio_bytes', 'label_id']])

hf_train = hf_train.map(lambda x: {"audio": load_audio_waveform(x["audio_bytes"])})
hf_test = hf_test.map(lambda x: {"audio": load_audio_waveform(x["audio_bytes"])})


# ====================== HuBERT 训练 ======================
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    LOCAL_HUBERT_PATH, local_files_only=True
)

def preprocess(examples):
    inputs = feature_extractor(
        examples["audio"], sampling_rate=SR, padding="max_length",
        max_length=SR * 5, truncation=True, return_tensors="pt"
    )
    return {"input_values": inputs["input_values"].squeeze(0)}

hf_train_proc = hf_train.map(preprocess, remove_columns=["audio_bytes", "audio"])
hf_test_proc = hf_test.map(preprocess, remove_columns=["audio_bytes", "audio"])

hf_train_proc = hf_train_proc.rename_column("label_id", "label")
hf_test_proc = hf_test_proc.rename_column("label_id", "label")


# ====================== 构建模型 ======================
model = HubertForSequenceClassification.from_pretrained(
    LOCAL_HUBERT_PATH,
    num_labels=num_classes,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
    local_files_only=True
).to(DEVICE)

model.hubert.feature_extractor._freeze_parameters()


# ====================== 计算指标 ======================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    preds_prob = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

    acc = accuracy_score(labels, preds)
    f1_m = f1_score(labels, preds, average='macro', zero_division=0)
    f1_w = f1_score(labels, preds, average='weighted', zero_division=0)

    return {
        "accuracy": acc,
        "f1_macro": f1_m,
        "f1_weighted": f1_w,
    }


# ====================== Early Stopping ======================
training_args = TrainingArguments(
    output_dir="./hubert_results",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",     # 监控 f1_macro
    greater_is_better=True,
    learning_rate=5e-5,
    weight_decay=0.01,
    bf16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_train_proc,
    eval_dataset=hf_test_proc,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)


# ====================== 开始训练 ======================
print("\n===== 开始训练 HuBERT（100轮 + 早停）=====")
trainer.train()


# ====================== 保存最优模型 ======================
trainer.save_model("./best_model_hubert")
print("\n✅ 最优 HuBERT 模型已保存： ./best_model_hubert")
