import pandas as pd
import numpy as np
from pathlib import Path
from imblearn.over_sampling import RandomOverSampler

# ====================== 配置参数（仅需确认路径） ======================
PARQUET_FOLDER = "E:/School/learning/语音信号处理/大作业/data"  # 你的数据文件夹
OUTPUT_PARQUET = "E:/School/learning/语音信号处理/大作业/data/final_balanced_data.parquet"
NEUTRAL_TARGET = 4000  # 中性下采样目标
OVERSAMPLING_THRESHOLD = 1000  # 少数类过采样阈值
# 无需合并多文件的标签（仅保留单个文件数据）
NO_MERGE_LABELS = ["happy", "angry"]

# ====================== 步骤1：读取文件+分离标签（关键调整） ======================
def load_and_split_labels(folder_path):
    parquet_files = list(Path(folder_path).glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"文件夹 {folder_path} 中未找到Parquet文件")
    
    # 存储不同标签的数据
    label_data = {label: [] for label in NO_MERGE_LABELS + ["surprised", "sad", "fearful", "disgusted", "other", "neutral"]}
    no_merge_label_used = {label: False for label in NO_MERGE_LABELS}  # 标记是否已保留单个文件数据

    for file in parquet_files:
        print(f"\n正在处理文件：{file.name}")
        df = pd.read_parquet(file)
        print(f"原始样本数：{len(df)}")

        # 基础清洗（label≠<unk>+confidence≥0.7+有效音频）
        df_clean = df[
            (df["label"] != "<unk>")
            & (df["confidence"] >= 0.7)
            & (df["audio_file"].apply(lambda x: isinstance(x, dict) and "bytes" in x))
        ].copy()
        print(f"清洗后样本数：{len(df_clean)}")

        # 按标签分离数据
        for label in df_clean["label"].unique():
            if label not in label_data:
                continue  # 跳过未知标签
            
            # 对NO_MERGE_LABELS（happy/angry）：仅保留第一个文件的样本
            if label in NO_MERGE_LABELS:
                if not no_merge_label_used[label]:
                    label_data[label].append(df_clean[df_clean["label"] == label].copy())
                    no_merge_label_used[label] = True
                    print(f"  - 保留该文件的{label}样本数：{len(df_clean[df_clean['label'] == label])}")
                else:
                    print(f"  - 已保留其他文件的{label}数据，跳过当前文件")
            # 对其他标签：合并所有文件的样本
            else:
                label_data[label].append(df_clean[df_clean["label"] == label].copy())
                print(f"  - 累计{label}样本数：{sum(len(d) for d in label_data[label])}")

    # 验证happy/angry是否已保留数据
    for label in NO_MERGE_LABELS:
        if not no_merge_label_used[label]:
            raise ValueError(f"未找到{label}标签的有效数据，请检查文件或放宽清洗条件")

    # 合并各标签数据
    final_dfs = []
    for label, dfs in label_data.items():
        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            final_dfs.append(merged)
            print(f"\n{label}最终样本数：{len(merged)}")

    # 合并所有标签的最终数据
    df_all = pd.concat(final_dfs, ignore_index=True)
    print(f"\n所有标签合并后总样本数：{len(df_all)}")
    print("合并后各标签分布：")
    print(df_all["label"].value_counts(), "\n")
    return df_all

# ====================== 步骤2：中性数据下采样 ======================
def downsample_neutral(df_all, target=NEUTRAL_TARGET):
    df_neutral = df_all[df_all["label"] == "neutral"].copy()
    if len(df_neutral) > target:
        df_neutral_down = df_neutral.sample(n=target, random_state=42, replace=False)
        print(f"中性数据下采样：{len(df_neutral)} → {len(df_neutral_down)}")
    else:
        df_neutral_down = df_neutral
        print(f"中性数据不足{target}条，直接保留：{len(df_neutral_down)}")
    return df_neutral_down

# ====================== 步骤3：少数类过采样（仅针对非happy/angry/neutral） ======================
def balance_minority_classes(df_all):
    # 分离需要平衡的少数类（surprised/sad/fearful/disgusted/other）
    minority_labels = ["surprised", "sad", "fearful", "disgusted", "other"]
    df_minority = df_all[df_all["label"].isin(minority_labels)].copy()
    df_happy_angry = df_all[df_all["label"].isin(NO_MERGE_LABELS)].copy()
    df_neutral = df_all[df_all["label"] == "neutral"].copy()

    print("\n需要平衡的少数类初始分布：")
    print(df_minority["label"].value_counts())

    # 对少数类过采样
    minority_counts = df_minority["label"].value_counts()
    labels_to_oversample = minority_counts[minority_counts < OVERSAMPLING_THRESHOLD].index.tolist()
    if labels_to_oversample:
        sampling_strategy = {label: OVERSAMPLING_THRESHOLD for label in labels_to_oversample}
        ros = RandomOverSampler(random_state=42, sampling_strategy=sampling_strategy)
        X = df_minority.index.values.reshape(-1, 1)
        y = df_minority["label"].values
        X_resampled, y_resampled = ros.fit_resample(X, y)
        df_minority_balanced = df_minority.loc[X_resampled.flatten()].reset_index(drop=True)
        print(f"\n少数类过采样完成，平衡后分布：")
        print(df_minority_balanced["label"].value_counts())
    else:
        df_minority_balanced = df_minority
        print("所有少数类均≥阈值，无需过采样")

    # 合并所有部分
    df_final = pd.concat([df_happy_angry, df_minority_balanced, df_neutral], ignore_index=True)
    # 打乱数据
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_final

# ====================== 主流程执行 ======================
if __name__ == "__main__":
    # 1. 读取文件+分离标签（happy/angry仅保留单个文件）
    df_all = load_and_split_labels(PARQUET_FOLDER)
    
    # 2. 中性数据下采样
    df_neutral_down = downsample_neutral(df_all)
    # 替换原始中性数据为下采样后的数据
    df_all = df_all[df_all["label"] != "neutral"].copy()
    df_all = pd.concat([df_all, df_neutral_down], ignore_index=True)
    
    # 3. 平衡少数类
    df_final = balance_minority_classes(df_all)
    
    # 4. 保存最终文件
    df_final.to_parquet(OUTPUT_PARQUET, index=False)
    
    # 输出最终统计
    print("\n" + "="*50)
    print("✅ 数据处理完成！最终统计：")
    print("="*50)
    print(f"文件路径：{OUTPUT_PARQUET}")
    print(f"最终总样本数：{len(df_final)}")
    print(f"保留字段：{df_final.columns.tolist()}")
    print("\n最终各标签平衡分布：")
    print(df_final["label"].value_counts())