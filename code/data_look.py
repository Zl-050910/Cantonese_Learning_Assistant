import pandas as pd
import sys

# -------------------------- 1. 修复路径+读取数据 --------------------------
# df = pd.read_parquet(r"E:\School\learning\语音信号处理\大作业\data\train-00030-of-00045.parquet")
# df = pd.read_parquet(r"E:\School\learning\语音信号处理\大作业\data\train-00000-of-00045.parquet")
df = pd.read_parquet(r"E:\School\learning\语音信号处理\大作业\data\train-00001-of-00045.parquet")
# df = pd.read_parquet(r"E:\School\learning\语音信号处理\大作业\final_balanced_data.parquet")


# -------------------------- 2. 设置Pandas显示参数（关键：取消截断） --------------------------
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', 100)      # 一次显示100行（可调整）
pd.set_option('display.width', None)        # 不限制显示宽度
pd.set_option('display.max_colwidth', 100)  # 每列最大显示宽度（避免文本/路径截断）

# -------------------------- 3. 基础信息概览 --------------------------
print("="*80)
print("📊 数据基本信息")
print("="*80)
print(f"总样本数：{len(df)}")
print(f"所有字段：{df.columns.tolist()}")  # 输出如：['audio_file'音频路劲, 'duration'时长,'label'情感标签,'confidence'置信度, 'text'转录文本]
print("\n📈 各情感标签数量：")
print(df["label"].value_counts())

# -------------------------- 4. 完整查看指定字段（核心） --------------------------
# 只保留需要的字段，避免无关列干扰
target_cols = ['audio_file', 'duration', 'label', 'confidence', 'text']
df_target = df[target_cols].copy()

# 方案A：查看前N行（比如前20行，快速预览）
print("\n" + "="*80)
print("📋 前20行数据详情（完整字段）")
print("="*80)
print(df_target.head(20))

# 方案B：分页查看全量数据（按回车看_next页，按q退出）
print("\n" + "="*80)
print("📖 分页查看全量数据（每页50行），按回车翻页，输入q退出")
print("="*80)
page_size = 50
total_pages = (len(df_target) + page_size - 1) // page_size  # 总页数

for page in range(total_pages):
    start = page * page_size
    end = min((page + 1) * page_size, len(df_target))
    print(f"\n🔍 第{page+1}/{total_pages}页（行{start+1}至{end}）")
    print(df_target.iloc[start:end])
    
    # 交互控制
    user_input = input("\n按【回车】查看下一页 | 输入【q】退出：").strip().lower()
    if user_input == "q":
        print("退出查看！")
        sys.exit()

print("已查看全部数据！")

# import pandas as pd

# # -------------------------- 1. 修复路径+读取数据 --------------------------
# df = pd.read_parquet(r"E:\School\learning\语音信号处理\大作业\data\train-00000-of-00045.parquet")

# # -------------------------- 2. 只保留需要的字段 --------------------------
# target_cols = ['audio_file', 'duration', 'label', 'confidence', 'text']
# df_target = df[target_cols].copy()

# # -------------------------- 3. 导出为Excel（推荐，支持筛选） --------------------------
# excel_path = r"E:\School\learning\语音信号处理\大作业\data_full_info.xlsx"
# df_target.to_excel(excel_path, index=False)  # index=False不导出行号，更整洁
# print(f"✅ Excel文件已导出：{excel_path}")

# # -------------------------- 4. 备选：导出为CSV（体积更小，记事本可打开） --------------------------
# csv_path = r"E:\School\learning\语音信号处理\大作业\data_full_info.csv"
# df_target.to_csv(csv_path, index=False, encoding='utf-8-sig')  # utf-8-sig避免中文乱码
# print(f"✅ CSV文件已导出：{csv_path}")