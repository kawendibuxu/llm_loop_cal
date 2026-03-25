from huggingface_hub import snapshot_download
import os

# 定义模型名称和本地保存路径
model_name = "Qwen/Qwen1.5-1.8B-Chat"
local_model_path = "./models/Qwen1.5-1.8B-Chat"

print(f"开始下载模型 '{model_name}' 到 '{local_model_path}' ...")

# 确保目录存在
os.makedirs(local_model_path, exist_ok=True)

# 使用 snapshot_download 函数下载整个模型仓库
snapshot_download(
    repo_id=model_name,
    local_dir=local_model_path,
    local_dir_use_symlinks=False  # 避免使用符号链接，直接复制文件
)

print("模型下载完成！")