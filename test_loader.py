# test_loader.py
from datasets.kitti_dataset import KittiDataset

# 使用修改后的数据集加载器
dataset = KittiDataset(config_path="configs/config.yaml", sequence_id="00")

print(f"Dataset length: {len(dataset)}")

# 尝试加载第一帧
first_img_path, first_pose = dataset[0]
print(f"First image path: {first_img_path}")
print(f"First pose shape: {first_pose.shape}")
print(f"First pose:\n{first_pose}")