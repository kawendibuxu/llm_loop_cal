# datasets/kitti_dataset.py
import os
import numpy as np
import yaml
from torch.utils.data import Dataset


class KittiDataset(Dataset):
    def __init__(self, config_path, sequence_id=None, transform=None):
        print(f"[INFO] Loading configuration from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(f"[ERROR] Error parsing YAML file: {exc}")
                raise

        self.root_dir = config.get('dataset', {}).get('root_dir')
        if self.root_dir is None:
            raise ValueError("Configuration file must contain 'dataset.root_dir'. Please check your config.yaml.")

        if sequence_id is not None:
            self.sequence_id = sequence_id
        else:
            self.sequence_id = config.get('dataset', {}).get('sequence_id', '00')

        print(f"[INFO] Dataset root: {self.root_dir}")
        print(f"[INFO] Sequence ID: {self.sequence_id}")

        # 构建图像目录路径
        self.sequence_dir = os.path.join(self.root_dir, "sequences", self.sequence_id)
        self.image_dir = os.path.join(self.sequence_dir, "image_2")
        # 修改：构建位姿文件路径，指向 poses 文件夹下的 <sequence_id>.txt
        self.pose_file = os.path.join(self.root_dir, "poses", f"{self.sequence_id}.txt")

        if not os.path.exists(self.sequence_dir):
            raise FileNotFoundError(
                f"KITTI sequence directory does not exist: {self.sequence_dir}. \n"
                f"Please ensure the dataset is downloaded and extracted correctly, \n"
                f"and that the 'root_dir' in '{config_path}' is set correctly."
            )
        if not os.path.exists(self.pose_file):  # 修改：检查新的位姿文件路径
            raise FileNotFoundError(f"Pose file not found: {self.pose_file}")

        self.image_filenames = sorted(
            [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        # 加载地面实况位姿
        self.poses = self.load_poses(self.pose_file)  # 传入新的位姿文件路径
        self.transform = transform

        print(f"[INFO] Loaded {len(self)} frames from sequence {self.sequence_id}.")

    def load_poses(self, path):
        """Loads ground truth poses (T_w_cam) from a text file."""
        print(f"[INFO] Loading poses from: {path}")
        poses = []
        with open(path, "r", encoding='utf-8') as f:
            for line in f.readlines():
                if line.strip():  # Skip empty lines
                    T_w_cam = np.array(line.split(), dtype=np.float32).reshape(3, 4)
                    # Convert to 4x4 matrix
                    T_w_cam_4x4 = np.eye(4)
                    T_w_cam_4x4[:3, :] = T_w_cam
                    poses.append(T_w_cam_4x4)
        return poses

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        pose = self.poses[idx]

        return img_path, pose