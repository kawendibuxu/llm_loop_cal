# main.py
import os
import yaml
from datasets.kitti_dataset import KittiDataset
# 暂时使用基线检测器
from models.baseline_detector import BaselineLoopDetector


def main():
    CONFIG_PATH = "configs/config.yaml"
    print(f"[INFO] Starting main program with config: {CONFIG_PATH}")

    # --- 初始化 ---
    dataset = KittiDataset(config_path=CONFIG_PATH)
    # 使用基线检测器
    detector = BaselineLoopDetector()

    print("[INFO] Starting BASELINE loop closure detection simulation...")

    N_FRAMES_TO_PROCESS = 100  # 先处理少量帧进行测试

    for i in range(N_FRAMES_TO_PROCESS):
        print(f"\n--- Processing Frame {i} ---")
        image_path, pose = dataset[i]

        if i > 10:  # 等待一些历史帧
            is_loop, match_idx, transform = detector.detect_loop(image_path, pose)
            if is_loop:
                print(f"*** BASELINE LOOP FOUND between frame {i} and frame {match_idx} ***\n")

        detector.add_frame_to_database(image_path, pose)


if __name__ == "__main__":
    main()