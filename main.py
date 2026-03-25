# main.py
import os
import yaml
from datasets.kitti_dataset import KittiDataset
# 使用 MLLMBasedLoopDetector
from models.mllm_based_detector import MLLMBasedLoopDetector

def main():
    CONFIG_PATH = "configs/config.yaml"
    print(f"[INFO] Starting main program with config: {CONFIG_PATH}")

    # --- 初始化 ---
    dataset = KittiDataset(config_path=CONFIG_PATH)
    # 使用新的基于MLLM的检测器
    detector = MLLMBasedLoopDetector()

    print("[INFO] Starting MLLM-based loop closure detection simulation...")

    N_FRAMES_TO_PROCESS = 50  # 处理少一点，看效果

    for i in range(N_FRAMES_TO_PROCESS):
        print(f"\n--- Processing Frame {i} ---")
        image_path, pose = dataset[i]

        if i > 5:  # Wait for some history
            is_loop, match_idx, transform = detector.detect_loop(image_path, pose)
            if is_loop:
                print(f"*** LOOP FOUND between frame {i} and frame {match_idx} ***\n")

        detector.add_frame_to_database(image_path, pose)

if __name__ == "__main__":
    main()