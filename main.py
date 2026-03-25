import logging
import argparse
from configs.config_loader import load_config
from datasets.kitti_dataset import KittiDataset
from models.mllm_based_detector import MLLMBasedLoopDetector
from models.attribute_predictor import AttributePredictor

def main(config_path: str, local_model_path: str):
    logger = logging.getLogger(__name__)
    logger.info("Starting MLLM-based loop closure detection with LOCAL MODEL...")

    # 加载配置文件
    config = load_config(config_path)

    # 创建数据集实例
    dataset = KittiDataset(config_path=config_path)

    # 使用本地模型路径初始化
    detector = MLLMBasedLoopDetector(dataset=dataset, local_model_path=local_model_path)

    for i in range(len(dataset)):
        logger.info(f"--- Processing Frame {i} ---")

        image_path, pose = dataset[i]

        description = AttributePredictor.predict_and_format(image_path)

        is_loop, matched_idx = detector.detect_loop(current_frame_idx=i, current_description=description)

        if is_loop:
            print(f"[QUERY] Current description: {description}")
            print(f"*** LOOP FOUND between frame {i} and frame {matched_idx} ***")

        detector.add_to_database(description, pose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MLLM-based Loop Closure Detection.')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to local LLM model (e.g., Qwen/Qwen1.5-1.8B-Chat)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(config_path=args.config, local_model_path=args.model_path)