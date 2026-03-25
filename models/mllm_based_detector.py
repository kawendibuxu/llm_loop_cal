import numpy as np
import logging
from sklearn.metrics import jaccard_score
from datasets.kitti_dataset import KittiDataset
from models.attribute_predictor import AttributePredictor
from models.ml_model_local import LocalMLLMInterface  # 导入本地接口

logger = logging.getLogger(__name__)


class MLLMBasedLoopDetector:
    def __init__(self, dataset: KittiDataset, local_model_path: str):  # 参数改为模型路径
        self.dataset = dataset
        self.description_db = []
        self.poses_db = []

        self.attribute_predictor = AttributePredictor()

        # 初始化本地MLLM接口
        self.mllm_interface = LocalMLLMInterface(model_name_or_path=local_model_path)

        self.temporal_window = 30

    def detect_loop(self, current_frame_idx: int, current_description: str) -> tuple[bool, int | None]:
        start_idx = max(0, current_frame_idx - self.temporal_window)

        best_candidate_idx = None
        highest_confidence = 0.0
        found_loop = False

        for i in range(start_idx, current_frame_idx):
            candidate_description = self.description_db[i]

            # 使用本地模型进行判断
            is_loop, confidence = self.mllm_interface.check_loop_closure(
                query_description=current_description,
                candidate_description=candidate_description
            )

            if is_loop and confidence > highest_confidence:
                highest_confidence = confidence
                best_candidate_idx = i
                found_loop = True

        if found_loop and highest_confidence > 0.7:
            return True, best_candidate_idx
        else:
            return False, None

    def add_to_database(self, description: str, pose: np.ndarray):
        self.description_db.append(description)
        self.poses_db.append(pose)