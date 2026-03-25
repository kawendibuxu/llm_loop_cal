# models/mllm_based_detector.py
import os
import cv2
import torch
import numpy as np
from models.attribute_predictor import AttributePredictor  # 导入我们新的模型


class MLLMBasedLoopDetector:
    """
    基于MLLM的回环检测器
    阶段1: 使用AttributePredictor生成结构化描述
    阶段2: 将描述发送给MLLM进行最终判断 (此步骤将在下一阶段实现)
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # 使用新的属性预测器作为“语义编码器”
        self.semantic_encoder = AttributePredictor().to(device)
        # 临时加载预训练权重（如果有的话，或保持随机初始化进行测试）
        # self.semantic_encoder.load_state_dict(torch.load('path_to_pretrained_attr_pred.pth'))

        # 存储历史帧的语义描述
        self.database_descriptions = []
        self.database_poses = []

    def add_frame_to_database(self, image_path, pose):
        """
        将新帧的语义描述和位姿添加到数据库中
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 使用语义编码器生成描述
        description = self.semantic_encoder.predict_and_format(image_rgb)

        self.database_descriptions.append(description)
        self.database_poses.append(pose)
        print(f"[INFO] Added description to DB: {description[:50]}...")  # 打印描述的前50个字符

    def detect_loop(self, query_image_path, query_pose):
        """
        检测回环：比较当前帧的描述与历史帧的描述
        """
        query_image = cv2.imread(query_image_path)
        query_image_rgb = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

        query_description = self.semantic_encoder.predict_and_format(query_image_rgb)
        print(f"[QUERY] Current description: {query_description}")

        # --- 这里是关键：如何比较两个结构化描述？ ---
        # 方案1: 简单的字符串相似度 (非常粗糙)
        # 方案2: 将描述向量化后计算距离 (更好)
        # 方案3: 将描述和候选描述一起发给MLLM进行判断 (我们的最终目标)

        # 我们先实现方案2作为过渡
        best_similarity = -1
        best_match_idx = -1
        for i, db_desc in enumerate(self.database_descriptions):
            # 这里可以用更复杂的文本相似度计算，如Sentence-BERT
            # 为了演示，我们用一个非常简单的Jaccard相似度
            similarity = self._simple_text_similarity(query_description, db_desc)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = i

        # 设定一个阈值
        if best_similarity > 0.3:  # 阈值需要根据实际情况调整
            print(f"[LOOP DETECTED] Match with DB index {best_match_idx}, Similarity: {best_similarity:.3f}")
            # TODO: 在这里，我们应该调用一个函数，将两个描述发送给MLLM
            # mllm_output = self.query_mllm(query_description, self.database_descriptions[best_match_idx])
            # is_valid = self.parse_mllm_output(mllm_output)
            # return is_valid, best_match_idx, ...
            return True, best_match_idx, np.eye(4)  # Placeholder
        else:
            print(f"[NO LOOP] Best similarity was {best_similarity:.3f}, below threshold.")
            return False, None, None

    def _simple_text_similarity(self, desc1, desc2):
        """
        一个非常简单的文本相似度计算示例
        """
        set1 = set(desc1.split())
        set2 = set(desc2.split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if union else 0