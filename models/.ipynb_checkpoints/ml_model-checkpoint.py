import openai
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MLLMInterface:
    """
    一个封装了与大型语言模型交互的接口类。
    它负责构造提示(prompt)并发送给LLM，然后解析返回结果。
    """

    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        """
        初始化LLM接口。

        Args:
            api_key (str): LLM服务的API密钥 (例如 OpenAI API Key)。
            model_name (str): 使用的模型名称，默认为 "gpt-3.5-turbo"。
        """
        # 设置API密钥
        openai.api_key = api_key
        self.model_name = model_name

    def check_loop_closure(self, query_description: str, candidate_description: str) -> tuple[bool, float]:
        """
        询问LLM，判断query和candidate是否为同一地点。

        Args:
            query_description (str): 查询帧的描述。
            candidate_description (str): 候选帧的描述。

        Returns:
            tuple[bool, float]: 一个元组，第一个元素是True/False表示是否为回环，
                                第二个元素是一个0-1之间的置信度分数。
        """
        # --- 1. 构造提示(Prompt) ---
        # 这个prompt的设计至关重要，它告诉LLM如何思考和回答问题。
        prompt = f"""
        You are an expert in visual place recognition. I will provide you with descriptions of two scenes captured by a camera.

        Query Scene Description: "{query_description}"
        Candidate Scene Description: "{candidate_description}"

        Your task is to determine if these two scene descriptions represent the same physical location. 
        Please analyze the scene type, key objects, their attributes (like color), and any spatial relationships mentioned.

        Provide your answer in the following strict JSON format ONLY, without any additional text or explanation:
        {{
          "is_same_location": true/false,
          "confidence": <a number between 0.0 and 1.0>
        }}

        Example of a high confidence match:
        Query: "SCENE_TYPE: parking_lot. KEY_OBJECT: A white car."
        Candidate: "SCENE_TYPE: parking_lot. KEY_OBJECT: A white car."
        Response: {{"is_same_location": true, "confidence": 0.9}}

        Example of a low confidence match:
        Query: "SCENE_TYPE: parking_lot. KEY_OBJECT: A white car."
        Candidate: "SCENE_TYPE: parking_lot. KEY_OBJECT: A black car."
        Response: {{"is_same_location": false, "confidence": 0.3}}
        """

        try:
            # --- 2. 发送请求到LLM ---
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # 降低随机性，使结果更稳定
                max_tokens=150,  # 限制响应长度
            )

            # --- 3. 解析LLM的响应 ---
            # 获取LLM返回的文本
            content = response.choices[0].message['content'].strip()

            # 尝试解析为JSON
            import json
            result = json.loads(content)

            is_loop = bool(result.get("is_same_location", False))
            confidence = float(result.get("confidence", 0.0))

            logger.info(f"MLLM evaluated pair. Is Loop: {is_loop}, Confidence: {confidence:.3f}")
            return is_loop, confidence

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse MLLM response as JSON: {content}. Error: {e}")
            # 如果解析失败，返回一个安全的默认值
            return False, 0.0
        except Exception as e:
            logger.error(f"Error calling MLLM: {e}")
            # 发生任何错误，都返回False
            return False, 0.0