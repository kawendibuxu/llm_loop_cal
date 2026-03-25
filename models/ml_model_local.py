import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class LocalMLLMInterface:
    def __init__(self, model_name_or_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading local model: {model_name_or_path}")
        try:
            # 添加 torch_dtype 和 device_map 以节省内存
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,  # 使用半精度浮点数，节省一半显存
                device_map="auto",  # 自动分配模型层到CPU和GPU，非常省内存
                trust_remote_code=True,
            ).to(self.device)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise

    def generate_text(self, prompt: str, max_length: int = 512) -> str:
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from the output
        final_response = generated_text[len(prompt):].strip()
        return final_response
