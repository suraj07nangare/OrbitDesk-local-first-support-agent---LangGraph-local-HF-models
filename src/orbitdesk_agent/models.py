import time
from typing import Dict

from . import config
from .logging_utils import get_logger

logger = get_logger(__name__)


class GenerationModel:
    def __init__(
        self,
        model_name: str = config.GENERATION_MODEL_NAME,
        revision: str = config.GENERATION_MODEL_REVISION,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        start = time.perf_counter()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.float32,
        )
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.load_time_seconds = time.perf_counter() - start
        self.revision_hash = getattr(self.model.config, "_commit_hash", None) or revision
        logger.info(
            "Loaded generation model %s (revision=%s) on %s in %.2fs",
            model_name,
            self.revision_hash,
            self.device,
            self.load_time_seconds,
        )

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = config.MAX_NEW_TOKENS) -> Dict:
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency_seconds = time.perf_counter() - start

        generated_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return {"text": text, "latency_seconds": latency_seconds}
