import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class Qwen:
    def __init__(self, ckpt="Qwen/Qwen2.5-72B-Instruct") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt,
            quantization_config=bnb,
            device_map={"": 0},
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    def get_prompt(self, instruction, query, demos=None):
        if demos is not None:
            demonstration = "Consider the following examples:\n"
            for d in demos:
                demonstration += f"{d}\n"
            prompt = instruction + demonstration + query
        else:
            prompt = instruction + query
        return prompt

    def get_length_of_prompt(self, original_prompt: str) -> str:
        return len(original_prompt)

    def predict(self, prompt, max_new_tokens):
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        )
        if not torch.is_tensor(input_ids):
            input_ids = input_ids["input_ids"]
        input_ids = input_ids.to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
