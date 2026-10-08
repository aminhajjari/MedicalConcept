import torch


class MedGemma:
    """MedGemma-4b-it (Gemma-3 multimodal) used as a TEXT-ONLY concept classifier.

    Mirrors the interface of MMedLlama3 (get_prompt / get_length_of_prompt /
    predict) so it drops into run_x_to_c_to_y.py with no other changes.
    No image is ever passed; only the dermoscopic-concept text is classified.
    """

    def __init__(self, ckpt="google/medgemma-4b-it") -> None:
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer  # keep .eos_token_id available

        # 4b-it is image-text-to-text; load the multimodal LM and use it text-only.
        try:
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                ckpt, torch_dtype=torch.bfloat16, device_map="cuda")
        except Exception:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                ckpt, torch_dtype=torch.bfloat16, device_map="cuda")

    def get_prompt(self, instruction, query, demos=None):
        # Identical string-building to MMedLlama3 so the comparison stays fair.
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
        # Text-only chat turn via the Gemma-3 processor chat template (no image).
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        in_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][in_len:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()
