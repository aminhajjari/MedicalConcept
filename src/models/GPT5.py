from openai import OpenAI
import base64

class GPT5:
    def __init__(self, model, reasoning_effort="none") -> None:
        self.client = OpenAI()
        self.model = model
        self.reasoning_effort = reasoning_effort

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def inference_text(self, instruction, query, max_new_tokens=16, demos=None):
        user_content = query if not demos else "\n\n".join(demos) + "\n\n" + query
        completion = self.client.chat.completions.create(
            model=self.model,                          # actual model = whatever ckpt you pass
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_content},
            ],
            reasoning_effort=self.reasoning_effort,    # "none" = direct answer for GPT-5.5
            max_completion_tokens=max_new_tokens,      # NOT max_tokens for GPT-5 family
        )
        return completion.choices[0].message.content

    def inference_vision(self, instruction, query, base64_image, max_new_tokens):
        # NOTE: unused for the classifier. temperature/max_tokens here are NOT
        # compatible with GPT-5.5 — only use this with a non-reasoning model.
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": instruction},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "low"
                        }},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=max_new_tokens,
        )
        print(f"Assistant: {completion.choices[0].message.content}")
        return completion.choices[0].message.content


if __name__ == "__main__":
    gpt = GPT5(model="gpt-5.5-2026-04-23")
    instruction = "You're a english doctor, make a good choice based on the question and options. You need to answer the letter of the option without further explanations."
    query = """###Question: What is the type of skin lesion that is associated with the presence following dermoscopic concepts: {}. ###Options: A. Nevus\nB. Melanoma. ###Answer:"""
    input_query = query.format("atypical pigment network, regular streaks, regular dots and globules, regression structures")
    response = gpt.inference_text(instruction=instruction, query=input_query, max_new_tokens=16)
    print(response)
