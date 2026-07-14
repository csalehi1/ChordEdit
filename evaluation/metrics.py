import re

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class QwenPreservationScorer(torch.nn.Module):
    """
    Qwen-VL judge for how well unedited regions survived an edit.
    """

    PROMPT_TEMPLATE = (
        'The requested edit was: "{instruction}".\n\n'
        "Compare the original image with the edited image. Ignore the region/object that was "
        "supposed to change, and rate how well the rest of the image was preserved.\n\n"
        "Output only one number from 0 to 5, where 0 means that the region outside the requested edit is "
        "mostly changed, and 5 means everything except the requested edit is almost perfectly preserved."
    )

    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct"):
        super().__init__()

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(model_id)

    def _run(self, src_image, tgt_image, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "The original image:"},
                    {"type": "image", "image": src_image},
                    {"type": "text", "text": "The edited image:"},
                    {"type": "image", "image": tgt_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]

    def get_score(self, instruction, src_image, tgt_image):
        """
        Args:
            instruction (str): The requested edit, in natural language.
            src_image (PIL.Image): Original image.
            tgt_image (PIL.Image): Edited image.

        Returns:
            int score from 0-5, or "nan" if the model failed to produce one.
        """
        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)

        try:
            answer = self._run(src_image, tgt_image, prompt)
            match = re.search(r"[0-5]", answer)
            if match is None:
                print(f"Could not parse a 0-5 score out of: {answer!r}")
                return "nan"
            return int(match.group())
        except Exception as e:
            print(f"An error occurred: {e}")
            return "nan"
