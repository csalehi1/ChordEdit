import re

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


class QwenPreservationScorer(torch.nn.Module):
    """
    Qwen-VL judge for how well unedited regions survived an edit.
    """

    PROMPT_TEMPLATE = (
        'The requested edit was: "{instruction}".\n\n'
        "Compare the original image with the edited image. Excluding the requested edit, "
        "rate how well the rest of the image was preserved.\n\n"
        "Output only one number from 0 to 10, where 0 means that the region outside the requested edit is "
        "mostly changed, and 10 means everything except the requested edit is almost perfectly preserved."
    )

    def __init__(
        self, model_id="Qwen/Qwen3-VL-8B-Instruct",
        max_new_tokens=128, repetition_penalty=1.0, enable_thinking=False,
    ):
        super().__init__()

        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.enable_thinking = enable_thinking

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype="auto",
            device_map={"": 0},
            attn_implementation="eager",
        )

        self.processor = AutoProcessor.from_pretrained(model_id)
        # Left padding is required so `out_ids[len(in_ids):]` trims the same
        # (padded) prompt length off every row when batching generation.
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        tokenizer.padding_side = "left"

    def _run_batch(self, items):
        """items: list of (src_image, tgt_image, prompt) triples."""
        messages = [
            [
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
            for src_image, tgt_image, prompt in items
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, repetition_penalty=self.repetition_penalty, do_sample=False,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_texts

    def get_score(self, instruction, src_image, tgt_image):
        """
        Args:
            instruction (str): The requested edit, in natural language.
            src_image (PIL.Image): Original image.
            tgt_image (PIL.Image): Edited image.

        Returns:
            int score from 0-10, or "nan" if the model failed to produce one.
        """
        return self.get_score_batch([instruction], [src_image], [tgt_image])[0]

    def get_score_batch(self, instructions, src_images, tgt_images):
        """
        Batched version of get_score, for higher throughput when scoring many
        independent image pairs.

        Args:
            instructions (list[str]), src_images (list[PIL.Image]), tgt_images (list[PIL.Image])

        Returns:
            list of int scores 0-10 (or "nan" per item that failed to parse).
        """
        prompts = [self.PROMPT_TEMPLATE.format(instruction=instr) for instr in instructions]
        items = list(zip(src_images, tgt_images, prompts))

        try:
            answers = self._run_batch(items)
        except Exception as e:
            print(f"An error occurred during batched generation: {e}")
            torch.cuda.empty_cache()
            return ["nan"] * len(items)

        scores = []
        for answer in answers:
            # try "10" before a lone digit so it isn't parsed as "1"
            match = re.search(r"\b(?:10|[0-9])\b", answer)
            if match is None:
                print(f"Could not parse a 0-10 score out of: {answer!r}")
                scores.append("nan")
            else:
                scores.append(int(match.group()))
        return scores
