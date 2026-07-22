import json
import re
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class QwenImagePreservationJudge(torch.nn.Module):
    """
    Three-factor image-preservation judge -- Unchanged Regions, Global
    Consistency, Identity Preservation -- ported from the "Image
    Preservation" category prompt in Liu et al., "Human-Aligned MLLM Judges
    for Fine-Grained Image Editing Evaluation" (arXiv:2602.13028),
    prompts/v2_image_preservation_online.txt, adapted from GPT/Gemini to
    Qwen. Mirrors their own judge_evaluation.py structure: the prompt lives
    in its own text file (prompts/image_preservation_online.txt) with literal
    "[input image]" / "[edited image]" / "[text instruction]" placeholders,
    which get replaced -- not used as text/image splice points -- exactly
    like their call_azure_openai_with_images does.
    """

    FACTORS = ["unchanged_regions", "global_consistency", "identity_preservation"]

    PROMPT_PATH = PROMPTS_DIR / "image_preservation_online.txt"

    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct"):
        super().__init__()

        self.prompt_template = self.PROMPT_PATH.read_text(encoding="utf-8")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        tokenizer.padding_side = "left"

    def _build_prompt(self, instruction):
        # Same three replacements as the paper's call_azure_openai_with_images:
        # substitute the instruction, then delete both image placeholders
        # (the images themselves are attached as separate content blocks,
        # appended after this text -- not spliced in at these positions).
        complete_prompt = self.prompt_template.replace("[text instruction]", instruction)
        complete_prompt = complete_prompt.replace("[input image]", "")
        complete_prompt = complete_prompt.replace("[edited image]", "")
        return complete_prompt

    def _run_batch(self, items):
        """items: list of (instruction, src_image, tgt_image) triples."""
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._build_prompt(instruction)},
                        {"type": "text", "text": "Input Image:"},
                        {"type": "image", "image": src_image},
                        {"type": "text", "text": "Edited Image:"},
                        {"type": "image", "image": tgt_image},
                    ],
                }
            ]
            for instruction, src_image, tgt_image in items
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=400, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_texts

    @staticmethod
    def _extract_json_block(text):
        """Same fallback strategies as the paper's extract_json_from_response
        (json fence, bare fence, brace span, raw text), each retried with
        newlines stripped in case a value has an unescaped line break. Unlike
        theirs, every strategy is tried rather than stopping at the first
        brace match."""
        candidates = []

        for pattern in (r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                candidates.append(match.group(1))

        start_idx = text.find("{")
        end_idx = text.rfind("}") + 1
        if start_idx != -1 and end_idx != 0:
            candidates.append(text[start_idx:end_idx])

        candidates.append(text)

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                pass
            # same newline-stripping recovery the paper's parser does, applied
            # to every candidate rather than just the raw brace span
            cleaned = candidate.replace("\n", " ").replace("\r", " ")
            try:
                return json.loads(cleaned)
            except (json.JSONDecodeError, ValueError):
                continue

        return None

    def _nan_result(self):
        return {factor: {"score": "nan", "justification": ""} for factor in self.FACTORS}

    @classmethod
    def combined_score(cls, result):
        """
        Unweighted mean of the three factor scores, matching Table 2's
        "Image Preservation" category formula in the paper (verified against
        Table 1's per-factor numbers). Returns "nan" if any factor failed to
        parse, rather than silently averaging over a gap.
        """
        scores = [result[factor]["score"] for factor in cls.FACTORS]
        if any(score == "nan" for score in scores):
            return "nan"
        return sum(scores) / len(scores)

    def judge(self, instruction, src_image, tgt_image):
        """
        Args:
            instruction (str): The requested edit, in natural language.
            src_image (PIL.Image): Original image.
            tgt_image (PIL.Image): Edited image.

        Returns:
            dict mapping each factor in FACTORS to {"score": int 1-7 (or
            "nan"), "justification": str}.
        """
        return self.judge_batch([instruction], [src_image], [tgt_image])[0]

    def judge_batch(self, instructions, src_images, tgt_images):
        """
        Batched version of judge, for higher throughput when scoring many
        independent image pairs.

        Args:
            instructions (list[str]), src_images (list[PIL.Image]), tgt_images (list[PIL.Image])

        Returns:
            list of dicts, one per item, each mapping FACTORS to
            {"score": int 1-7 (or "nan"), "justification": str}.
        """
        items = list(zip(instructions, src_images, tgt_images))

        try:
            answers = self._run_batch(items)
        except Exception as e:
            print(f"An error occurred during batched generation: {e}")
            return [self._nan_result() for _ in items]

        results = []
        for answer in answers:
            parsed = self._extract_json_block(answer)
            factor_results = parsed.get("online_factor_results") if parsed else None
            if not factor_results:
                print(f"Could not parse factor JSON out of: {answer!r}")
                results.append(self._nan_result())
                continue

            result = {}
            for factor in self.FACTORS:
                entry = factor_results.get(factor) or {}
                try:
                    score = int(entry.get("score"))
                except (TypeError, ValueError):
                    score = "nan"
                result[factor] = {
                    "score": score,
                    "justification": entry.get("justification", ""),
                }
            results.append(result)
        return results
