import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class QwenFullMetricsJudge(torch.nn.Module):
    """
    Full 12-factor MLLM-as-a-Judge, ported from the "Main" all-factors-at-once
    prompt in Liu et al., "Human-Aligned MLLM Judges for Fine-Grained Image
    Editing Evaluation" (arXiv:2602.13028), prompts/v3_judge_online_all_factors.txt
    -- the prompt their own judge_evaluation.py actually loads by default, and
    the one behind the paper's headline Table 1/Table 2 numbers -- adapted
    from GPT/Gemini to Qwen.

    Same architecture as QwenImagePreservationJudge: prompt lives in its own
    text file, "[input image]" / "[edited image]" / "[text instruction]"
    placeholders get replaced (not used as splice points), and an explicit
    "Input Image:" / "Edited Image:" caption precedes each image as a
    cross-model safety net beyond what the paper's own code does.
    """

    FACTORS = [
        "unchanged_regions", "global_consistency", "identity_preservation",
        "scale_realism", "spatial_relationship", "texture_and_detail",
        "image_quality", "color_and_lighting", "seamlessness",
        "alignment", "completeness", "plausibility",
    ]

    # Table 3's three higher-order categories, used for the Table-2-style
    # per-category rollups (each category's score is the unweighted mean of
    # its factors, same formula verified for combined_score).
    CATEGORIES = {
        "image_preservation": ["unchanged_regions", "global_consistency", "identity_preservation"],
        "edit_quality": [
            "scale_realism", "spatial_relationship", "texture_and_detail",
            "image_quality", "color_and_lighting", "seamlessness",
        ],
        "instruction_fidelity": ["alignment", "completeness", "plausibility"],
    }

    PROMPT_PATH = PROMPTS_DIR / "full_metrics_online.txt"

    def __init__(
        self, model_id="Qwen/Qwen3-VL-8B-Instruct",
        max_new_tokens=1200, repetition_penalty=1.0, enable_thinking=False,
    ):
        super().__init__()

        self.prompt_template = self.PROMPT_PATH.read_text(encoding="utf-8")
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.enable_thinking = enable_thinking

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        tokenizer.padding_side = "left"

    def _build_prompt(self, instruction):
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
            enable_thinking=self.enable_thinking,
        )
        inputs = inputs.to(self.model.device)

        # 12 factors x (~25-word justification + score) needs far more room
        # than the 3-factor judge's 400 -- generously over-provisioned since
        # this is a ceiling generate() stops well short of on a clean
        # response, not a fixed cost, and truncation here means losing all
        # 12 scores, not just one. max_new_tokens/repetition_penalty match
        # Qwen-Image-Bench's own documented fixed inference parameters when
        # that checkpoint is in use (see model_id in __init__).
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
            cleaned = candidate.replace("\n", " ").replace("\r", " ")
            try:
                return json.loads(cleaned)
            except (json.JSONDecodeError, ValueError):
                continue

        return None

    def _nan_result(self):
        return {factor: {"score": "nan", "justification": ""} for factor in self.FACTORS}

    @classmethod
    def overall_score(cls, result):
        """
        Unweighted mean of all 12 factor scores, matching Table 3's
        definition: "for MLLM-as-a-Judge evaluators, the overall factor score
        is calculated from 12 other factors." Returns "nan" if any factor
        failed to parse.
        """
        scores = [result[factor]["score"] for factor in cls.FACTORS]
        if any(score == "nan" for score in scores):
            return "nan"
        return sum(scores) / len(scores)

    @classmethod
    def category_scores(cls, result):
        """
        Table 2's per-category rollups: the unweighted mean of each higher-
        order category's factors (image_preservation, edit_quality,
        instruction_fidelity). Each entry is "nan" if any factor in that
        category failed to parse.
        """
        rollups = {}
        for category, factors in cls.CATEGORIES.items():
            scores = [result[factor]["score"] for factor in factors]
            rollups[category] = "nan" if any(s == "nan" for s in scores) else sum(scores) / len(scores)
        return rollups

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
