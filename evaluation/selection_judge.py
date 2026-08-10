import re
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class QwenSelectionJudge(torch.nn.Module):
    """
    Tournament-style MLLM-as-a-Judge that SELECTS the single best candidate
    edited image out of a small set, rather than scoring each (source,
    edited) pair independently the way QwenPreservationScorer,
    QwenImagePreservationJudge, and QwenFullMetricsJudge do.

    A new selection-based methodology meant to
    be compared against those per-cell scoring judges. Given the source
    image, the editing instruction, and N candidate edited images, asks
    Qwen to pick the index of the best one, balancing preservation of
    unedited content, edit quality/realism, and instruction alignment.

    Includes judge_tournament_batch, which runs a full 2-round tournament
    (one call per group of candidates, then a final call over the group
    winners) over a grid too large to show Qwen at once (e.g. the 121-cell
    (t_start, t_end) grid used elsewhere in this package), batching across
    samples within each round for throughput. The caller only decides how
    to partition each sample's candidates into round-1 groups -- this
    class owns running the tournament itself.

    Same architecture as the other judges here: prompt lives in its own
    text file, "[text instruction]" / "[n_candidates]" placeholders get
    replaced (never used as splice points), no system prompt (matching the
    paper-derived judges' own user-prompt-only convention), and an
    explicit "Source Image:" / "Candidate N:" caption precedes each image.
    """

    PROMPT_PATH = PROMPTS_DIR / "selection_judge.txt"

    def __init__(
        self, model_id="Qwen/Qwen3-VL-8B-Instruct",
        max_new_tokens=700, repetition_penalty=1.0, enable_thinking=False,
        device_map={"": 0},
    ):
        super().__init__()

        self.prompt_template = self.PROMPT_PATH.read_text(encoding="utf-8")
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.enable_thinking = enable_thinking

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype="auto",
            device_map=device_map,
            attn_implementation="eager",
        )

        self.processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        tokenizer.padding_side = "left"

    def _build_prompt(self, instruction, n_candidates):
        complete_prompt = self.prompt_template.replace("[text instruction]", instruction)
        complete_prompt = complete_prompt.replace("[n_candidates]", str(n_candidates))
        return complete_prompt

    def _run_batch(self, items):
        """items: list of (instruction, src_image, candidates) triples,
        where candidates is a list of PIL images."""
        messages = []
        for instruction, src_image, candidates in items:
            content = [{"type": "text", "text": self._build_prompt(instruction, len(candidates))}]
            content.append({"type": "text", "text": "Source Image:"})
            content.append({"type": "image", "image": src_image})
            for i, candidate in enumerate(candidates, start=1):
                content.append({"type": "text", "text": f"Candidate {i}:"})
                content.append({"type": "image", "image": candidate})
            messages.append([{"role": "user", "content": content}])

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
    def _nan_result():
        return {"best_index": "nan", "reasoning": ""}

    def select(self, instruction, src_image, candidates):
        """
        Args:
            instruction (str): The requested edit, in natural language.
            src_image (PIL.Image): Original image.
            candidates (list[PIL.Image]): Candidate edited images, indexed
                1..len(candidates) in the prompt shown to Qwen.

        Returns:
            dict with "best_index" (1-based int, or "nan" if parsing
            failed or the returned index was out of range) and
            "reasoning" (str).
        """
        return self.select_batch([instruction], [src_image], [candidates])[0]

    def select_batch(self, instructions, src_images, candidates_list):
        """
        Batched version of select. Items may have differing candidate
        counts (see _run_batch) -- useful for the tournament usage this
        class is designed for, where a round-1 group occasionally fails to
        produce a winner, leaving that sample with fewer round-2 candidates
        than the rest.

        Args:
            instructions (list[str]), src_images (list[PIL.Image]),
            candidates_list (list[list[PIL.Image]])

        Returns:
            list of dicts, one per item, each with "best_index" (1-based
            int, or "nan") and "reasoning" (str).
        """
        items = list(zip(instructions, src_images, candidates_list))

        try:
            answers = self._run_batch(items)
        except Exception as e:
            print(f"An error occurred during batched generation: {e}")
            torch.cuda.empty_cache()
            return [self._nan_result() for _ in items]

        results = []
        for answer, (_, _, candidates) in zip(answers, items):
            match = re.search(r"(?<!\d)\d{1,2}(?!\d)", answer)
            if not match:
                print(f"Could not parse a bare index out of: {answer!r}")
                results.append(self._nan_result())
                continue

            best_index = int(match.group())
            if not (1 <= best_index <= len(candidates)):
                print(f"best_index {best_index} out of range for {len(candidates)} candidates: {answer!r}")
                best_index = "nan"

            results.append({"best_index": best_index, "reasoning": ""})
        return results

    @staticmethod
    def _batched(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    def judge_tournament_batch(self, samples, batch_size=8, on_progress=None):
        """
        Run a 2-round tournament for each sample in `samples`, batching
        round-1 calls across ALL samples together, then round-2 calls
        across all samples together -- this is the batched, multi-sample
        way to run this judge over a grid too large to reliably show Qwen at
        once (e.g. the 121-cell (t_start, t_end) grid used elsewhere in this
        package). Round 2 only runs for a sample if every one of its
        round-1 groups produced a winner -- a sample missing a winner
        would otherwise compare a lopsided, unfair subset of the grid, so
        it's marked failed (best_key=None) instead.

        Args:
            samples: list of dicts, each with:
                "instruction" (str)
                "src_image" (PIL.Image)
                "groups" (list[list[tuple]]): round-1 groups, each a list
                    of (key, PIL.Image) pairs. key can be anything
                    hashable (e.g. (t_start, t_end)) -- used only to
                    identify which candidate won.
            batch_size (int): batch size for each round's underlying
                select_batch calls.
            on_progress (callable, optional): called after each underlying
                batch as on_progress(round_name, done, total), where
                round_name is "round1" or "round2" -- lets a long-running
                caller print/log progress without this method owning any
                particular logging format.

        Returns:
            list of dicts, one per sample, each with:
                "best_key": the winning candidate's key, or None if the
                    tournament failed for this sample (a group -- or the
                    final round -- failed to produce a parseable pick)
                "reasoning" (str): empty if best_key is None
                "round1_winners" (list[tuple]): (key, reasoning) for each
                    group that produced a winner (may be shorter than
                    len(sample["groups"]) if some groups failed)
        """
        key_to_image = []
        for sample in samples:
            mapping = {}
            for group in sample["groups"]:
                for key, image in group:
                    mapping[key] = image
            key_to_image.append(mapping)

        # ---- Round 1: every (sample, group) task, batched across all of them ----
        round1_tasks = []
        for sample_idx, sample in enumerate(samples):
            for group in sample["groups"]:
                keys = [key for key, _ in group]
                images = [image for _, image in group]
                round1_tasks.append((sample_idx, keys, images))

        round1_winners = [[] for _ in samples]
        round1_done = 0
        for batch in self._batched(round1_tasks, batch_size):
            instructions = [samples[sample_idx]["instruction"] for sample_idx, _, _ in batch]
            src_images = [samples[sample_idx]["src_image"] for sample_idx, _, _ in batch]
            candidate_lists = [images for _, _, images in batch]

            results = self.select_batch(instructions, src_images, candidate_lists)

            for (sample_idx, keys, _), result in zip(batch, results):
                if result["best_index"] == "nan":
                    continue
                key = keys[result["best_index"] - 1]
                round1_winners[sample_idx].append((key, result["reasoning"]))

            round1_done += len(batch)
            if on_progress:
                on_progress("round1", round1_done, len(round1_tasks))

        # ---- Round 2: one call per sample with a complete set of winners ----
        final_results = [None] * len(samples)
        round2_sample_idxs = []
        for sample_idx, sample in enumerate(samples):
            winners = round1_winners[sample_idx]
            if len(winners) != len(sample["groups"]):
                final_results[sample_idx] = {"best_key": None, "reasoning": "", "round1_winners": winners}
                continue
            round2_sample_idxs.append(sample_idx)

        round2_done = 0
        for batch_idxs in self._batched(round2_sample_idxs, batch_size):
            instructions = [samples[i]["instruction"] for i in batch_idxs]
            src_images = [samples[i]["src_image"] for i in batch_idxs]
            candidate_lists = [
                [key_to_image[i][key] for key, _ in round1_winners[i]] for i in batch_idxs
            ]

            results = self.select_batch(instructions, src_images, candidate_lists)

            for sample_idx, result in zip(batch_idxs, results):
                winners = round1_winners[sample_idx]
                if result["best_index"] == "nan":
                    final_results[sample_idx] = {"best_key": None, "reasoning": "", "round1_winners": winners}
                else:
                    key, _ = winners[result["best_index"] - 1]
                    final_results[sample_idx] = {
                        "best_key": key, "reasoning": result["reasoning"], "round1_winners": winners,
                    }

            round2_done += len(batch_idxs)
            if on_progress:
                on_progress("round2", round2_done, len(round2_sample_idxs))

        return final_results
