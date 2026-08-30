"""
VLM Reasoning Agent: run Qwen2-VL on selected frames with a fixed prompt.
Returns a binary event label (violent/dangerous vs not) from model output.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    from PIL import Image
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError:
        Qwen2_5_VLForConditionalGeneration = None
    try:
        # Generic image-text-to-text auto class covers Qwen3-VL, InternVL, LLaVA.
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None
    try:
        from qwen_vl_utils import process_vision_info
        QWEN_VL_UTILS = True
    except ImportError:
        process_vision_info = None
        QWEN_VL_UTILS = False
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    Qwen2VLForConditionalGeneration = None
    Qwen2_5_VLForConditionalGeneration = None
    AutoModelForImageTextToText = None
    AutoProcessor = None
    Image = None
    process_vision_info = None
    QWEN_VL_UTILS = False


def _pil_from_rgb(arr: np.ndarray):
    if Image is None:
        raise ImportError("PIL is required")
    return Image.fromarray(arr.astype(np.uint8))


def _parse_yes_no(text: str) -> bool:
    """
    Heuristic: Yes -> True (violent/anomalous), No -> False.
    Prefer explicit YES; allow anomaly keywords to override lone NO for better recall.
    """
    t = text.strip().upper()
    # Explicit Yes (and no strong No) -> positive
    if "YES" in t:
        # "No" only wins if it's the direct answer, e.g. "No." or "No, ..." at start
        if t.startswith("NO") or t.startswith("NO,") or t.startswith("NO."):
            pass  # check keywords below
        else:
            return True
    # Anomaly/violence keywords override a lone No (e.g. "No clear violence but accident visible")
    anomaly_keywords = (
        "VIOLENT", "DANGER", "ABNORMAL", "ANOMALY", "FIGHT", "FIGHTING",
        "ROBBERY", "ABUSE", "ACCIDENT", "BURGLARY", "VANDALISM", "CRIME",
        "CRIMINAL", "EXPLOSION", "SHOOTING", "ASSAULT", "CRASH",
        # Additional UCF-Crime class names for multi-choice prompts
        "ARREST", "ARSON", "SHOPLIFTING", "STEALING", "THEFT", "ROADACCIDENT",
    )
    for kw in anomaly_keywords:
        if kw in t:
            return True
    # Explicit No (or unclear) -> negative
    if "NO" in t:
        return False
    return False


class VLMReasoningAgent:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 64,
    ):
        if not VLM_AVAILABLE:
            raise ImportError("transformers and PIL required; pip install transformers Pillow")
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None

    def _is_cuda_kernel_error(self, e: BaseException) -> bool:
        msg = str(e).lower()
        return "no kernel image" in msg or "is not compatible" in msg

    def _family(self) -> str:
        """Coarse model-family tag driving class + vision handling."""
        mid = self.model_id.lower()
        if "internvl" in mid:
            return "internvl"
        if "llava-next-video" in mid or "llava_next_video" in mid:
            return "llava_video"
        if "llava" in mid or "onevision" in mid:
            return "llava"
        if "qwen3" in mid:
            return "qwen3"
        if "2.5" in mid or "qwen2_5" in mid or "qwen2.5" in mid:
            return "qwen2.5"
        return "qwen2"

    def _uses_qwen_vision(self) -> bool:
        return self._family() in ("qwen2", "qwen2.5", "qwen3") and QWEN_VL_UTILS

    def _model_cls(self):
        """Pick the model class from the model family."""
        fam = self._family()
        if fam == "qwen2.5":
            if Qwen2_5_VLForConditionalGeneration is None:
                raise ImportError(
                    "Qwen2.5-VL requested but transformers lacks "
                    "Qwen2_5_VLForConditionalGeneration; upgrade transformers."
                )
            return Qwen2_5_VLForConditionalGeneration
        if fam == "qwen2":
            return Qwen2VLForConditionalGeneration
        # qwen3 / internvl / llava(-video/-onevision): generic auto class.
        if AutoModelForImageTextToText is None:
            raise ImportError(
                f"model '{self.model_id}' needs AutoModelForImageTextToText; "
                "upgrade transformers."
            )
        return AutoModelForImageTextToText

    def load(self):
        if self.model is not None:
            return
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True, use_fast=True
            )
            self.model = self._model_cls().from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_available()) else torch.float32,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.model.eval()
        except RuntimeError as e:
            if self.device == "cuda" and self._is_cuda_kernel_error(e):
                raise RuntimeError(
                    "GPU required but PyTorch has no kernel for this device (e.g. GB10/sm_121). "
                    "Install PyTorch nightly cu130 or run with --device cpu."
                ) from e
            raise

    def run(
        self,
        frames: list[np.ndarray],
        prompt: str,
    ) -> tuple[str, bool]:
        """
        frames: list of RGB (H,W,3). prompt: question text.
        Returns (raw_output_text, parsed_yes_no).
        """
        self.load()
        if not frames:
            return "No frames.", False
        pil_images = [_pil_from_rgb(f) for f in frames]
        content = []
        for img in pil_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self._uses_qwen_vision() and process_vision_info is not None:
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            # LLaVA / InternVL processors take a flat list of PIL images.
            inputs = self.processor(
                text=[text],
                images=pil_images,
                padding=True,
                return_tensors="pt",
            )
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
        except RuntimeError as e:
            if self.device == "cuda" and self._is_cuda_kernel_error(e):
                raise RuntimeError(
                    "GPU required but PyTorch has no kernel for this device (e.g. GB10/sm_121). "
                    "Install PyTorch nightly cu130 or run with --device cpu."
                ) from e
            raise
        input_len = inputs["input_ids"].shape[1]
        output_ids = out[0][input_len:]
        answer = self.processor.decode(output_ids, skip_special_tokens=True).strip()
        return answer, _parse_yes_no(answer)


def run_vlm_on_retrieved_frames(
    frames: list[tuple[int, np.ndarray]],
    prompt: str,
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    device: str = "cuda",
) -> tuple[str, bool]:
    """
    Convenience: take list of (index, RGB), run VLM, return (answer, is_violent).
    """
    agent = VLMReasoningAgent(model_id=model_id, device=device)
    rgb_list = [f[1] for f in frames]
    return agent.run(rgb_list, prompt)
