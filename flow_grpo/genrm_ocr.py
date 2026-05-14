"""OCR scoring backed by a generative reward model (GRM) served via vLLM.

Mirrors ``verl_omni.utils.reward_score.genrm_ocr.compute_score_ocr`` so that
the flow_grpo training loop can be benchmarked against verl-omni using the
same async HTTP reward path. The GRM is expected to be served behind an
OpenAI-compatible endpoint (e.g. ``vllm.entrypoints.openai.api_server``).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
from typing import List, Optional, Sequence, Union

import aiohttp
import numpy as np
import torch
from PIL import Image

DEFAULT_GRM_PROMPT = (
    "Please output only the text content from the image without any additional descriptions or formatting."
)
DEFAULT_SAMPLING_PARAMS = {"temperature": 0.7, "top_p": 0.8, "max_tokens": 4096}


def _to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image
    if isinstance(image, torch.Tensor):
        image = image.float().cpu().numpy()
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).round().clip(0, 255).astype(np.uint8)
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = image.transpose(1, 2, 0)
        return Image.fromarray(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def _pil_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_ground_truth(prompt: str) -> str:
    """Extract the OCR ground truth string from a flow_grpo OCR prompt.

    Prompts look like: ``... reads "Take With Food" prominently ...``
    """
    parts = prompt.split('"')
    if len(parts) >= 3:
        return parts[1]
    return prompt


def _levenshtein_score(text: str, ground_truth: str) -> float:
    try:
        import Levenshtein

        dist_fn = Levenshtein.distance
    except ImportError:  # pragma: no cover - fallback for environments without Levenshtein
        from difflib import SequenceMatcher

        def dist_fn(a, b):
            # crude approximation good enough as a fallback
            ratio = SequenceMatcher(None, a, b).ratio()
            return int(round(max(len(a), len(b)) * (1.0 - ratio)))

    gt = re.sub(r"\s+", "", ground_truth).lower()
    txt = re.sub(r"\s+", "", text or "").lower()
    if not gt:
        return 1.0 if not txt else 0.0
    if gt in txt:
        dist = 0
    else:
        dist = dist_fn(txt, gt)
    dist = min(dist, len(gt))
    return 1.0 - dist / len(gt)


class GenRMOcrScorer:
    """Async batched OCR scorer that POSTs images to a vLLM OpenAI server.

    Parameters
    ----------
    router_address: ``host:port`` of the GRM router (no scheme).
    model_name: served-model-name to pass to the chat completion API.
    api_key: optional bearer token (matches ``--api-key`` of the server).
    request_timeout: per-request timeout in seconds (None = no limit).
    max_concurrency: cap concurrent in-flight requests to avoid overloading
        the single-card server.
    """

    def __init__(
        self,
        router_address: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout: Optional[float] = None,
        max_concurrency: int = 64,
    ):
        self.router_address = router_address or os.environ.get(
            "REWARD_ROUTER_ADDRESS", "127.0.0.1:17140"
        )
        self.model_name = model_name or os.environ.get(
            "REWARD_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct"
        )
        self.api_key = api_key if api_key is not None else os.environ.get(
            "REWARD_API_KEY", "flowgrpo"
        )
        self.request_timeout = request_timeout
        self.max_concurrency = max_concurrency

    async def _chat_complete(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        image_b64: str,
    ) -> str:
        url = f"http://{self.router_address}/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_b64}},
                        {"type": "text", "text": DEFAULT_GRM_PROMPT},
                    ],
                },
            ],
            **DEFAULT_SAMPLING_PARAMS,
        }
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with sem:
            async with session.post(url, json=payload, headers=headers) as resp:
                text = await resp.text()
        try:
            data = json.loads(text)
            return data["choices"][0]["message"]["content"]
        except Exception:
            return ""

    async def _score_async(
        self,
        images: Sequence[Image.Image],
        ground_truths: Sequence[str],
    ) -> List[float]:
        loop = asyncio.get_event_loop()
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        sem = asyncio.Semaphore(self.max_concurrency)
        # Encode images to base64 in thread pool (CPU-bound PNG encoding)
        images_b64 = await asyncio.gather(
            *[loop.run_in_executor(None, _pil_to_base64, img) for img in images]
        )
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [self._chat_complete(session, sem, img) for img in images_b64]
            responses = await asyncio.gather(*tasks)
        return [_levenshtein_score(r, gt) for r, gt in zip(responses, ground_truths)]

    def __call__(
        self,
        images: Union[List[Image.Image], List[np.ndarray], np.ndarray, torch.Tensor],
        prompts: Sequence[str],
    ) -> List[float]:
        # Normalize to list of PIL images.
        if isinstance(images, torch.Tensor):
            arr = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            if arr.ndim == 4 and arr.shape[1] == 3:
                arr = arr.transpose(0, 2, 3, 1)
            images = [Image.fromarray(a) for a in arr]
        elif isinstance(images, np.ndarray):
            images = [Image.fromarray(a) for a in images]
        else:
            images = [_to_pil(img) for img in images]

        ground_truths = [_extract_ground_truth(p) for p in prompts]

        # Run async loop synchronously inside the worker thread.
        # Image encoding to base64 happens inside the async context via run_in_executor.
        try:
            loop = asyncio.new_event_loop()
            return loop.run_until_complete(self._score_async(images, ground_truths))
        finally:
            loop.close()
