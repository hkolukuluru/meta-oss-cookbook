"""
sft.py -- supervised fine-tuning of Muse Spark Open with TorchTitan: text, then
image and audio.

Each JSONL line is ``{"messages": [{"role": ..., "content": ...}]}``; images and
audio are message parts ``{"type": "image" | "audio", "path": ...}``. Only
tokens the assistant generates are trained on. Run a recipe from a TorchTitan
checkout, with this directory on ``PYTHONPATH``::

    MUSE_SPARK_SFT_DATA="/data/sft/*.jsonl" \\
        MODULE=sft CONFIG=muse_spark_816b_a42b_sft ./run_train.sh

See README.md for the walkthrough.
"""

import math
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

import numpy as np
import torch

from torchtitan.components.data import (
    DatasetBuildContext,
    FirstFitPackingConfig,
    IndexedJsonlSource,
    SampleProcessor,
    SingleDatasetConfig,
)
from torchtitan.components.data.multimodal.mm_collator import MultiModalCollator
from torchtitan.components.data.multimodal.mm_datasets import MMSamplePackingConfig
from torchtitan.components.data.multimodal.utils.audio import (
    load_pcm_waveform,
    resample_waveform,
)
from torchtitan.components.data.multimodal.utils.image import process_image
from torchtitan.components.data.text_datasets import ChatProcessor
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.optim import Optim
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.muse_spark import (
    build_model_config,
    muse_spark_audio_encoder_config,
)
from torchtitan.models.muse_spark.renderer import MuseSparkRendererConfig
from torchtitan.trainer import Trainer
from torchtitan_recipes.models.muse_spark import (
    _muse_spark_816b_a42b,
    _resize_to_muse_glimmer_patch_grid,
    _use_released_tokenizer_with_debug_model,
    muse_spark_debugmodel,
)


# -- Text: TorchTitan's ChatProcessor with the Muse Spark renderer (_TEXT below) --


def chat_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return row["messages"]


# -- Image and audio -----------------------------------------------------------

# transformers caches compiled chat templates process-wide, and the extension
# behind `generation` blocks keeps per-render state, so concurrent renders on
# Grain's reader threads crash. Per-thread tokenizers do not help.
_RENDER_LOCK = threading.Lock()


class MediaChatSFTProcessor(SampleProcessor):
    """Chat SFT where messages may carry images and audio as ``{"type", "path"}``.

    Muse Spark's renderer is text-only, so this renders with the checkpoint's
    chat template, whose ``generation`` blocks mark assistant tokens. It trains
    on the same tokens as ``ChatProcessor``: what the assistant generates, not
    the ``<|start|>assistant`` opener the prompt supplies at inference.

    The template renders each media part as one placeholder. This expands it to
    as many tokens as the encoder emits, between start and end markers, never
    trained on, and returns the media in placeholder order.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        hf_assets_path: str
        """Directory holding tokenizer.json and chat_template.jinja."""

        template_kwargs: dict[str, str] = field(default_factory=dict)
        """Chat-template variables; pin clock-dependent ones."""

        patch_size: int
        merge_size: int
        """An image token covers a (patch_size * merge_size)-pixel square."""

        samples_per_token: int
        sample_rate: int = 24_000
        """An audio token covers samples_per_token samples at sample_rate."""

        resize_fn: Callable[..., tuple[int, int, int, int]]
        """Picks the resized image size; must match the model's image processor."""

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        from transformers import AutoTokenizer

        self._config = config
        self._hf = AutoTokenizer.from_pretrained(config.hf_assets_path)
        self._max_len = context.max_context_length
        tok = context.tokenizer
        # media type -> (placeholder id, start id, end id)
        self._ids = {
            "image": (tok.image_id, tok.vision_start_id, tok.vision_end_id),
            "audio": (tok.audio_id, tok.audio_start_id, tok.audio_end_id),
        }
        self._opener = self._hf.convert_tokens_to_ids(["<|start|>", "assistant"])

    def _load(self, part: dict[str, Any]) -> tuple[torch.Tensor | None, int]:
        """Load one media part; return it and how many tokens the encoder emits."""
        c = self._config
        if part["type"] == "image":
            image = process_image(
                part["path"],
                patch_size=c.patch_size,
                merge_size=c.merge_size,
                resize_fn=c.resize_fn,
            )
            if image is None:
                return None, 0
            side = c.patch_size * c.merge_size
            return image, (image.shape[1] // side) * (image.shape[2] // side)
        waveform, rate = load_pcm_waveform(part["path"], expected_num_channels=1)
        waveform = resample_waveform(
            waveform, source_sample_rate=rate, target_sample_rate=c.sample_rate
        )
        return waveform, math.ceil(len(waveform) / c.samples_per_token)

    def __call__(
        self, sample: dict[str, Any], rng: np.random.Generator
    ) -> dict[str, Any] | None:
        del rng
        messages = sample["messages"]
        media = {kind: [] for kind in self._ids}
        for m in messages:
            for part in m["content"] if isinstance(m["content"], list) else ():
                if part.get("type") in media:
                    media[part["type"]].append(self._load(part))
        if any(image is None for image, _ in media["image"]):
            return None

        with _RENDER_LOCK:
            out = self._hf.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
                **self._config.template_kwargs,
            )
        ids, mask = out["input_ids"], list(out["assistant_masks"])
        for i in range(len(ids) - 1):
            if [ids[i], ids[i + 1]] == self._opener:
                mask[i] = mask[i + 1] = 0

        # placeholder id -> (start id, end id, token count per media item)
        pending = {
            p: (s, e, iter(n for _, n in media[kind]))
            for kind, (p, s, e) in self._ids.items()
        }
        new_ids, new_mask = [], []
        for tok, m in zip(ids, mask):
            if tok not in pending:
                new_ids.append(tok)
                new_mask.append(m)
                continue
            start, end, counts = pending[tok]
            n = next(counts, None)
            if n is None:
                raise ValueError(
                    "the template rendered more media placeholders than there are "
                    "loadable parts; write media as {'type': 'image' | 'audio', "
                    "'path': ...}"
                )
            new_ids += [start] + [tok] * n + [end]
            new_mask += [0] * (n + 2)

        # Drop rather than truncate: a clipped answer teaches stopping mid-sentence.
        if len(new_ids) - 1 > self._max_len or not any(new_mask):
            return None
        tokens = torch.tensor(new_ids)
        # labels[i] targets token i + 1, so it trains when token i + 1 does.
        labels = torch.where(
            torch.tensor(new_mask[1:], dtype=torch.bool), tokens[1:], IGNORE_INDEX
        )
        return {
            "input_ids": tokens[:-1],
            "labels": labels,
            "positions": torch.arange(len(new_ids) - 1),
            "pixel_values": [image for image, _ in media["image"]],
            "waveforms": [waveform for waveform, _ in media["audio"]],
        }


# -- Recipes -------------------------------------------------------------------


def sft_recipe(
    base: Trainer.Config,
    *,
    data: tuple[str, ...],
    processor: SampleProcessor.Config,
    packing: Callable = FirstFitPackingConfig,
    lr: float | None = None,
    shuffle: bool = True,
) -> Trainer.Config:
    """Turn a pretraining recipe into an SFT recipe.

    Replaces the dataset and optionally the learning rate; the model, tokenizer,
    collator, parallelism, loss, and optimizer groups carry over. A modality
    swaps the ``processor`` (conversation -> tokens, labels, media) and the
    ``packing`` (whole conversations per row, media kept with them).

    Args:
        base: The pretraining recipe to derive from.
        data: JSONL globs, one ``{"messages": [...]}`` per line.
        processor: Sample processor config.
        packing: Packing config class. First-fit, not concat-then-split, which
            would cut a response in half.
        lr: Learning rate for the catch-all ``.*`` optimizer group; other groups
            keep their ratio to it. Default keeps the base recipe's rates.
        shuffle: Shuffle before data-parallel sharding.
    """
    dataset = packing(
        dataset=SingleDatasetConfig(
            source=IndexedJsonlSource.Config(patterns=data),
            processor=processor,
            post_filters=(lambda sample: sample is not None,),
        ),
    )
    return replace(
        base,
        dataloader=replace(base.dataloader, dataset=dataset, shuffle=shuffle),
        optim=base.optim if lr is None else _with_lr(base.optim, lr),
    )


def _with_lr(optim: Optim.Config, lr: float) -> Optim.Config:
    """Set the catch-all group's LR, scaling every other group by the same factor.

    Raises:
        ValueError: If no optimizer group has the catch-all pattern ``.*``.
    """
    groups = optim.optimizer.optimizers
    base_lr = next((g.lr for g in groups if g.pattern == ".*"), None)
    if base_lr is None:
        raise ValueError("lr= needs an optimizer group with pattern '.*'")
    scale = lr / base_lr
    return replace(
        optim,
        optimizer=replace(
            optim.optimizer, optimizers=[replace(g, lr=g.lr * scale) for g in groups]
        ),
    )


def env_paths(name: str, default: str | None = None) -> tuple[str, ...]:
    """Read comma-separated paths from an environment variable.

    The launcher calls recipes with no arguments, so the environment names data.

    Raises:
        ValueError: If the variable is unset and there is no default.
    """
    paths = tuple(p.strip() for p in os.environ.get(name, default or "").split(","))
    paths = tuple(p for p in paths if p)
    if not paths:
        raise ValueError(f'Set {name} to comma-separated JSONL globs, e.g. "/data/*.jsonl"')
    return paths


_HF_ASSETS = os.environ.get(
    "MUSE_SPARK_HF_ASSETS_PATH", "./assets/hf/Muse-Spark-Open-816B-A42B"
)
_SAMPLE_DATA = os.path.join(os.path.dirname(__file__), "sample.jsonl")
# Pinned: the date defaults to today, which shifts tokens daily. The effort is
# one the renderer and the released chat template agree on.
_TEMPLATE = {"current_date": "2026-01-15", "reasoning_effort": "xhigh"}
_TEXT = ChatProcessor.Config(
    messages_fn=chat_messages, renderer=MuseSparkRendererConfig(**_TEMPLATE)
)


def _media(config: Trainer.Config, *, max_image_tokens: int) -> dict[str, Any]:
    """Wire image and audio into ``config``; return the processor and packing.

    Token geometry is read from the model's encoders, so it cannot drift from
    them. Ranks whose batch lacks a modality run that encoder on a dummy input,
    so FSDP collectives stay matched across ranks.
    """
    model = config.model
    vision, audio = model.vision_encoder, model.audio_encoder
    model.dataset_modalities = ("vision", "audio")
    config.dataloader = replace(
        config.dataloader,
        collator=MultiModalCollator.Config(
            patch_size=vision.patch_size,
            temporal_patch_size=vision.patch_temporal,
            spatial_merge_size=vision.downsample_factor,
            patch_order="raster",
            waveform_pad_multiple=audio.patch_size,
            audio_input_channels=audio.input_channels,
        ),
    )
    return {
        "processor": MediaChatSFTProcessor.Config(
            hf_assets_path=_HF_ASSETS,
            template_kwargs=_TEMPLATE,
            patch_size=vision.patch_size,
            merge_size=vision.downsample_factor,
            samples_per_token=audio.patch_size,
            resize_fn=partial(
                _resize_to_muse_glimmer_patch_grid, max_image_tokens=max_image_tokens
            ),
        ),
        "packing": MMSamplePackingConfig,
    }


def _from_release(config: Trainer.Config, folder: str) -> Trainer.Config:
    """Start from the released weights in ``$MUSE_SPARK_HF_ASSETS_PATH``."""
    config.hf_assets_path = _HF_ASSETS
    config.checkpointer = replace(
        config.checkpointer,
        folder=folder,
        initial_load_path=os.path.abspath(_HF_ASSETS),
    )
    return config


def muse_spark_816b_a42b_sft() -> Trainer.Config:
    """Full-parameter text SFT from the released Muse Spark Open weights."""
    config = sft_recipe(
        _muse_spark_816b_a42b(),
        data=env_paths("MUSE_SPARK_SFT_DATA"),
        processor=_TEXT,
        lr=1e-5,  # below the 5e-5 pretraining rate; SFT overfits faster
    )
    return _from_release(config, "muse_spark_816b_a42b_sft")


def muse_spark_816b_a42b_media_sft() -> Trainer.Config:
    """Image and audio SFT; rows may mix text-only, image, and audio chats."""
    config = _muse_spark_816b_a42b()
    config.model = build_model_config(
        "816B-A42B_mm",
        parallelism=config.parallelism,
        moe_comm_backend="standard",
        attn_backend="flex",
    )
    config = sft_recipe(
        config,
        data=env_paths("MUSE_SPARK_SFT_DATA"),
        **_media(config, max_image_tokens=4096),
        lr=1e-5,
    )
    return _from_release(config, "muse_spark_816b_a42b_media_sft")


def muse_spark_debugmodel_sft() -> Trainer.Config:
    """One-GPU text smoke test: random-weight debug model, released tokenizer."""
    config = sft_recipe(
        muse_spark_debugmodel(),
        data=env_paths("MUSE_SPARK_SFT_DATA", default=_SAMPLE_DATA),
        processor=_TEXT,
        shuffle=False,
    )
    return _one_gpu(config, "debugmodel")


def muse_spark_debugmodel_media_sft() -> Trainer.Config:
    """One-GPU image and audio smoke test on ``$MUSE_SPARK_SFT_DATA``.

    The debug audio encoder takes the real 1920-samples-per-token geometry, and
    images are capped at 64 tokens to fit the 512-token context.
    """
    config = _one_gpu(muse_spark_debugmodel(), "debugmodel_mm")
    model = config.model
    model.audio_encoder = muse_spark_audio_encoder_config(
        output_size=model.audio_encoder.output_size, hidden_sizes=(16, 16, 32, 32)
    )
    return sft_recipe(
        config,
        data=env_paths("MUSE_SPARK_SFT_DATA"),
        **_media(config, max_image_tokens=64),
        shuffle=False,
    )


def _one_gpu(config: Trainer.Config, flavor: str) -> Trainer.Config:
    # Smoke tests must not checkpoint: the next run would resume and train nothing.
    config.checkpointer = None
    # The model's sharding is fixed at build time, so rebuild it for one device.
    config.parallelism = ParallelismConfig(enable_sequence_parallel=False)
    config.model = build_model_config(flavor, parallelism=config.parallelism)
    _use_released_tokenizer_with_debug_model(config)
    config.hf_assets_path = _HF_ASSETS
    return config
