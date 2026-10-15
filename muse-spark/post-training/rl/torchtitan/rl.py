"""
rl.py -- reinforcement learning for Muse Spark Open with TitanRL, TorchTitan's
RL stack, with pluggable tasks.

The Muse Spark Open system -- models, parallelism, Dist-MoE, loss, trainer,
generator -- comes unchanged from TorchTitan's DAPO-Math recipe. ``with_task``
runs any task on it; a task is a dataset, an environment, and a reward (see
qa_task.py). Run a recipe from a TorchTitan checkout, with this directory on
``PYTHONPATH``::

    python -m torchtitan.rl.train --module rl --config muse_spark_dapo_math

See README.md for the walkthrough.
"""

from __future__ import annotations

import os
from dataclasses import replace

from torchtitan.rl.controller.controller import Controller
from torchtitan.rl.examples.dapo_math.data import AIME2025Dataset, DapoMathDataset
from torchtitan.rl.rollout.rollouter import Rollouter
from torchtitan_recipes.rl.dapo_math import muse_spark_816b_a42b_dapo_math_gb300_config

from qa_task import qa_task

_HF_ASSETS = os.environ.get(
    "MUSE_SPARK_HF_ASSETS_PATH", "./assets/hf/Muse-Spark-Open-816B-A42B"
)
_QA_SAMPLE = os.path.join(os.path.dirname(__file__), "qa_sample.jsonl")


def with_task(
    config: Controller.Config, task: Rollouter.Config, *, dump_folder: str
) -> Controller.Config:
    """Run ``task`` on the Muse Spark Open system in ``config``.

    Everything but the task is kept. The task's rollout budget is set to the
    trainer's context length, so rollouts are neither cut off nor shorter than
    what the trainer packs.
    """
    token_env = replace(
        task.worker.token_env,
        max_rollout_tokens=config.trainer.training.max_context_length,
    )
    task = replace(task, worker=replace(task.worker, token_env=token_env))
    return replace(config, rollouter=task, dump_folder=dump_folder)


def muse_spark_dapo_math() -> Controller.Config:
    """TorchTitan's Muse Spark Open DAPO-Math recipe, unchanged.

    Datasets default to the Hugging Face IDs; set ``MUSE_SPARK_DAPO_MATH_REPO_ID``
    and ``MUSE_SPARK_AIME2025_REPO_ID`` to local copies on clusters without
    network access.
    """
    return muse_spark_816b_a42b_dapo_math_gb300_config(
        hf_assets_path=_HF_ASSETS,
        dapo_math_repo_id=os.environ.get(
            "MUSE_SPARK_DAPO_MATH_REPO_ID", DapoMathDataset.Config().repo_id
        ),
        aime2025_repo_id=os.environ.get(
            "MUSE_SPARK_AIME2025_REPO_ID", AIME2025Dataset.Config().repo_id
        ),
    )


def muse_spark_qa() -> Controller.Config:
    """The template task in qa_task.py, on the same system.

    Reads ``QA_TRAIN_DATA`` and ``QA_VALIDATION_DATA`` (JSONL); both default to
    the bundled qa_sample.jsonl.
    """
    return with_task(
        muse_spark_dapo_math(),
        qa_task(
            train_path=os.environ.get("QA_TRAIN_DATA", _QA_SAMPLE),
            validation_path=os.environ.get("QA_VALIDATION_DATA", _QA_SAMPLE),
        ),
        dump_folder="outputs/rl/muse_spark_qa",
    )
