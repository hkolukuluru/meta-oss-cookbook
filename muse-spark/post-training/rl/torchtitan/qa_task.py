"""
qa_task.py -- a template RL task for Muse Spark Open: exact-answer questions
from your own JSONL. Copy this file to start a new task.

A task is three pieces, each a subclass of a TorchTitan base class:

    dataset  QADataset         yields one sample per prompt        (RLDataset)
    env      QAEnv             turns a sample into a conversation  (MessageEnv)
    reward   RewardExactMatch  scores a finished rollout           (RewardFn)

``qa_task`` wires them into a ``Rollouter.Config``; ``rl.with_task`` runs it on
the Muse Spark Open system. Each JSONL row is ``{"question": ..., "answer": ...}``.
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset
from renderers import Message

from torchtitan.rl.examples.dapo_math.data import _CyclingDataset
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.data import IterableRLDataLoader, RLDataset
from torchtitan.rl.rollout.message_env import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)
from torchtitan.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.rl.rollout.rubric import RewardFn, Rubric
from torchtitan.rl.rollout.token_env import TokenEnv
from torchtitan.rl.rollout.types import Rollout


@dataclass(frozen=True, kw_only=True, slots=True)
class QASample:
    """What the dataset yields, and what the env and reward receive."""

    question: str
    answer: str


class QADataset(_CyclingDataset):
    """Endless, resumable stream over a JSONL file.

    Shuffling and checkpoint resume come from TorchTitan's DAPO-Math cycling
    dataset, which is private upstream; update the import if it moves.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RLDataset.Config):
        path: str
        seed: int = 42
        shuffle: bool = True

    def __init__(self, config: Config) -> None:
        rows = load_dataset("json", data_files=config.path, split="train")
        samples = [QASample(question=r["question"], answer=str(r["answer"])) for r in rows]
        super().__init__(samples, seed=config.seed, shuffle=config.shuffle)


class QAEnv(MessageEnv):
    """Single turn: ask the question, end after one reply.

    For tools, return them from ``init`` and answer tool calls in ``step``; for
    more turns, return ``done=False`` with the env's reply.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        instruction: str = "Reply with only the final answer."

    def __init__(self, config: Config, *, env_input: QASample) -> None:
        self._prompt = f"{env_input.question}\n\n{config.instruction}"

    async def init(self) -> MessageEnvInitOutput:
        return MessageEnvInitOutput(
            init_prompt_messages=[{"role": "user", "content": self._prompt}]
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        return MessageEnvStepOutput(done=True)


class RewardExactMatch(RewardFn):
    """1.0 when the reply equals the answer, ignoring case and surrounding space."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: QASample) -> float:
        message = rollout.turns[-1].completion_message if rollout.turns else None
        reply = (message or {}).get("content") or ""
        return float(reply.strip().lower() == env_input.answer.strip().lower())


def qa_task(*, train_path: str, validation_path: str) -> Rollouter.Config:
    """Wire the three pieces; the rest is TorchTitan's default rollout path."""
    return Rollouter.Config(
        training_dataloader=IterableRLDataLoader.Config(
            dataset=QADataset.Config(path=train_path)
        ),
        validation_dataset=QADataset.Config(path=validation_path, shuffle=False),
        worker=RolloutWorker.Config(
            message_env=QAEnv.Config(),
            rubric=Rubric.Config(
                reward_fns=[RewardExactMatch.Config(weight=1.0)], error_reward=0.0
            ),
            token_env=TokenEnv.Config(max_num_turns=1),
            advantage=AdvantageEstimator.Config(should_std_normalize=False),
        ),
    )
