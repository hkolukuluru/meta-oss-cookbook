# Post-training

Fine-tune Muse Spark Open with an open-source training framework on your own GPUs, or with a hosted service that runs the job for you. Each option has its own section below.

| Option | Type | Methods |
| --- | --- | --- |
| [Miles](rl/miles/README.md) | Open-source framework | Reinforcement learning (RL) with Group Relative Policy Optimization (GRPO) on 64 GPUs, with SGLang rollouts |
| [NVIDIA NeMo AutoModel](#nvidia-nemo-automodel) | Open-source framework | Full-parameter supervised fine-tuning (SFT) and low-rank adaptation (LoRA) |
| [NVIDIA Megatron Bridge](#nvidia-megatron-bridge) | Open-source framework | Megatron-based training, with Hugging Face checkpoint conversion |
| [TRL](sft/trl/README.md) | Open-source framework | SFT with LoRA |
| [TorchTitan](sft/torchtitan/README.md) | Open-source framework | Full-parameter SFT on text, image, and audio conversations |
| [Together AI](together-ai.md) | Hosted service | SFT and RL (GRPO) |
| [Fireworks AI](fireworks-ai.md) | Hosted service | Serverless LoRA: SFT, Direct Preference Optimization (DPO), Odds Ratio Preference Optimization (ORPO), distillation, RL; full-parameter on dedicated GPUs |

## NVIDIA NeMo AutoModel

[NVIDIA NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) is a post-training library built on PyTorch DTensor that loads Hugging Face checkpoints directly, with no conversion. See the [Muse Spark Open guide in NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/model-coverage/vlm/muse/muse_spark_open.mdx) for setup and training recipes.

## NVIDIA Megatron Bridge

[NVIDIA Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) is a training library for Megatron-based models that converts checkpoints to and from Hugging Face format. See the [Muse Spark Open example in Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/muse_spark_open) for setup and training recipes.

## Next steps

Serve the tuned model with a runtime from [Run inference](../inference-server/README.md), using the [recommended settings](../inference-server/settings.md) and the chat format in the [prompting guide](../prompting/README.md).
