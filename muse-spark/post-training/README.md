# Post-training

Fine-tune Muse Spark Open with an open-source training framework on your own GPUs, or with a hosted service that runs the job for you. Each option has its own section below.

| Option | Type | Methods |
| --- | --- | --- |
| [SFT with TorchTitan](sft/torchtitan/README.md) | Open-source framework | Full-parameter SFT on text, image, and audio conversations |
| [RL with TorchTitan](rl/torchtitan/README.md) | Open-source framework | RL (GRPO with the DAPO loss) with pluggable tasks |
| [Miles](rl/miles/README.md) | Open-source framework | Reinforcement learning (RL) with Group Relative Policy Optimization (GRPO) on 64 GPUs, with SGLang rollouts |
| [NVIDIA NeMo AutoModel](#nvidia-nemo-automodel) | Open-source framework | Full-parameter supervised fine-tuning (SFT) and low-rank adaptation (LoRA) |
| [NVIDIA Megatron Bridge](#nvidia-megatron-bridge) | Open-source framework | Megatron-based training, with Hugging Face checkpoint conversion |
| [TRL](sft/trl/README.md) | Open-source framework | SFT with LoRA |
| [Together AI](together-ai.md) | Hosted service | SFT and RL (GRPO) |
| [Fireworks AI](fireworks-ai.md) | Hosted service | Serverless LoRA: SFT, Direct Preference Optimization (DPO), Odds Ratio Preference Optimization (ORPO), distillation, RL; full-parameter on dedicated GPUs |

## TorchTitan fine-tuning

[TorchTitan](https://github.com/pytorch/torchtitan) provides PyTorch-native training for Muse Spark Open, and [TitanRL](https://github.com/pytorch/torchtitan/tree/main/torchtitan/rl) is its reinforcement-learning (RL) library. The trainer and vLLM generators share TorchTitan model definitions. Monarch orchestrates training and rollouts, and TorchStore transfers updated weights to the generators.

TorchTitan ships reference recipes for the BF16 checkpoint on NVIDIA Blackwell GPUs (SM100 or newer):

- **SFT with LoRA:** `muse_spark_816b_a42b_ultradata_lora_sft_cp16_128k` trains a rank-8 LoRA adapter on agent conversations with a 131,072-token context, on 16 GB300 GPUs.
- **RL with DAPO:** `muse_spark_816b_a42b_dapo_math_gb300_config` trains on math reasoning with vLLM rollouts, on 256 trainer GPUs plus 16 generator GPUs. It needs a cluster launcher that places the trainer and the generator on separate hosts.

The cookbook has step-by-step recipes for both: [full-parameter SFT on your own text, image, and audio conversations](sft/torchtitan/README.md), and [RL on a task you plug in](rl/torchtitan/README.md), starting from the DAPO-Math recipe.

### Links

- [Muse Spark Open model guide in TorchTitan](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/muse_spark/README.md): setup, checkpoint download, and the SFT recipes
- [Muse Spark Open DAPO-Math RL configuration](https://github.com/pytorch/torchtitan/blob/main/torchtitan_recipes/rl/dapo_math.py)
- [TitanRL setup and experiment guide](https://github.com/pytorch/torchtitan/blob/main/torchtitan/rl/README.md)

## NVIDIA NeMo AutoModel

[NVIDIA NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) is a post-training library built on PyTorch DTensor that loads Hugging Face checkpoints directly, with no conversion. See the [Muse Spark Open guide in NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/model-coverage/vlm/meta-models/muse_spark_open.mdx) for setup and training recipes.

## NVIDIA Megatron Bridge

[NVIDIA Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) is a training library for Megatron-based models that converts checkpoints to and from Hugging Face format. See the [Muse Spark Open example in Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/muse_spark_open) for setup and training recipes.

## Next steps

Serve the tuned model with a runtime from [Run inference](../inference-server/README.md), using the [recommended settings](../inference-server/settings.md) and the chat format in the [prompting guide](../prompting/README.md).
