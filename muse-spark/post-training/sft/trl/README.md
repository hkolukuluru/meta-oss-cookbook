# TRL

[TRL](https://github.com/huggingface/trl) is Hugging Face's post-training library. Its `SFTTrainer` fine-tunes Muse Spark Open (816B parameters, 42B active) on a multi-node cluster: Transformers shards the model across the GPUs while loading it, the experts are split across groups of ranks with token dispatch, and a LoRA adapter is trained on top. The recipe below runs on 8 nodes of 8 H100s and trains 500 steps of supervised fine-tuning.

## Links

- Recipe: [examples/sft_muse_spark_open](https://github.com/huggingface/trl/tree/main/examples/sft_muse_spark_open) in the TRL repository (training script and Slurm launcher).
- [SFTTrainer documentation](https://huggingface.co/docs/trl/sft_trainer)
- [Distributing training](https://huggingface.co/docs/trl/distributing_training) and the [PEFT integration](https://huggingface.co/docs/trl/peft_integration)

## Install

TRL brings datasets and the other training dependencies; PEFT provides the LoRA adapter; Muse Spark Open needs Transformers 5.19 or later.

```bash
pip install trl peft "transformers>=5.19"
```

## Prepare your data

Format your training data as JSONL in the chat format, one conversation per line:

```json
{"messages": [{"role": "system", "content": "You are a medical assistant."}, {"role": "user", "content": "What are the symptoms of type 2 diabetes?"}, {"role": "assistant", "content": "Common symptoms include increased thirst, frequent urination, fatigue, and blurred vision."}]}
{"messages": [{"role": "system", "content": "You are a medical assistant."}, {"role": "user", "content": "How is hypertension diagnosed?"}, {"role": "assistant", "content": "Hypertension is diagnosed by measuring blood pressure. A reading consistently at or above 130/80 mmHg indicates hypertension."}]}
```

`SFTTrainer` applies the model's chat template for you. Load the file with `datasets`, or use any Hub dataset in this format; the recipe uses [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture):

```python
from datasets import load_dataset

train_dataset = load_dataset("json", data_files="train.jsonl", split="train")
# or
train_dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train[:500000]")
```

See [dataset formats](https://huggingface.co/docs/trl/dataset_formats) for the other accepted layouts.

## Start training

The whole parallelism configuration is two numbers passed to Transformers when the model is loaded: `fsdp_size`, the number of GPUs the model is sharded over, and `ep_size`, the number of groups the experts are split into. The model's own expert plan selects token dispatch.

```python
from datasets import load_dataset
from peft import LoraConfig
from transformers.distributed import DistributedConfig

from trl import SFTConfig, SFTTrainer

train_dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train[:500000]")

training_args = SFTConfig(
    output_dir="muse-spark-open-sft",
    model_init_kwargs={
        "dtype": "auto",  # keep the checkpoint's bf16 weights
        "distributed_config": DistributedConfig(fsdp_size=64, ep_size=8),  # shard over 64 GPUs, experts split 8 ways
    },
    per_device_train_batch_size=1,
    max_steps=500,
    max_length=2048,  # 96% of Tulu 3 conversations fit; longer ones are truncated
    dataset_num_proc=32,  # preprocesses the dataset with 32 workers instead of one
    packing=True,  # packs several examples per sequence so almost no token is padding
    logging_steps=1,
    save_only_model=True,  # required, the optimizer state of a model sharded at load time cannot be checkpointed
    save_steps=50,  # the adapter is 194 MB, so saving is cheap
)
trainer = SFTTrainer(
    model="meta-models/Muse-Spark-Open-816B-A42B",
    args=training_args,
    train_dataset=train_dataset,
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
)
trainer.train()
```

Launch it with `torchrun` on every node, 8 processes per node. With Slurm:

```bash
#!/bin/bash
#SBATCH --nodes=8
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0

MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_IP=$(getent ahostsv4 "$MASTER" | awk 'NR==1{print $1}')

srun torchrun \
    --nnodes "$SLURM_NNODES" \
    --nproc_per_node 8 \
    --rdzv_backend c10d \
    --rdzv_endpoint "$MASTER_IP:29700" \
    sft_muse_spark_open.py
```

On 64 H100s this runs at 14.2 s per step with 131k tokens per step, using 40.2 GB per GPU. A checkpoint of the adapter (194 MB) is written every 50 steps.

## Use the tuned model

Training saves the LoRA adapter only, 194 MB per checkpoint. The base weights are about 1.65 TB in bf16, so the adapter is used the same way the base model is run: load the model sharded across your GPUs with `DistributedConfig`, as in [Hugging Face Transformers](../../../inference-server/transformers.md), then attach the adapter with PEFT. Save the script and launch it with `torchrun` on every node, with the same Slurm script as that section.

```python
import os

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor, DistributedConfig

model_id = "meta-models/Muse-Spark-Open-816B-A42B"
distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]),
    ep_size=int(os.environ["WORLD_SIZE"]),
)
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, distributed_config=distributed_config)
model.requires_grad_(False)  # inference: no parameter is trained
model = PeftModel.from_pretrained(model, "muse-spark-open-sft/checkpoint-500")

messages = [{"role": "user", "content": "What is the capital of France?"}]
inputs = processor.apply_chat_template(
    messages, reasoning_effort="low", add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
if int(os.environ["RANK"]) == 0:
    print(processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The adapter alone can also be shared: `trainer.push_to_hub()` uploads it, and anyone with the base model can attach it the same way.

## More

- `SFTTrainer` also supports full fine-tuning, vision-language data and continued pre-training; see the [SFTTrainer documentation](https://huggingface.co/docs/trl/sft_trainer).
- TRL also provides preference and reinforcement-learning trainers (DPO, GRPO); see the [TRL documentation](https://huggingface.co/docs/trl).
- To run on managed hardware, see [training on Hugging Face Jobs](https://huggingface.co/docs/trl/jobs_training).
- Questions and issues: [github.com/huggingface/trl/issues](https://github.com/huggingface/trl/issues).
