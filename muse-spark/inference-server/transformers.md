# Hugging Face Transformers

Muse Spark Open comes with a native Hugging Face Transformers implementation, so you can run it using familiar APIs and integrate it into your workflows. It supports text, images, audio, and video, as well as agentic use cases including tool calls. Checkpoints are available in [BF16](#install) and [NVFP4 formats](#nvfp4-weights). You can post-train Muse Spark Open with Transformer Reinforcement Learning (TRL) to adapt it to your own tasks and datasets; see [Post-train with TRL](../post-training/sft/trl/README.md).

## Links

- [Transformers docs](https://hf.co/docs/transformers/main/en/model_doc/muse_spark). These are always the source of truth for using Muse Spark Open with Transformers.
- [Hugging Face collection](https://huggingface.co/collections/meta-models/muse-spark-open)
- [Hugging Face blog post on Muse Spark Open](https://hf.co/blog/meta-muse-spark-open)
- [Post-train with TRL](https://github.com/huggingface/trl/tree/main/examples/sft_muse_spark_open)

## Install

Install Hugging Face Transformers with:

```bash
pip install -U "transformers>=5.19"
```

See the [Transformers installation guide](https://hf.co/docs/transformers/main/en/installation) for more options.

## Run the model

The Muse Spark Open implementation provides `MuseSparkForConditionalGeneration` for text generation from multimodal inputs and `MuseSparkProcessor` for preparing conversations. The processor combines the tokenizer, image and video processors, audio feature extractor, and chat template.

Use a `DistributedConfig` to configure the parallelism settings. `tp_size` is the number of devices across which the dense layers are sharded, and `ep_size` is the number of devices across which the experts are distributed.

The example below shows how to define the distributed config, load the model with `AutoModelForMultimodalLM`, preprocess the inputs, and generate and decode the output.

```python
import os

from transformers import AutoModelForMultimodalLM, AutoProcessor, DistributedConfig

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]),
    ep_size=int(os.environ["WORLD_SIZE"]),
)

processor = AutoProcessor.from_pretrained("meta-models/Muse-Spark-Open-816B-A42B")
model = AutoModelForMultimodalLM.from_pretrained(
    "meta-models/Muse-Spark-Open-816B-A42B",
    distributed_config=distributed_config,
)

messages = [
    {"role": "user", "content": "Explain in one sentence why the sky is usually blue."},
]
inputs = processor.apply_chat_template(
    messages,
    reasoning_effort="low",  # default: xhigh, options: low/medium/high/xhigh
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device).to(model.dtype)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
# to=self<|message|>We need to explain in one sentence why sky is usually blue.
# ...
# <|eom|><|start|>assistant to=user<|message|>The sky appears blue because air molecules scatter shorter-wavelength blue light from the sun more strongly than longer-wavelength red light.<|eot|>
```

Save the script above as `run_muse_spark_open.py` and launch it with `torchrun` on every node using Slurm:

```bash
#!/bin/bash
#SBATCH --job-name=muse-spark-open
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

srun bash -c 'torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=8 \
    --node-rank=$SLURM_NODEID \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    run_muse_spark_open.py'
```

## NVFP4 weights

NVFP4 [quantized weights](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4) allow running the model with lower memory requirements on compatible hardware. The inference code is the same apart from the model ID, but NVFP4 needs a Transformers build from a pull request branch, installed below.

> [!NOTE]
> **Check the Transformers documentation first**
>
> The [Transformers documentation](https://hf.co/docs/transformers/main/en/model_doc/muse_spark) is the source of truth on how to run NVFP4 weights in Transformers. The Transformers package moves faster than this page: check the documentation first for the latest information.

NVFP4 support requires Transformers from the [NVFP4 pull request branch](https://github.com/huggingface/transformers/pull/49015):

```bash
pip install git+https://github.com/huggingface/transformers.git@finegrained-per-scheme-demo
```

The weights can then be loaded by setting the model name to `"meta-models/Muse-Spark-Open-816B-A42B-NVFP4"` as demonstrated below.

```python
import os

from transformers import AutoModelForMultimodalLM, AutoProcessor, DistributedConfig

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]),
    ep_size=int(os.environ["WORLD_SIZE"]),
)

processor = AutoProcessor.from_pretrained("meta-models/Muse-Spark-Open-816B-A42B-NVFP4")
model = AutoModelForMultimodalLM.from_pretrained(
    "meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    distributed_config=distributed_config,
)

messages = [
    {"role": "user", "content": "Explain in one sentence why the sky is usually blue."},
]
inputs = processor.apply_chat_template(
    messages,
    reasoning_effort="low",
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device).to(model.dtype)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
```

Launch it with `torchrun` and the same Slurm script as [Run the model](#run-the-model).

## More

See the full [Transformers documentation for Muse Spark](https://hf.co/docs/transformers/main/en/model_doc/muse_spark) to see how to use the `pipeline` API and run inference for all modalities, including image, video, and audio.

The [Hugging Face blog post on Muse Spark Open](https://hf.co/blog/meta-muse-spark-open) covers the model in depth and more ways to deploy it with Transformers.
