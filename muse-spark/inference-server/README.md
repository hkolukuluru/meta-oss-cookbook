# Run inference

Run Muse Spark Open on your own infrastructure or call it through a hosted inference provider. The self-hosted runtimes load the released checkpoints and use the Muse Spark Open chat format described in the [prompting guide](../prompting/README.md).

| Runtime | Best for | Checkpoints | Serving API |
| --- | --- | --- | --- |
| [SGLang](sglang.md) | High-throughput serving for concurrent users | BF16, NVFP4 | OpenAI-compatible HTTP |
| [vLLM](vllm.md) | Production serving, high throughput | BF16, NVFP4 | OpenAI-compatible HTTP |
| [TensorRT-LLM](tensorrt-llm.md) | Optimized serving on NVIDIA Blackwell, on a single 4-GPU node | NVFP4 | OpenAI-compatible HTTP |
| [NVIDIA Dynamo](dynamo.md) | Disaggregated serving and KV-aware routing | NVFP4 | OpenAI-compatible HTTP |
| [Hugging Face Transformers](transformers.md) | Native Python implementation for research and custom pipelines | BF16, NVFP4 | Python API |
| [llama.cpp](llama-cpp.md) | Portable inference across Metal, CUDA, ROCm, Vulkan, and CPU | ggml-org GGUFs | OpenAI-compatible HTTP |
| [Unsloth](unsloth.md) | Local inference on macOS, Windows, Linux, and WSL | Unsloth Dynamic GGUFs | OpenAI- and Anthropic-compatible HTTP |
| [MLX](mlx.md) | Inference on Apple silicon, across two or four Mac Studio computers | NVFP4 | OpenAI-compatible HTTP |

To call the model without provisioning GPUs, use a [hosted provider](../hosted/README.md).

Muse Spark Open is a mixture-of-experts model that serves across multiple GPUs. NVFP4 serves at TP4 on 4 × B300 and BF16 at TP8 on 8 × B300. Read [Choose a checkpoint](../quickstart/README.md#choose-a-checkpoint) before you provision hardware.

> [!TIP]
> **Stream long reasoning**
>
> Muse Spark Open produces long reasoning traces at higher reasoning strengths. Request streaming (`stream: true`) for reasoning workloads so long generations don't hit request timeouts. See [Recommended settings](settings.md#stream-long-generations).

## Self-host

Each block below is a quick start for one runtime: a command that starts a server or, for Transformers, a script that loads the model and generates a reply. Each runtime's page covers prerequisites, checkpoint variants, and sending requests.

**SGLang**

```bash
sglang serve \
  --model-path meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
  --tp-size 4 \
  --ep-size 4 \
  --quantization modelopt_fp4 \
  --moe-runner-backend flashinfer_trtllm \
  --mem-fraction-static 0.8 \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-layernorm-sp \
  --enable-megamoe-for-layernorm-sp \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path RadixArk/Muse-Spark-Open-DFlash2 \
  --speculative-draft-attention-backend fa4 \
  --speculative-num-draft-tokens 8 \
  --mm-feature-transport cpu \
  --enable-hierarchical-cache \
  --hicache-size 200 \
  --reasoning-parser muse \
  --tool-call-parser muse
```

**vLLM**

```bash
export VLLM_USE_RUST_FRONTEND=1

vllm serve meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len 1048576 \
    --reasoning-parser muse \
    --tool-call-parser muse \
    --enable-auto-tool-choice \
    --speculative-config '{"method":"dspark","model": "Inferact/Muse-Spark-Open-DSpark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}'
```

**TensorRT-LLM**

```bash
# serve_config.yaml: see tensorrt-llm.md
trtllm-serve meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
    --config serve_config.yaml \
    --custom_tokenizer musespark_open \
    --tool_parser musespark \
    --reasoning_parser musespark
```

**Transformers**

```python
# Launch with torchrun, which sets WORLD_SIZE; see transformers.md
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

**llama.cpp**

```bash
llama serve -hf ggml-org/Muse-Spark-Open-816B-A42B-GGUF
```

**Unsloth**

```bash
unsloth run \
  --model unsloth/Muse-Spark-Open-816B-A42B-GGUF:UD-Q2_K_XL \
  -H 127.0.0.1 \
  -p 8888 \
  --context-length 32768 \
  --temperature 1.0 \
  --top-p 1.0 \
  --min-p 0.0 \
  --reasoning-effort medium
```

**MLX**

```bash
# hostfile.json lists every Mac; see mlx.md
mlx.launch --hostfile hostfile.json --backend jaccl --env MLX_METAL_FAST_SYNCH=1 -- \
  mlx_lm.server --model meta-models/Muse-Spark-Open-816B-A42B-NVFP4 --port 8080
```

The SGLang and vLLM commands both serve NVFP4 on 4 × B300; the SGLang command adds MegaMoE. Each runtime's page has the other variants.

## Runtimes

- [SGLang](sglang.md)
- [vLLM](vllm.md)
- [TensorRT-LLM](tensorrt-llm.md)
- [NVIDIA Dynamo](dynamo.md)
- [Hugging Face Transformers](transformers.md)
- [llama.cpp](llama-cpp.md)
- [Unsloth](unsloth.md)
- [MLX](mlx.md)

## Next steps

Once your server is up, set the sampling parameters, parallelism, and stop tokens from the [recommended settings](settings.md), and frame your requests with the [prompting guide](../prompting/README.md). To fine-tune the model, see [Post-training](../post-training/README.md).
