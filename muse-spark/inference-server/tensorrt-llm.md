# TensorRT-LLM

[NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) serves Muse Spark Open on NVIDIA Blackwell GPUs through an OpenAI-compatible server (`trtllm-serve`). The validated deployment runs the NVFP4 checkpoint on a single 4-GPU Blackwell node (tensor parallel 4, expert parallel 4). The configuration uses an FP8 KV cache, CUDA graphs, chunked prefill, and KV-cache block reuse. The server supports image, video, and audio input, guided decoding (JSON mode and JSON schema), tool calling, and optional speculative decoding.

## Links

- [Muse Spark Open deployment guide for TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/feat/musespark-open/docs/source/deployment-guide/deployment-guide-for-musespark-open-on-trtllm.md)
- [Runnable example and validated serving configuration](https://github.com/NVIDIA/TensorRT-LLM/tree/feat/musespark-open/examples/musespark_open)
- [NVIDIA technical blog: scale agentic coding workflows with Muse Spark Open on NVIDIA GB300 NVL72](https://developer.nvidia.com/blog/scale-agentic-coding-workflows-with-metas-muse-spark-open-on-nvidia-gb300-nvl72/)

## Install

Muse Spark Open support is in the TensorRT-LLM `feat/musespark-open` branch. Build TensorRT-LLM from that branch; no pre-built release container or wheel includes Muse Spark Open support.

Clone the branch and build from source following the [build-from-source guide](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source.html). The only differences from that guide are the branch and the Blackwell CUDA architecture (`100-real`):

```bash
git clone --branch feat/musespark-open https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git lfs pull
```

Then, inside the TensorRT-LLM development container (see the guide for the `devel` container tag and the required `docker run` flags), build and install in editable mode, targeting Blackwell:

```bash
python3 scripts/build_wheel.py --use_ccache -a "100-real" \
    --skip_building_wheel --linking_install_binary
pip install -e .
```

You also need a complete Hugging Face-format Muse Spark Open checkpoint with its tokenizer and processor files: [meta-models/Muse-Spark-Open-816B-A42B-NVFP4](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4) (a BF16 variant, [meta-models/Muse-Spark-Open-816B-A42B](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B), is also published). The NVFP4 checkpoint ships the routed-expert weights pre-quantized to NVFP4; no additional quantization step is needed.

## Start the server

Muse Spark Open at TP4/EP4 fits on a single Blackwell 4-GPU node, so `trtllm-serve` launches directly (no multi-node launcher needed). Write the recommended settings (next section) to a YAML file and pass it with `--config`:

```bash
trtllm-serve meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
    --config serve_config.yaml \
    --custom_tokenizer musespark_open \
    --tool_parser musespark \
    --reasoning_parser musespark
```

Pass `--custom_tokenizer musespark_open`, `--tool_parser musespark`, and `--reasoning_parser musespark` explicitly; the two parsers enable tool-call and reasoning-content extraction on `/v1/chat/completions`. The server listens on `localhost:8000`. Add `--host 0.0.0.0 --port 8000` if it needs to be reachable from other hosts. The server is ready when the log shows `Application startup complete`; the first query may take longer due to warmup.

## Recommended settings

Start from the validated aggregated serving configuration, `serve_config.yaml`, and tune it for your workload:

```yaml
tensor_parallel_size: 4
moe_expert_parallel_size: 4
enable_attention_dp: false
max_batch_size: 256
max_num_tokens: 8192
max_seq_len: 49152
trust_remote_code: true
enable_chunked_prefill: true
disable_overlap_scheduler: false
attn_backend: TRTLLM
allreduce_strategy: NCCL
guided_decoding_backend: xgrammar
cuda_graph_config:
  max_batch_size: 256
  enable_padding: true
moe_config:
  backend: TRTLLM
kv_cache_config:
  # FP8 KV cache is required on GB300 (SM103) when chunked prefill or
  # block reuse is enabled.
  dtype: fp8
  enable_block_reuse: true
  enable_partial_reuse: true
  free_gpu_memory_fraction: 0.8
```

Notes:

- **FP8 KV cache is required on GB300 (`SM103`) with chunked prefill or KV reuse.** For this model's head dimension (64), the fused paged-context attention kernel exists only for the FP8 KV-cache dtype; a 16-bit KV cache in that combination is refused at startup.
- `max_seq_len: 49152` is the validated serving window; the model supports up to 1M tokens of context. Longer windows need proportionally more KV-cache memory.
- On out-of-memory errors, reduce `kv_cache_config.free_gpu_memory_fraction`, `max_batch_size`, or `max_seq_len`.

## Send requests

The server exposes the OpenAI-compatible API:

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    "messages": [{"role": "user", "content": "Where is New York?"}],
    "max_tokens": 16384
}'
```

Muse Spark Open is a reasoning model: with `--reasoning_parser musespark`, each
chat response returns the chain-of-thought in a separate `reasoning_content`
field and the user-facing answer in `content`. Both are drawn from the same
`max_tokens` budget, so set `max_tokens` large enough to cover the reasoning
*and* the answer. Too small a budget is spent entirely on reasoning and returns
an empty `content` with `"finish_reason": "length"`.

Guided decoding (JSON mode and JSON schema via the `xgrammar` backend) works through the standard `response_format` field. The constraint applies to `content` (the answer), so the budget must also leave room for the model to finish reasoning first:

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    "messages": [{"role": "user", "content": "Give me a city and its country as JSON."}],
    "max_tokens": 16384,
    "response_format": {"type": "json_object"}
}'
```

Image, video, and audio inputs are accepted through the standard multimodal chat-completions content parts; multimodal token budgets count against `max_num_tokens` and `max_seq_len` like text tokens.

## More

- The [TensorRT-LLM deployment guide for Muse Spark Open](https://github.com/NVIDIA/TensorRT-LLM/blob/feat/musespark-open/docs/source/deployment-guide/deployment-guide-for-musespark-open-on-trtllm.md) covers key YAML options, multimodal token budgets, speculative decoding, benchmarking with `tensorrt_llm.serve.scripts.benchmark_serving`, and troubleshooting.
- For disaggregated serving, with prefill and decode on separate workers, see [NVIDIA Dynamo](dynamo.md), which uses TensorRT-LLM as its backend engine.
