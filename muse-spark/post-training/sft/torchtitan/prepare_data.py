"""
prepare_data.py -- convert the public example datasets to the chat JSONL that
sft.py reads, with images and audio written to files and referenced by path.

    python prepare_data.py dolly       /data/dolly
    python prepare_data.py llava       /data/llava
    python prepare_data.py librispeech /data/librispeech

Each writes train.jsonl and val.jsonl to the output directory. Requires
huggingface_hub, pyarrow, Pillow, and soundfile.
"""

import io
import json
import os
import sys

from huggingface_hub import hf_hub_download

NUM_VAL = 500


def write_splits(rows: list[dict], out: str) -> None:
    for name, part in (("train", rows[:-NUM_VAL]), ("val", rows[-NUM_VAL:])):
        with open(os.path.join(out, f"{name}.jsonl"), "w") as f:
            f.writelines(json.dumps(row) + "\n" for row in part)
        print(f"{out}/{name}.jsonl: {len(part)} conversations")


def dolly(out: str) -> None:
    """Instruction, optional context, response -> one user and one assistant turn."""
    path = hf_hub_download(
        "databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset"
    )
    rows = []
    for line in open(path):
        r = json.loads(line)
        prompt = r["instruction"] + (f"\n\n{r['context']}" if r["context"] else "")
        rows.append({"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": r["response"]},
        ]})
    write_splits(rows, out)


def llava(out: str) -> None:
    """Already chat-formatted; write each image to a file and point at it."""
    import pyarrow.parquet as pq
    from PIL import Image

    path = hf_hub_download(
        "HuggingFaceH4/llava-instruct-mix-vsft",
        "data/test-00000-of-00002.parquet",
        repo_type="dataset",
    )
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    rows = []
    for i, row in enumerate(pq.read_table(path).to_pylist()):
        paths = []
        for j, image in enumerate(row["images"]):
            fmt = (Image.open(io.BytesIO(image["bytes"])).format or "png").lower()
            paths.append(os.path.join(out, "images", f"{i:05d}_{j}.{fmt}"))
            with open(paths[-1], "wb") as f:
                f.write(image["bytes"])
        messages = [
            {"role": m["role"], "content": [
                {"type": "image", "path": paths[p["index"]]} if p["type"] == "image"
                else {"type": "text", "text": p["text"]}
                for p in m["content"]
            ]}
            for m in row["messages"]
        ]
        rows.append({"messages": messages})
    write_splits(rows, out)


def librispeech(out: str) -> None:
    """Read speech -> "Transcribe this audio." with the transcript as the answer."""
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf

    path = hf_hub_download(
        "openslr/librispeech_asr", "clean/validation/0000.parquet", repo_type="dataset"
    )
    os.makedirs(os.path.join(out, "wav"), exist_ok=True)
    rows = []
    for row in pq.read_table(path).to_pylist():
        audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="int16")
        wav = os.path.join(out, "wav", f"{row['id']}.wav")
        sf.write(wav, np.asarray(audio), rate, subtype="PCM_16")
        text = row["text"]
        rows.append({"messages": [
            {"role": "user", "content": [
                {"type": "audio", "path": wav},
                {"type": "text", "text": "Transcribe this audio."},
            ]},
            {"role": "assistant", "content": text[0] + text[1:].lower() + "."},
        ]})
    write_splits(rows, out)


if __name__ == "__main__":
    name, out = sys.argv[1], os.path.abspath(sys.argv[2])
    os.makedirs(out, exist_ok=True)
    {"dolly": dolly, "llava": llava, "librispeech": librispeech}[name](out)
