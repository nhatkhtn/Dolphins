"""
Minimal smoke test: one forward pass of Dolphins (OpenFlamingo + MPT-7B + LoRA)
on the pinned DriveBench MCQ sample.

Mirrors the essential wiring of inference.py (model + processor loading,
prompt template, single-frame "video" tensor shape) but feeds a single still
image (as a 1-frame video) and the pinned DriveBench question instead of
inference.py's sample video/instruction.

Do NOT edit inference.py or any other file shipped by upstream Dolphins/;
this is a new, separate script.
"""
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.lora_config import openflamingo_tuning_config
from mllm.src.factory import create_model_and_transforms
from huggingface_hub import hf_hub_download
from peft import LoraConfig

import mllm.src.mpt_lora_patch.modeling_mpt as _mpt_mod

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_PATH = os.path.join(REPO_ROOT, "data", "test_sample.json")

# Monkeypatch (in our own script, not the shipped file) so the MPT-7B weights
# load in fp16 with low_cpu_mem_usage=True. The Slurm cgroup for a single job
# step on this cluster caps host RAM around 16GB by default, which is not
# enough to materialize a 7B-param model in fp32 (~28GB); fp16 + low memory
# loading keeps the peak well under that.
_orig_mpt_from_pretrained = _mpt_mod.MPTForCausalLM.from_pretrained.__func__


def _low_mem_mpt_from_pretrained(cls, *args, **kwargs):
    kwargs.setdefault("torch_dtype", torch.float16)
    kwargs.setdefault("low_cpu_mem_usage", True)
    # Transformers otherwise retains a full checkpoint shard alongside the
    # model while loading. Offload that temporary state so the 60 GB smoke
    # allocation can load MPT-7B without changing the resident model.
    kwargs.setdefault("offload_state_dict", True)
    if torch.cuda.is_available():
        # Keep MPT's checkpoint shards in the allocated B200's HBM while
        # loading. The normal caller moves the finished Flamingo model to
        # the same device immediately afterward.
        kwargs.setdefault("device_map", {"": 0})
    return _orig_mpt_from_pretrained(cls, *args, **kwargs)


_mpt_mod.MPTForCausalLM.from_pretrained = classmethod(_low_mem_mpt_from_pretrained)


def load_pretrained_model():
    peft_config = LoraConfig(**openflamingo_tuning_config)
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14-336",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-7b",
        tokenizer_path="anas-awadalla/mpt-7b",
        cross_attn_every_n_layers=4,
        use_peft=True,
        peft_config=peft_config,
    )

    checkpoint_path = hf_hub_download("gray311/Dolphins", "checkpoint.pt")
    state_dict = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    model.load_state_dict(state_dict, strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.half().cuda()
    model.eval()

    return model, image_processor, tokenizer, device


def get_model_inputs(image_path, instruction, model, image_processor, tokenizer, device):
    # Treat the single still image as a 1-frame "video", matching the shape
    # inference.py's get_model_inputs() builds for real videos:
    # vision_x: (batch=1, frames=1, channels=3, T=1, C, H, W) -> see assert below.
    image = Image.open(image_path).convert("RGB")
    frames = [image]
    vision_x = torch.stack([image_processor(f) for f in frames], dim=0).unsqueeze(0).unsqueeze(0)
    assert vision_x.shape[2] == len(frames)

    prompt = [f"USER: <image> is a driving video. {instruction} GPT:<answer>"]
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("vision_x shape:", vision_x.shape)
    print("prompt:", prompt)

    return vision_x, inputs


def main():
    with open(SAMPLE_PATH) as f:
        sample = json.load(f)

    question = sample["question"]
    # Single pinned camera view for this sample.
    cam, rel_path = next(iter(sample["image_path"].items()))
    image_path = os.path.join(REPO_ROOT, rel_path)
    assert os.path.isfile(image_path), f"missing image: {image_path}"

    model, image_processor, tokenizer, device = load_pretrained_model()
    vision_x, inputs = get_model_inputs(image_path, question, model, image_processor, tokenizer, device)

    generation_kwargs = {
        "max_new_tokens": 64,
        "temperature": 1,
        "top_k": 0,
        "top_p": 1,
        "no_repeat_ngram_size": 3,
        "length_penalty": 1,
        "do_sample": False,
        "early_stopping": True,
    }

    vision_x = vision_x.to(device)
    if device == "cuda":
        vision_x = vision_x.half()

    with torch.no_grad():
        generated_tokens = model.generate(
            vision_x=vision_x,
            lang_x=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=3,
            **generation_kwargs,
        )

    generated_tokens = generated_tokens.cpu().numpy()
    if isinstance(generated_tokens, tuple):
        generated_tokens = generated_tokens[0]

    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    print("=" * 40)
    print("GROUND TRUTH ANSWER (for reference only):", sample["answer"])
    print("DOLPHINS OUTPUT:")
    for t in generated_text:
        print(t)
    print("=" * 40)


if __name__ == "__main__":
    main()
