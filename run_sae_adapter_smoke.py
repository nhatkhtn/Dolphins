"""GPU smoke verification for the Dolphins SAE adapter.

Run this in the pinned Dolphins environment on a GPU.  It uses the repository's
single pinned image as a one-frame video, which intentionally leaves the
unresolved six-camera policy outside this smoke test.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_sample_dolphins import (  # noqa: E402
    SAMPLE_PATH,
    get_model_inputs,
    load_pretrained_model,
)
from sae_adapter import DolphinsSAEAdapter  # noqa: E402


def _assert_same_logits(left, right):
    torch.testing.assert_close(left, right, rtol=0, atol=0)


def _assert_batched_close(left, right, label):
    """Compare B=2 and B=1 paths, which may select different CUDA kernels."""

    try:
        torch.testing.assert_close(left, right, rtol=5e-2, atol=3e-1)
    except AssertionError as exc:
        delta = (left.float() - right.float()).abs().max().item()
        raise AssertionError(
            f"{label} drifted beyond the B=2 tolerance (max absolute difference {delta:g})"
        ) from exc
    return (left.float() - right.float()).abs().max().item()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Dolphins SAE smoke test requires CUDA")

    with open(SAMPLE_PATH) as handle:
        sample = json.load(handle)
    _, relative_image_path = next(iter(sample["image_path"].items()))
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(repo_root, relative_image_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)

    model, image_processor, tokenizer, device = load_pretrained_model()
    vision_x, inputs = get_model_inputs(
        image_path,
        sample["question"],
        model,
        image_processor,
        tokenizer,
        device,
    )
    vision_x = vision_x.to(device).half()
    lang_x = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    adapter = DolphinsSAEAdapter(
        model,
        # The factory places a visual cross-attention layer every four blocks,
        # starting at zero-based block 3. Capture a genuinely fused layer.
        layer=3,
        checkpoint="gray311/Dolphins/checkpoint.pt",
        tokenizer_path="anas-awadalla/mpt-7b",
        prompt_template="USER: <image> is a driving video. {question} GPT:<answer>",
    )

    # Count native visual encoder calls and record their inputs.  This proves
    # each example is re-encoded rather than reusing conditioned visual state.
    visual_inputs = []

    def record_visual_input(_module, args):
        if args:
            visual_inputs.append(args[0].detach().clone())

    visual_hook = model.vision_encoder.register_forward_pre_hook(record_visual_input)
    try:
        with torch.no_grad():
            baseline = model(
                vision_x=vision_x,
                lang_x=lang_x,
                attention_mask=attention_mask,
            )
            baseline_ids = model.generate(
                vision_x=vision_x,
                lang_x=lang_x,
                attention_mask=attention_mask,
                num_beams=1,
                max_new_tokens=16,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                do_sample=False,
            )
            captured = adapter.forward(
                vision_x=vision_x,
                lang_x=lang_x,
                attention_mask=attention_mask,
            )
            generation_hook = adapter._target.register_forward_hook(
                lambda _module, _inputs, _output: None
            )
            try:
                adapted_ids = model.generate(
                    vision_x=vision_x,
                    lang_x=lang_x,
                    attention_mask=attention_mask,
                    num_beams=1,
                    max_new_tokens=16,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                    do_sample=False,
                )
            finally:
                generation_hook.remove()
            disabled = adapter.forward(
                vision_x=vision_x,
                lang_x=lang_x,
                attention_mask=attention_mask,
                capture=False,
            )

            changed_vision_x = torch.flip(vision_x, dims=(-1,)).contiguous()
            visual_calls_before_changed = len(visual_inputs)
            changed = adapter.forward(
                vision_x=changed_vision_x,
                lang_x=lang_x,
                attention_mask=attention_mask,
            )
            if len(visual_inputs) != visual_calls_before_changed + 1:
                raise AssertionError("changed pass did not make one visual-encoder call")
            changed_visual_input = visual_inputs[-1]

            batch_vision_x = torch.cat((vision_x, changed_vision_x), dim=0)
            batch_lang_x = torch.cat((lang_x, lang_x), dim=0)
            batch_attention_mask = torch.cat((attention_mask, attention_mask), dim=0)
            batch_baseline = model(
                vision_x=batch_vision_x,
                lang_x=batch_lang_x,
                attention_mask=batch_attention_mask,
            )
            batch = adapter.forward(
                vision_x=batch_vision_x,
                lang_x=batch_lang_x,
                attention_mask=batch_attention_mask,
            )

            duplicate_vision_x = torch.cat((vision_x, vision_x), dim=0)
            duplicate_lang_x = torch.cat((lang_x, lang_x), dim=0)
            duplicate_attention_mask = torch.cat(
                (attention_mask, attention_mask), dim=0
            )
            batch_ids = model.generate(
                vision_x=duplicate_vision_x,
                lang_x=duplicate_lang_x,
                attention_mask=duplicate_attention_mask,
                num_beams=1,
                max_new_tokens=16,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                do_sample=False,
            )
            generation_hook = adapter._target.register_forward_hook(
                lambda _module, _inputs, _output: None
            )
            try:
                hooked_batch_ids = model.generate(
                    vision_x=duplicate_vision_x,
                    lang_x=duplicate_lang_x,
                    attention_mask=duplicate_attention_mask,
                    num_beams=1,
                    max_new_tokens=16,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                    do_sample=False,
                )
            finally:
                generation_hook.remove()
    finally:
        visual_hook.remove()

    _assert_same_logits(baseline.logits, captured.output.logits)
    _assert_same_logits(baseline.logits, disabled.output.logits)
    _assert_same_logits(batch_baseline.logits, batch.output.logits)
    if not torch.equal(baseline_ids, adapted_ids):
        raise AssertionError("baseline and adapted greedy token IDs differ")
    if not torch.equal(batch_ids, hooked_batch_ids):
        raise AssertionError("batched greedy token IDs changed under the temporary hook")
    if not (
        torch.equal(batch_ids[0], baseline_ids[0])
        and torch.equal(batch_ids[1], baseline_ids[0])
    ):
        raise AssertionError("batched greedy token IDs disagree with singleton decoding")
    if captured.activations.ndim != 2:
        raise AssertionError("captured activations are not [n, d_model]")
    model_d_model = int(adapter._lang_encoder.config.d_model)
    if (
        captured.activations.shape[-1] != model_d_model
        or captured.metadata.d_model != model_d_model
    ):
        raise AssertionError("activation width does not match MPT d_model")
    final_index = captured.metadata.final_prompt_sequence_indices[0]
    expected_first_token = baseline.logits[0, final_index].argmax().item()
    actual_first_token = baseline_ids[0, lang_x.shape[1]].item()
    if expected_first_token != actual_first_token:
        raise AssertionError("final prompt position does not produce first token")
    if not adapter.conditioned_layers_cleared():
        raise AssertionError("Flamingo conditioned layers were not cleared")
    if batch.activations.shape[0] != 2 * captured.activations.shape[0]:
        raise AssertionError("different-example batch did not yield per-example residuals")
    clean_batch_delta = _assert_batched_close(
        batch.output.logits[0],
        captured.output.logits[0],
        "clean B=2/B=1 logits",
    )
    changed_batch_delta = _assert_batched_close(
        batch.output.logits[1],
        changed.output.logits[0],
        "changed B=2/B=1 logits",
    )
    activation_count = captured.activations.shape[0]
    clean_activation_delta = _assert_batched_close(
        batch.activations[:activation_count],
        captured.activations,
        "clean B=2/B=1 residuals",
    )
    changed_activation_delta = _assert_batched_close(
        batch.activations[activation_count:],
        changed.activations,
        "changed B=2/B=1 residuals",
    )
    if len(visual_inputs) < 7:
        raise AssertionError(
            "native visual encoder was not called for each forward/generation pass"
        )
    if torch.equal(changed_visual_input, vision_x):
        raise AssertionError("changed image did not reach the native visual path")

    print("Dolphins SAE adapter smoke test passed")
    print("module_path:", captured.metadata.module_path)
    print("activations:", tuple(captured.activations.shape))
    print("final_prompt_sequence_index:", final_index)
    print("native_visual_encoder_calls:", len(visual_inputs))
    print("changed-image logits differ:", not torch.equal(baseline.logits, changed.output.logits))
    print(
        "batch_max_abs_delta:",
        max(
            clean_batch_delta,
            changed_batch_delta,
            clean_activation_delta,
            changed_activation_delta,
        ),
    )


if __name__ == "__main__":
    main()
