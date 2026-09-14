"""CPU checks for the Dolphins activation adapter.

These tests use a tiny Flamingo-shaped model.  The real checkpoint smoke test
is ``run_sae_adapter_smoke.py`` and must run in Dolphins' original GPU
environment.
"""

import os
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sae_adapter import DolphinsSAEAdapter


class _FakeFlamingoLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.decoder_layer = nn.Identity()
        self.vis_x = None
        self.d_model = d_model

    def condition_vis_x(self, value):
        self.vis_x = value

    def condition_media_locations(self, value):
        del value

    def condition_use_cached_media(self, value):
        del value

    def is_conditioned(self):
        return self.vis_x is not None

    def forward(self, x, **kwargs):
        del kwargs
        if self.vis_x is None:
            raise RuntimeError("fake layer was not conditioned")
        return (x + self.vis_x, None)


class _FakeLanguageEncoder(nn.Module):
    def __init__(self, n_layers=2, d_model=3):
        super().__init__()
        self._use_cached_vision_x = False
        self.transformer = nn.Module()
        self.transformer.blocks = nn.ModuleList(
            [_FakeFlamingoLayer(d_model) for _ in range(n_layers)]
        )

    def _get_decoder_layers(self):
        return self.transformer.blocks

    def clear_conditioned_layers(self):
        for layer in self.transformer.blocks:
            layer.condition_vis_x(None)

    def is_conditioned(self):
        return all(layer.is_conditioned() for layer in self.transformer.blocks)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        x = input_ids.float().unsqueeze(-1).expand(-1, -1, 3)
        for layer in self.transformer.blocks:
            x, _ = layer(x)
        return type("FakeOutput", (), {"logits": x})()


class _FakeDolphins(nn.Module):
    def __init__(self):
        super().__init__()
        self.lang_encoder = _FakeLanguageEncoder()

    def forward(
        self,
        vision_x,
        lang_x,
        attention_mask=None,
        clear_conditioned_layers=True,
        **kwargs
    ):
        del kwargs
        # Make image values affect the native conditioned state so the test
        # catches stale visual state across different examples.
        visual_value = vision_x.flatten(1).mean(dim=1).view(-1, 1, 1)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(visual_value)
        output = self.lang_encoder(
            input_ids=lang_x, attention_mask=attention_mask
        )
        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()
        return output

class DolphinsAdapterTest(unittest.TestCase):
    def test_capture_preserves_output_selects_mask_and_clears_state(self):
        model = _FakeDolphins()
        adapter = DolphinsSAEAdapter(
            model,
            layer=1,
            checkpoint="test-checkpoint",
            tokenizer_path="test-tokenizer",
            prompt_template="USER: <image> {question} GPT:<answer>",
        )
        vision_x = torch.ones(2, 1, 1, 1, 2, 2)
        vision_x[1].mul_(2)
        lang_x = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])

        baseline = model(
            vision_x=vision_x,
            lang_x=lang_x,
            attention_mask=attention_mask,
        )
        result = adapter.forward(
            vision_x=vision_x,
            lang_x=lang_x,
            attention_mask=attention_mask,
        )
        first = adapter.forward(
            vision_x=vision_x[:1],
            lang_x=lang_x[:1],
            attention_mask=attention_mask[:1],
        )
        second = adapter.forward(
            vision_x=vision_x[1:],
            lang_x=lang_x[1:],
            attention_mask=attention_mask[1:],
        )

        torch.testing.assert_close(result.output.logits, baseline.logits)
        torch.testing.assert_close(
            result.activations,
            torch.cat((first.activations, second.activations)),
        )
        self.assertEqual(tuple(result.activations.shape), (7, 3))
        self.assertEqual(result.metadata.d_model, 3)
        self.assertEqual(result.metadata.selected_sequence_indices, ((0, 1, 2), (0, 1, 2, 3)))
        self.assertEqual(result.metadata.final_prompt_sequence_indices, (2, 3))
        self.assertIn("lang_encoder.transformer.blocks.1", result.metadata.module_path)
        self.assertFalse(result.metadata.capture_after_final_norm)
        self.assertTrue(adapter.conditioned_layers_cleared())

        disabled = adapter.forward(
            vision_x=vision_x,
            lang_x=lang_x,
            attention_mask=attention_mask,
            capture=False,
        )
        torch.testing.assert_close(disabled.output.logits, baseline.logits)
        self.assertIsNone(disabled.activations)
        self.assertTrue(adapter.conditioned_layers_cleared())

    def test_different_images_are_reencoded_after_cleanup(self):
        model = _FakeDolphins()
        adapter = DolphinsSAEAdapter(model, layer=0)
        lang_x = torch.tensor([[1, 2]])
        first = adapter.forward(
            vision_x=torch.zeros(1, 1, 1, 1, 2, 2), lang_x=lang_x
        )
        second = adapter.forward(
            vision_x=torch.ones(1, 1, 1, 1, 2, 2), lang_x=lang_x
        )
        self.assertFalse(torch.equal(first.output.logits, second.output.logits))
        self.assertTrue(adapter.conditioned_layers_cleared())

if __name__ == "__main__":
    unittest.main()
