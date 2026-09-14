"""Small activation-capture adapter for the native Dolphins forward path.

The Dolphins language encoder is an OpenFlamingo language model.  Its decoder
layers are ``FlamingoLayer`` wrappers, so a hook on the underlying MPT block
would miss the gated visual cross-attention that runs immediately before the
block.  This adapter resolves and hooks the wrapper at runtime, and leaves the
model's normal forward return value untouched.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn


@dataclass
class DolphinsActivationMetadata:
    """Provenance and text-position metadata for captured residuals."""

    checkpoint: Optional[str]
    tokenizer_path: Optional[str]
    prompt_template: Optional[str]
    layer: int
    module_path: str
    capture_after_final_norm: bool
    valid_text_mask: torch.Tensor
    selected_sequence_indices: Tuple[Tuple[int, ...], ...]
    final_prompt_sequence_indices: Tuple[int, ...]
    d_model: int


@dataclass
class DolphinsForwardResult:
    """Native output plus optional SAE-ready residuals."""

    output: Any
    activations: Optional[torch.Tensor]
    metadata: Optional[DolphinsActivationMetadata]


def _underlying_language_encoder(model: nn.Module) -> nn.Module:
    """Return the object that owns Flamingo's decoder-layer accessor.

    LoRA wraps the original language encoder in a ``PeftModel``.  Depending on
    the PEFT version, methods on the original mixin are either delegated by
    ``__getattr__`` or only available through ``base_model.model``.  Walk the
    small wrapper chain instead of assuming either layout.
    """

    current = getattr(model, "lang_encoder", None)
    if current is None:
        raise TypeError("Dolphins model must expose lang_encoder")

    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if callable(getattr(current, "_get_decoder_layers", None)):
            return current

        next_module = None
        for attribute in ("base_model", "model"):
            candidate = getattr(current, attribute, None)
            if isinstance(candidate, nn.Module) and id(candidate) not in seen:
                next_module = candidate
                break
        current = next_module

    raise RuntimeError(
        "Could not find Flamingo's decoder-layer accessor under model.lang_encoder"
    )


def _module_path(root: nn.Module, target: nn.Module) -> Optional[str]:
    for name, module in root.named_modules():
        if module is target:
            return name
    return None


def _residual_from_layer_output(output: Any) -> torch.Tensor:
    """Extract the hidden-state tensor from an MPT layer tuple."""

    if isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError("Dolphins decoder layer returned an empty tuple")
        output = output[0]
    if not torch.is_tensor(output) or output.ndim != 3:
        raise RuntimeError(
            "Dolphins fused decoder hook did not return a [batch, sequence, d_model] tensor"
        )
    return output


class DolphinsSAEAdapter:
    """Capture layer residuals around Dolphins' native multimodal forward.

    ``layer`` is zero-based and indexes the runtime Flamingo decoder list.  A
    caller may use any layer in range; the adapter does not pick a training
    layer or introduce a camera policy.  ``forward`` always calls the native
    ``model(..., vision_x=..., lang_x=...)`` path and always clears conditioned
    layers before returning, including when the native call raises.
    """

    def __init__(
        self,
        model: nn.Module,
        layer: int,
        *,
        checkpoint: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        prompt_template: Optional[str] = None,
    ) -> None:
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            raise ValueError("layer must be a non-negative zero-based integer")
        self.model = model
        self.layer = layer
        self.checkpoint = checkpoint
        self.tokenizer_path = tokenizer_path
        self.prompt_template = prompt_template
        self._lang_encoder = _underlying_language_encoder(model)
        self._target, self.module_path = self._resolve_target_layer(layer)

    def _resolve_target_layer(self, layer_index: int) -> Tuple[nn.Module, str]:
        """Resolve a fused Flamingo layer from the live model module tree."""

        layers = self._lang_encoder._get_decoder_layers()
        if not isinstance(layers, (nn.ModuleList, list, tuple)):
            raise RuntimeError(
                "Flamingo decoder-layer accessor did not return a module sequence"
            )
        if layer_index >= len(layers):
            raise ValueError(
                "layer {} is out of range for {} Dolphins decoder layers".format(
                    layer_index, len(layers)
                )
            )

        target = layers[layer_index]
        # Hook the Flamingo wrapper, whose output includes cross-attention and
        # the decoder block; the child MPT block alone would miss conditioning.
        if not (
            isinstance(target, nn.Module)
            and hasattr(target, "decoder_layer")
            and hasattr(target, "condition_vis_x")
        ):
            raise RuntimeError(
                "Resolved Dolphins decoder layer is not a FlamingoLayer wrapper; "
                "refusing to hook an unconditioned MPT block"
            )

        path = _module_path(self.model.lang_encoder, target)
        if path is None:
            # A PEFT proxy can hide the path from its named-module iterator;
            # use the live decoder accessor to provide a useful diagnostic.
            path = "<lang_encoder._get_decoder_layers()[{}]>".format(layer_index)
        else:
            path = "model.lang_encoder." + path
        return target, path

    def _clear_conditioned_layers(self) -> None:
        uncache = getattr(self.model, "uncache_media", None)
        if callable(uncache):
            uncache()
            return
        clear = getattr(self._lang_encoder, "clear_conditioned_layers", None)
        if not callable(clear):
            raise RuntimeError(
                "Dolphins language encoder does not expose clear_conditioned_layers"
            )
        clear()
        if hasattr(self._lang_encoder, "_use_cached_vision_x"):
            self._lang_encoder._use_cached_vision_x = False

    def conditioned_layers_cleared(self) -> bool:
        """Return whether Flamingo has no image state left from the last pass."""

        is_conditioned = getattr(self._lang_encoder, "is_conditioned", None)
        if not callable(is_conditioned):
            raise RuntimeError(
                "Dolphins language encoder does not expose is_conditioned"
            )
        return not bool(is_conditioned()) and not bool(
            getattr(self._lang_encoder, "_use_cached_vision_x", False)
        )

    def _native_forward(self, kwargs: Dict[str, Any]) -> Any:
        requested_clear = kwargs.pop("clear_conditioned_layers", True)
        if requested_clear is not True:
            raise ValueError(
                "DolphinsSAEAdapter requires clear_conditioned_layers=True"
            )
        kwargs["clear_conditioned_layers"] = True
        try:
            return self.model(**kwargs)
        finally:
            self._clear_conditioned_layers()

    @staticmethod
    def _valid_text_mask(
        lang_x: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if lang_x.ndim != 2:
            raise ValueError("lang_x must have shape [batch, sequence]")
        if attention_mask is None:
            return torch.ones(
                lang_x.shape, dtype=torch.bool, device=lang_x.device
            )
        if attention_mask.shape != lang_x.shape:
            raise ValueError("attention_mask must have the same shape as lang_x")
        return attention_mask.to(dtype=torch.bool)

    def forward(
        self,
        *,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        capture: bool = True,
        **kwargs: Any,
    ) -> DolphinsForwardResult:
        """Run native Dolphins forward and optionally return SAE activations.

        The model output is never replaced or edited.  With ``capture=False``
        this is a thin native-forward call and ``activations``/``metadata`` are
        ``None``; this is useful for the output-equivalence check.
        """

        if vision_x.ndim != 6:
            raise ValueError("vision_x must have shape [B, T_img, F, C, H, W]")
        if lang_x.ndim != 2:
            raise ValueError("lang_x must have shape [B, sequence]")
        if vision_x.shape[0] != lang_x.shape[0]:
            raise ValueError("vision_x and lang_x batch sizes must match")

        valid_mask = self._valid_text_mask(lang_x, attention_mask)
        captured = []

        handle = None
        if capture:

            def save_output(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any):
                captured.append(_residual_from_layer_output(output).detach())

            handle = self._target.register_forward_hook(save_output)

        native_kwargs = dict(kwargs)
        native_kwargs.update(
            {
                "vision_x": vision_x,
                "lang_x": lang_x,
                "attention_mask": attention_mask,
            }
        )
        try:
            output = self._native_forward(native_kwargs)
        finally:
            if handle is not None:
                handle.remove()

        if not capture:
            return DolphinsForwardResult(output, None, None)
        if len(captured) != 1:
            raise RuntimeError(
                "Expected exactly one Dolphins fused-layer invocation, got {}".format(
                    len(captured)
                )
            )

        residual = captured[0]
        if residual.shape[:2] != lang_x.shape:
            raise RuntimeError(
                "Dolphins residual sequence shape {} does not match native text shape {}".format(
                    tuple(residual.shape[:2]), tuple(lang_x.shape)
                )
            )
        if valid_mask.device != residual.device:
            valid_mask = valid_mask.to(residual.device)

        selected_indices = []
        selected = []
        for batch_index in range(residual.shape[0]):
            indices = torch.nonzero(valid_mask[batch_index], as_tuple=False).flatten()
            if indices.numel() == 0:
                raise ValueError(
                    "attention_mask contains no valid text positions for batch item {}".format(
                        batch_index
                    )
                )
            selected_indices.append(tuple(int(index) for index in indices.tolist()))
            selected.append(residual[batch_index, indices])

        activations = torch.cat(selected, dim=0)
        metadata = DolphinsActivationMetadata(
            checkpoint=self.checkpoint,
            tokenizer_path=self.tokenizer_path,
            prompt_template=self.prompt_template,
            layer=self.layer,
            module_path=self.module_path,
            capture_after_final_norm=False,
            valid_text_mask=valid_mask.detach().clone(),
            selected_sequence_indices=tuple(selected_indices),
            final_prompt_sequence_indices=tuple(
                indices[-1] for indices in selected_indices
            ),
            d_model=int(residual.shape[-1]),
        )
        return DolphinsForwardResult(output, activations, metadata)
