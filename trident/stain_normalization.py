"""Optional H&E stain normalization applied to every patch before the patch encoder.

TRIDENT does not normalize stains by default: foundation encoders are usually
trained on raw colours. Enable it with ``--stain_norm macenko|reinhard`` in
``run_batch_of_slides.py``. The normalizer is inserted in front of the
encoder's own ``eval_transforms`` and the feature directory gets the method as
a suffix (``features_<encoder>_<method>``), so normalized and raw features of
the same encoder never overwrite each other. Patch coordinates are shared.

Implementation uses ``torchstain`` (``pip install torchstain``).
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

STAIN_NORMALIZATION_METHODS = ("macenko", "reinhard")
_LOGGER = logging.getLogger(__name__)


def _to_tensor_255(image: Image.Image) -> torch.Tensor:
    """PIL RGB image -> float tensor [3, H, W] in 0..255, the layout torchstain expects."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class StainNormalizer:
    """Callable ``PIL -> PIL`` that normalizes H&E stain appearance.

    Args:
        method: ``"macenko"`` (default reference stain vectors from the Macenko
            paper unless ``target`` is given) or ``"reinhard"`` (requires ``target``).
        target: optional path to a reference RGB image whose stain statistics
            become the normalization target.
        fallback_on_error: patches where the method fails (e.g. almost no
            tissue, singular stain matrix) are passed through unchanged instead
            of aborting the slide. Failures are counted in ``failures``.
    """

    def __init__(self, method: str = "macenko", target: Optional[str] = None, fallback_on_error: bool = True):
        method = (method or "").lower()
        if method not in STAIN_NORMALIZATION_METHODS:
            raise ValueError(f"Unknown stain normalization method {method!r}; choose from {STAIN_NORMALIZATION_METHODS}")
        try:
            import torchstain
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError("Stain normalization requires the `torchstain` package: pip install torchstain") from error
        if method == "reinhard" and target is None:
            raise ValueError("Reinhard normalization requires a reference image (--stain_target)")
        self.method = method
        self.target = str(target) if target is not None else None
        self.fallback_on_error = fallback_on_error
        self.failures = 0
        self.calls = 0
        self._lock = threading.Lock()
        factory = torchstain.normalizers.MacenkoNormalizer if method == "macenko" else torchstain.normalizers.ReinhardNormalizer
        self._normalizer = factory(backend="torch")
        if target is not None:
            path = Path(target)
            if not path.is_file():
                raise FileNotFoundError(f"Stain target image not found: {path}")
            with Image.open(path) as reference:
                self._normalizer.fit(_to_tensor_255(reference))

    @property
    def suffix(self) -> str:
        return self.method

    def __call__(self, image: Image.Image) -> Image.Image:
        tensor = _to_tensor_255(image)
        try:
            if self.method == "macenko":
                normalized, _, _ = self._normalizer.normalize(I=tensor, stains=False)
            else:
                normalized = self._normalizer.normalize(I=tensor)
            array = normalized.detach().cpu().numpy()
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(f"unexpected normalized shape {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError("normalized patch contains NaN/Inf")
            result = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")
        except Exception as error:  # noqa: BLE001 - any numerical failure falls back to the raw patch
            if not self.fallback_on_error:
                raise
            with self._lock:
                self.failures += 1
                failures = self.failures
            if failures <= 3:
                _LOGGER.warning("Stain normalization (%s) failed on a patch, using the raw patch: %s", self.method, error)
            result = image.convert("RGB") if image.mode != "RGB" else image
        with self._lock:
            self.calls += 1
        return result

    def __repr__(self) -> str:
        return f"StainNormalizer(method={self.method!r}, target={self.target!r})"


def apply_stain_normalization(patch_encoder, normalizer: StainNormalizer):
    """Prepend ``normalizer`` to a patch encoder's transforms and suffix its name.

    After this call ``patch_encoder.eval_transforms`` runs the normalizer on
    the PIL patch first, and ``patch_encoder.enc_name`` becomes
    ``<name>_<method>`` so TRIDENT writes features to ``features_<name>_<method>``.
    """
    from torchvision import transforms

    if getattr(patch_encoder, "stain_normalizer", None) is not None:
        raise RuntimeError("Stain normalization is already attached to this encoder")
    original = patch_encoder.eval_transforms
    patch_encoder.eval_transforms = transforms.Compose([normalizer, original]) if original is not None else normalizer
    patch_encoder.stain_normalizer = normalizer
    patch_encoder.raw_enc_name = patch_encoder.enc_name
    patch_encoder.enc_name = f"{patch_encoder.enc_name}_{normalizer.suffix}"
    return patch_encoder
