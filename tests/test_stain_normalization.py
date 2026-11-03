import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

try:
    import torchstain  # noqa: F401
    HAS_TORCHSTAIN = True
except ImportError:  # pragma: no cover
    HAS_TORCHSTAIN = False

from trident.stain_normalization import StainNormalizer, apply_stain_normalization


def he_like_patch(size=96, seed=0):
    """Synthetic pink/purple patch with enough colour variation for Macenko."""
    rng = np.random.default_rng(seed)
    base = np.full((size, size, 3), (225, 190, 215), dtype=np.float32)   # eosin-like background
    nuclei = rng.random((size, size)) < 0.25
    base[nuclei] = (110, 60, 150)                                        # hematoxylin-like
    base += rng.normal(0, 8, base.shape)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


@unittest.skipUnless(HAS_TORCHSTAIN, "torchstain not installed")
class StainNormalizerTests(unittest.TestCase):
    def test_macenko_returns_rgb_pil_of_same_size(self):
        normalizer = StainNormalizer("macenko")
        image = he_like_patch()
        out = normalizer(image)
        self.assertIsInstance(out, Image.Image)
        self.assertEqual(out.size, image.size)
        self.assertEqual(out.mode, "RGB")
        self.assertEqual(normalizer.failures, 0)
        self.assertFalse(np.array_equal(np.asarray(out), np.asarray(image)))

    def test_reinhard_fits_on_target_image(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp) / "target.png"
            he_like_patch(seed=1).save(target)
            normalizer = StainNormalizer("reinhard", target=str(target))
            out = normalizer(he_like_patch(seed=2))
            self.assertEqual(out.size, (96, 96))
            self.assertEqual(normalizer.failures, 0)

    def test_reinhard_requires_target(self):
        with self.assertRaisesRegex(ValueError, "reference image"):
            StainNormalizer("reinhard")

    def test_unknown_method_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown stain normalization"):
            StainNormalizer("vahadane")

    def test_blank_patch_falls_back_to_raw(self):
        normalizer = StainNormalizer("macenko")
        blank = Image.new("RGB", (64, 64), (255, 255, 255))
        out = normalizer(blank)
        self.assertTrue(np.array_equal(np.asarray(out), np.asarray(blank)))
        self.assertEqual(normalizer.failures, 1)
        strict = StainNormalizer("macenko", fallback_on_error=False)
        with self.assertRaises(Exception):
            strict(blank)

    def test_apply_to_encoder_prepends_transform_and_suffixes_name(self):
        calls = []
        encoder = SimpleNamespace(enc_name="uni_v1", eval_transforms=lambda img: calls.append(img.size) or "features")
        normalizer = StainNormalizer("macenko")
        apply_stain_normalization(encoder, normalizer)
        self.assertEqual(encoder.enc_name, "uni_v1_macenko")
        self.assertEqual(encoder.raw_enc_name, "uni_v1")
        self.assertEqual(encoder.eval_transforms(he_like_patch()), "features")
        self.assertEqual(calls, [(96, 96)])
        self.assertEqual(normalizer.calls, 1)
        with self.assertRaisesRegex(RuntimeError, "already attached"):
            apply_stain_normalization(encoder, normalizer)


if __name__ == "__main__":
    unittest.main()
