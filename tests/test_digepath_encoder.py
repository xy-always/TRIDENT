import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from trident.patch_encoder_models import (
    DigePathInferenceEncoder,
    encoder_factory,
    encoder_registry,
)


class _DummyDigePath(torch.nn.Module):
    def forward(self, images):
        return torch.zeros((images.shape[0], 1024), device=images.device)


class TestDigePathEncoder(unittest.TestCase):
    def test_registered_with_factory(self):
        self.assertIs(encoder_registry["digepath"], DigePathInferenceEncoder)

    @patch("timm.create_model", return_value=_DummyDigePath())
    def test_local_safetensors_load_and_forward(self, create_model):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            checkpoint.touch()

            encoder = encoder_factory("digepath", weights_path=str(checkpoint))

        create_model.assert_called_once_with(
            "vit_large_patch16_224",
            pretrained=False,
            checkpoint_path=str(checkpoint),
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        self.assertEqual(encoder.enc_name, "digepath")
        self.assertEqual(encoder.embedding_dim, 1024)
        self.assertEqual(encoder.precision, torch.float32)

        image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        transformed = encoder.eval_transforms(image)
        self.assertEqual(tuple(transformed.shape), (3, 224, 224))
        self.assertEqual(tuple(encoder(torch.zeros(2, 3, 224, 224)).shape), (2, 1024))


if __name__ == "__main__":
    unittest.main()
