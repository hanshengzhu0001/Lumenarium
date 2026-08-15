import tempfile
from pathlib import Path
import unittest

try:
    import torch
except (ImportError, OSError):  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required")
class ProcessedMaskCompactorTest(unittest.TestCase):
    def setUp(self):
        from tools import compact_processed_masks as compactor

        self.compactor = compactor

    def test_compacts_aliased_rgba_storage_without_changing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processed_masks.pt"
            rgba = torch.arange(
                3 * 4 * 5 * 6, dtype=torch.float32
            ).reshape(3, 4, 5, 6)
            mask_view = rgba[:, 3]
            self.assertGreater(
                self.compactor.storage_nbytes(mask_view),
                self.compactor.logical_nbytes(mask_view),
            )
            torch.save(mask_view, path)

            result = self.compactor.compact_one(path)
            reloaded = torch.load(path, map_location="cpu")
            self.assertEqual(result["status"], "compacted")
            self.assertTrue(torch.equal(mask_view, reloaded))
            self.assertEqual(
                self.compactor.storage_nbytes(reloaded),
                self.compactor.logical_nbytes(reloaded),
            )
            self.assertGreater(result["reclaimed_bytes"], 0)

    def test_compact_file_is_idempotently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processed_masks.pt"
            tensor = torch.rand(2, 3, 4).contiguous()
            torch.save(tensor, path)
            result = self.compactor.compact_one(path)
            self.assertEqual(result["status"], "already_compact")
            self.assertEqual(result["reclaimed_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
