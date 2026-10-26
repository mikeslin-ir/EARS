from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ears.api import SelectionResult
from ears.cli import main


class CliTests(unittest.TestCase):
    @patch("ears.cli.EARSSelector")
    def test_cli_writes_json(self, selector_type: Mock):
        selector_type.return_value.select.return_value = SelectionResult(
            "video.mp4", (2, 8), (0.2, 0.8), (0.7, 0.9), 10.0, 20
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "selection.json"
            self.assertEqual(main(["video.mp4", "-k", "2", "--json", str(output)]), 0)
            payload = json.loads(output.read_text())
        self.assertEqual(payload["frame_indices"], [2, 8])
        selector_type.return_value.select.assert_called_once_with(
            Path("video.mp4"), 2, return_frames=False
        )

    @patch("ears.cli.EARSSelector")
    def test_cli_saves_frames_when_requested(self, selector_type: Mock):
        result = Mock()
        result.to_dict.return_value = {"frame_indices": [2]}
        result.save_frames.return_value = [Path("frames/frame_00000002.jpg")]
        selector_type.return_value.select.return_value = result

        self.assertEqual(
            main(["video.mp4", "-k", "1", "--frames-dir", "frames"]),
            0,
        )
        selector_type.return_value.select.assert_called_once_with(
            Path("video.mp4"), 1, return_frames=True
        )
        result.save_frames.assert_called_once_with(Path("frames"))


if __name__ == "__main__":
    unittest.main()
