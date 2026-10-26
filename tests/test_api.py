from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from ears import EARSConfig, EARSSelector, SelectionResult
from ears.selection import select_peaks
from ears.signals import candidate_indices, evidence_scores, normalize
from ears.video import VideoInfo


class SignalTests(unittest.TestCase):
    def test_candidates_cover_the_full_video(self):
        indices = candidate_indices(100_000, 256)
        self.assertEqual((indices[0], indices[-1], len(indices)), (0, 99_999, 256))
        self.assertTrue(np.all(np.diff(indices) > 0))

    def test_candidates_reject_invalid_inputs(self):
        with self.assertRaises(ValueError):
            candidate_indices(0, 256)
        with self.assertRaises(ValueError):
            candidate_indices(100, 1)

    def test_normalize_handles_constant_signal(self):
        np.testing.assert_array_equal(normalize(np.ones(3)), np.zeros(3))

    def test_evidence_scores_match_sigmoid_fusion(self):
        audio = np.array([0.0, 1.0])
        motion = np.array([1.0, 0.0])
        expected = np.full(2, 1 / (1 + np.exp(-1)))
        np.testing.assert_allclose(evidence_scores(audio, motion), expected)

    def test_evidence_scores_reject_mismatched_signals(self):
        with self.assertRaises(ValueError):
            evidence_scores(np.zeros(2), np.zeros(3))


class SelectionTests(unittest.TestCase):
    def test_temporal_gap_separates_peaks(self):
        scores = np.array([0.1, 0.9, 0.8, 0.1, 0.7])
        np.testing.assert_array_equal(select_peaks(scores, 2, 1), [1, 4])

    def test_selection_backfills_to_exact_k(self):
        scores = np.array([0.9, 0.8, 0.7])
        result = select_peaks(scores, 3, 10)
        np.testing.assert_array_equal(result, [0, 1, 2])

    def test_ties_prefer_earlier_frames(self):
        scores = np.ones(4)
        np.testing.assert_array_equal(select_peaks(scores, 2, 0), [0, 1])

    def test_selection_rejects_nonpositive_k(self):
        with self.assertRaises(ValueError):
            select_peaks(np.ones(3), 0, 0)


class ApiTests(unittest.TestCase):
    def test_config_validation(self):
        with self.assertRaises(ValueError):
            EARSConfig(motion_candidates=1).validate()
        with self.assertRaises(ValueError):
            EARSConfig(flow_scale=2).validate()

    @patch("ears.api.read_frames")
    @patch("ears.api.audio_signal")
    @patch("ears.api.motion_signal")
    @patch("ears.api.probe_video")
    def test_select_returns_indices_timestamps_scores_and_frames(
        self,
        probe: Mock,
        motion: Mock,
        audio: Mock,
        read: Mock,
    ):
        probe.return_value = VideoInfo(5, 2.0, 8, 8)
        motion.return_value = np.array([0.0, 1.0, 0.0, 0.5, 0.0])
        audio.return_value = np.zeros(5)
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        read.return_value = {1: frame, 3: frame}

        result = EARSSelector().select("video.mp4", 2, return_frames=True)

        self.assertEqual(result.frame_indices, (1, 3))
        self.assertEqual(result.timestamps, (0.5, 1.5))
        self.assertEqual(len(result.scores), 2)
        self.assertEqual(len(result.frames or ()), 2)

    def test_result_to_dict_omits_numpy_frames(self):
        result = SelectionResult(
            "video.mp4", (1,), (0.5,), (0.7,), 2.0, 10,
            (np.zeros((2, 2, 3), dtype=np.uint8),),
        )
        self.assertNotIn("frames", result.to_dict())

    @patch("ears.api.cv2.imwrite", return_value=True)
    def test_result_saves_rgb_frames(self, write: Mock):
        result = SelectionResult(
            "video.mp4", (3,), (1.5,), (0.7,), 2.0, 10,
            (np.zeros((2, 2, 3), dtype=np.uint8),),
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = result.save_frames(directory)
        self.assertEqual(paths[0].name, "frame_00000003.jpg")
        write.assert_called_once()

    def test_save_requires_returned_frames(self):
        result = SelectionResult("video.mp4", (1,), (0.5,), (0.7,), 2.0, 10)
        with self.assertRaises(ValueError):
            result.save_frames("unused")

    @patch("ears.signals.decode_audio", return_value=np.zeros(8_000, dtype=np.float32))
    def test_end_to_end_video_selection(self, _decode_audio: Mock):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "moving.avi"
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (32, 32),
            )
            self.assertTrue(writer.isOpened())
            for index in range(20):
                frame = np.zeros((32, 32, 3), dtype=np.uint8)
                frame[8:16, index:index + 8] = 255
                writer.write(frame)
            writer.release()

            result = EARSSelector(
                EARSConfig(motion_candidates=10, flow_scale=1.0)
            ).select(video, 3, return_frames=True)

        self.assertEqual(len(result.frame_indices), 3)
        self.assertEqual(result.frame_indices, tuple(sorted(result.frame_indices)))
        self.assertEqual(len(result.frames or ()), 3)


if __name__ == "__main__":
    unittest.main()
