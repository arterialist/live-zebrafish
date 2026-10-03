"""Optional local video calibration must not prevent ordinary lab startup."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from lab import rest_routes, server, video_pipeline
from lab.rest_routes import BackendVideoFrameBody
from lab.sim_runtime import LabSimRuntime


class VideoCacheAvailabilityTests(unittest.TestCase):
    def test_empty_cache_allows_pipeline_construction_and_upload_paths(self):
        with TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            uploads = cache / "video_stimuli" / "uploads"
            pipeline = video_pipeline.BackendVideoPipeline(
                cache_root=cache, upload_root=uploads
            )
            self.assertTrue(uploads.is_dir())
            self.assertEqual(pipeline.uploaded_path("clip.mp4").parent, uploads)
            self.assertFalse((cache / "zapbench").exists())

    def test_missing_cache_blocks_extraction_before_video_or_action_processing(self):
        for missing in ("stimulus", "labels"):
            with self.subTest(missing=missing), TemporaryDirectory() as temporary:
                cache = Path(temporary) / "cache"
                if missing == "labels":
                    stimulus = cache / "zapbench" / "zapbench_stimulus_features.npz"
                    stimulus.parent.mkdir(parents=True)
                    stimulus.touch()
                pipeline = video_pipeline.BackendVideoPipeline(
                    cache_root=cache, upload_root=cache / "uploads"
                )
                with patch.object(video_pipeline, "_read_frame") as read_frame, \
                     patch.object(pipeline._simzfish, "action_from_frame") as action:
                    with self.assertRaises(
                        video_pipeline.VideoCalibrationUnavailableError
                    ) as error:
                        pipeline.extract(
                            path=cache / "clip.mp4", file_name="clip.mp4",
                            frame_index=0, video_time_s=0.0, sample_hz=15.0,
                        )
                    self.assertIn("missing ZAPBench", str(error.exception))
                    read_frame.assert_not_called()
                    action.assert_not_called()

    def test_lab_health_and_transport_work_while_calibrated_video_returns_503(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = MagicMock(spec=LabSimRuntime)
            runtime.transport_snapshot.return_value = {"tick": 0, "playing": False}
            with patch.object(rest_routes, "_repo_root", return_value=root), \
                 patch.object(server, "LabSimRuntime", return_value=runtime):
                app, context, thread = server.build_app()
                self.assertFalse(thread.is_alive())
                endpoints = {
                    (route.path, method): route.endpoint
                    for route in app.routes
                    for method in getattr(route, "methods", ())
                }
                self.assertEqual(endpoints[("/api/health", "GET")](), {"ok": True, "tick": 0})
                self.assertEqual(
                    endpoints[("/api/sim/transport", "GET")](),
                    {"tick": 0, "playing": False},
                )
                video = context.video_pipeline.upload_root / "clip.mp4"
                video.touch()
                with self.assertRaises(HTTPException) as error:
                    endpoints[("/api/video-stimulus/backend-frame", "POST")](
                        BackendVideoFrameBody(file_name=video.name)
                    )
                self.assertEqual(error.exception.status_code, 503)
                self.assertIn("missing ZAPBench stimulus cache", error.exception.detail)
                runtime.push_video_stimulus_frame_and_step.assert_not_called()
                runtime.run_loop.assert_not_called()
