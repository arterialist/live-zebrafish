"""The default backend port must match the documented Vite proxy."""

import unittest
from unittest.mock import patch

from lab import server


class LabLaunchTests(unittest.TestCase):
    def launch(self, arguments):
        app = object()
        with patch("sys.argv", ["zebrafish-lab-server", *arguments]), \
             patch.object(server, "_configure_logging"), \
             patch.object(server, "build_app", return_value=(app, None, None)), \
             patch.object(server.uvicorn, "run") as run:
            server.main()
        return app, run

    def test_default_port_matches_local_proxy(self):
        app, run = self.launch([])
        run.assert_called_once_with(app, host="127.0.0.1", port=8765, log_level="info")

    def test_explicit_port_is_preserved(self):
        app, run = self.launch(["--port", "18811"])
        run.assert_called_once_with(app, host="127.0.0.1", port=18811, log_level="info")
