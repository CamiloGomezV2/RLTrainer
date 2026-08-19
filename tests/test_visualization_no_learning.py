"""Verify Visualization actions do not modify a zero-initialized Q-table."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app import app  # noqa: E402


CONFIG = {
    "environment": "CliffWalking-v0",
    "agent": "Q_learning",
    "learning_rate": 0.5,
    "exploration_probability": 0.2,
    "discount_factor": 0.99,
}


class CliffWalkingVisualizationNoLearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()
        response = self.client.post("/api/config", json=CONFIG)
        self.assertEqual(response.status_code, 200, response.get_json())
        payload = self.client.get("/api/agent/q-table").get_json()
        q_table = np.asarray(payload["q_table"], dtype=float)
        np.testing.assert_allclose(
            q_table,
            0,
            atol=0,
            err_msg="Q-table must start at 0 after applying configuration",
        )

    def _q_table(self) -> np.ndarray:
        payload = self.client.get("/api/agent/q-table").get_json()
        return np.asarray(payload["q_table"], dtype=float)

    def test_run_an_action_keeps_q_table_at_zero(self) -> None:
        for action in (0, 1, 2, 3, "policy"):
            response = self.client.post(
                "/api/environment/run-action",
                json={"action": action},
            )
            self.assertEqual(response.status_code, 200, response.get_json())
            np.testing.assert_allclose(
                self._q_table(),
                0,
                atol=0,
                err_msg=f"Q-table is not all zeros after Run an action ({action})",
            )

    def test_run_episode_keeps_q_table_at_zero(self) -> None:
        response = self.client.post(
            "/api/environment/run-episode",
            json={"action": "policy"},
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        np.testing.assert_allclose(
            self._q_table(),
            0,
            atol=0,
            err_msg="Q-table is not all zeros after Run episode (policy)",
        )

        response = self.client.post(
            "/api/environment/run-episode",
            json={"action": 1},
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        np.testing.assert_allclose(
            self._q_table(),
            0,
            atol=0,
            err_msg="Q-table is not all zeros after Run episode (fixed action)",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
