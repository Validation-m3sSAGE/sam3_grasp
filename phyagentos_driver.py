"""
SAM3/phyagentos_driver.py

PhyAgentOS HAL driver for the SAM3 + Agilex Piper grasping system.

This file is the single integration point between PhyAgentOS and the SAM3
project.  It implements the ``BaseDriver`` contract so that the HAL Watchdog
can load it by name and route ``execute_action`` calls to the real hardware.

Two operating modes
-------------------
``mode="local"``  (default when running on the robot machine)
    Imports ``GraspManager`` directly from the SAM3 package.
    Requires: piper_sdk, pyrealsense2, sam3 model weights, CAN bus active.

``mode="remote"``  (when PhyAgentOS runs on a separate host)
    Calls the lightweight HTTP bridge (``phyagentos_bridge_server.py``) that
    runs on the robot machine.
    Requires: bridge server reachable at ``bridge_url``.

Plugin installation (one-time, on the PhyAgentOS host)
-------------------------------------------------------
    python -c "
    from hal.plugins import register_plugin
    register_plugin('/home/ubuntu/sunqianran/SAM3')
    "

Then start the watchdog:
    python -m hal.hal_watchdog --driver sam3_piper [--workspace <path>]

Or in remote mode:
    python -m hal.hal_watchdog --driver sam3_piper \\
        --driver-config /path/to/sam3_driver_config.json

where sam3_driver_config.json contains:
    {"mode": "remote", "bridge_url": "http://192.168.1.50:18791"}
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── profile path ──────────────────────────────────────────────────────────────
_PROFILE_PATH = Path(__file__).resolve().parent / "profiles" / "sam3_piper.md"

# ── BaseDriver import — works whether installed as plugin or in-tree ──────────
try:
    from hal.base_driver import BaseDriver
except ImportError:
    # Fallback: add PhyAgentOS to sys.path if running standalone
    _phy_root = Path(__file__).resolve().parent.parent / "PhyAgentOS"
    if _phy_root.exists() and str(_phy_root) not in sys.path:
        sys.path.insert(0, str(_phy_root.parent))
    from hal.base_driver import BaseDriver  # type: ignore[no-redef]


# ── constants ─────────────────────────────────────────────────────────────────
ROBOT_ID = "sam3_piper_001"
_DEFAULT_BRIDGE_URL = "http://localhost:18791"
_HTTP_TIMEOUT = 60  # seconds — SAM3 inference can take up to 2 s per call


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ── remote HTTP helpers ───────────────────────────────────────────────────────

def _http_post(url: str, body: dict, timeout: int = _HTTP_TIMEOUT) -> dict:
    """POST JSON to *url* and return the parsed response dict."""
    data = json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _http_get(url: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ── local GraspManager wrapper ────────────────────────────────────────────────

class _LocalBackend:
    """Thin wrapper around GraspManager for in-process use."""

    def __init__(self, calib_path: str | None = None):
        # Import lazily so the driver module can be imported without hardware
        from grasp_manager import GraspManager  # type: ignore[import-untyped]
        cp = calib_path or str(Path(__file__).parent / "calibration_result.npz")
        self._mgr = GraspManager(calib_path=cp)

    def dispatch(self, action_type: str, params: dict) -> dict:
        mgr = self._mgr
        try:
            if action_type == "grasp":
                r = mgr.grasp_simple(params.get("target_name", "apple"))
                return {"success": r.get("success", False), "result": str(r), "error": None}
            if action_type == "explore_and_grasp":
                r = mgr.explore_and_grasp(params.get("target_name", "apple"))
                return {"success": r.get("success", False), "result": str(r), "error": None}
            if action_type == "explore_and_place":
                r = mgr.explore_and_place(params.get("target_name", "trash can"))
                return {"success": r.get("success", False), "result": str(r), "error": None}
            if action_type == "explore_right_and_place":
                r = mgr.explore_right_and_place(params.get("target_name", "trash can"))
                return {"success": r.get("success", False), "result": str(r), "error": None}
            if action_type == "check_target":
                target = params.get("target_name", "apple")
                r = mgr.check_target_in_scene(target)
                found = r.get("found", False)
                return {
                    "success": True,
                    "result": f"Target '{target}' {'found' if found else 'not found'} in scene.",
                    "error": None,
                }
            if action_type == "go_home":
                mgr.go_standby()
                return {"success": True, "result": "Arm returned to standby pose.", "error": None}
            if action_type == "calibrate":
                mgr.calibrate_axes()
                return {"success": True, "result": "Hand-eye calibration completed.", "error": None}
            if action_type == "connect":
                mgr.arm.enable_arm(True)
                return {"success": True, "result": "Arm enabled via CAN.", "error": None}
            if action_type == "disconnect":
                mgr.arm.enable_arm(False)
                return {"success": True, "result": "Arm disabled.", "error": None}
            return {"success": False, "result": "", "error": f"Unknown action: {action_type!r}"}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "result": "", "error": str(exc)}

    def health_check(self) -> bool:
        try:
            return bool(self._mgr.arm)
        except Exception:
            return False

    def close(self) -> None:
        try:
            self._mgr.shutdown()
        except Exception:
            pass


# ── remote HTTP backend ───────────────────────────────────────────────────────

class _RemoteBackend:
    """Calls the phyagentos_bridge_server over HTTP."""

    def __init__(self, bridge_url: str):
        self._url = bridge_url.rstrip("/")

    def dispatch(self, action_type: str, params: dict) -> dict:
        try:
            return _http_post(
                f"{self._url}/action",
                {"action_type": action_type, "params": params},
            )
        except urllib.error.URLError as exc:
            return {"success": False, "result": "", "error": f"Bridge unreachable: {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "result": "", "error": str(exc)}

    def health_check(self) -> bool:
        try:
            r = _http_get(f"{self._url}/health", timeout=5)
            return r.get("status") == "ok"
        except Exception:
            return False

    def close(self) -> None:
        pass  # stateless HTTP — nothing to close


# ── SAM3PiperDriver ───────────────────────────────────────────────────────────

class SAM3PiperDriver(BaseDriver):
    """PhyAgentOS HAL driver for the SAM3 + Agilex Piper grasping system.

    Parameters
    ----------
    mode:
        ``"local"`` — import GraspManager directly (default).
        ``"remote"`` — call the HTTP bridge server.
    bridge_url:
        URL of the bridge server (only used when ``mode="remote"``).
        Default: ``"http://localhost:18791"``.
    calib_path:
        Path to ``calibration_result.npz`` (local mode only).
    gui:
        Accepted for API compatibility; ignored.
    """

    def __init__(
        self,
        mode: str = "local",
        bridge_url: str = _DEFAULT_BRIDGE_URL,
        calib_path: str | None = None,
        gui: bool = False,
        **_kwargs: Any,
    ) -> None:
        self._mode = mode
        self._connected = False
        self._objects: dict[str, dict] = {}
        self._arm_state: dict[str, Any] = {
            "mode": "idle",
            "last_action": None,
            "last_result": None,
            "last_error": None,
            "updated_at": _utcnow(),
        }
        self._connection_state: dict[str, Any] = {
            "status": "disconnected",
            "mode": mode,
            "bridge_url": bridge_url if mode == "remote" else "local",
            "last_heartbeat": None,
            "last_error": None,
        }

        if mode == "remote":
            self._backend: _LocalBackend | _RemoteBackend = _RemoteBackend(bridge_url)
        else:
            self._backend = _LocalBackend(calib_path=calib_path)

    # ── BaseDriver — required ─────────────────────────────────────────────────

    def get_profile_path(self) -> Path:
        return _PROFILE_PATH

    def load_scene(self, scene: dict[str, dict]) -> None:
        """Store the initial scene from ENVIRONMENT.md (objects dict)."""
        self._objects = dict(scene)

    def execute_action(self, action_type: str, params: dict) -> str:
        """Dispatch an action to the backend and return a result string."""
        self._arm_state["mode"] = "executing"
        self._arm_state["last_action"] = action_type
        self._arm_state["updated_at"] = _utcnow()

        result = self._backend.dispatch(action_type, params)

        success = result.get("success", False)
        result_msg = result.get("result", "")
        error_msg = result.get("error")

        self._arm_state["mode"] = "idle"
        self._arm_state["last_result"] = result_msg
        self._arm_state["last_error"] = error_msg
        self._arm_state["updated_at"] = _utcnow()

        if success:
            return result_msg or f"Action '{action_type}' completed successfully."
        else:
            raise RuntimeError(
                error_msg or f"Action '{action_type}' failed (no error message)."
            )

    def get_scene(self) -> dict[str, dict]:
        """Return the current objects dict (unchanged by arm actions)."""
        return dict(self._objects)

    # ── BaseDriver — optional ─────────────────────────────────────────────────

    def connect(self) -> bool:
        result = self._backend.dispatch("connect", {})
        ok = result.get("success", False)
        self._connection_state.update({
            "status": "connected" if ok else "degraded",
            "last_heartbeat": _utcnow(),
            "last_error": result.get("error") if not ok else None,
        })
        self._connected = ok
        return ok

    def disconnect(self) -> None:
        self._backend.dispatch("disconnect", {})
        self._connection_state.update({
            "status": "disconnected",
            "last_heartbeat": _utcnow(),
            "last_error": None,
        })
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def health_check(self) -> bool:
        ok = self._backend.health_check()
        self._connection_state["last_heartbeat"] = _utcnow()
        if not ok:
            self._connection_state["status"] = "degraded"
            self._connection_state["last_error"] = "health_check_failed"
        else:
            self._connection_state["status"] = "connected" if self._connected else "disconnected"
            self._connection_state["last_error"] = None
        return ok

    def get_runtime_state(self) -> dict[str, Any]:
        return {
            "robots": {
                ROBOT_ID: {
                    "robot_pose": {
                        "frame": "arm_base",
                        "note": "joint-space robot; no global pose",
                        "stamp": _utcnow(),
                    },
                    "arm_state": dict(self._arm_state),
                    "connection_state": dict(self._connection_state),
                }
            }
        }

    def close(self) -> None:
        self._backend.close()
        self._connected = False
