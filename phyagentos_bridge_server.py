#!/usr/bin/env python3
"""
SAM3/phyagentos_bridge_server.py

Lightweight HTTP bridge that exposes GraspManager operations as a REST API.

This server runs ON the machine where the Piper arm and RealSense camera are
physically connected (i.e. the same machine that runs mcp_server.py).

The PhyAgentOS HAL driver (SAM3PiperDriver) calls this server when running
in ``mode="remote"``.  In ``mode="local"`` the driver imports GraspManager
directly and no bridge is needed.

Endpoints
---------
POST /action
    Body: {"action_type": "<str>", "params": {<kwargs>}}
    Response: {"success": true/false, "result": "<str>", "error": "<str|null>"}

GET /health
    Response: {"status": "ok", "arm_enabled": true/false}

Usage
-----
    # On the robot machine (inside SAM3/ directory):
    python phyagentos_bridge_server.py [--port 18791]

    # The driver on the PhyAgentOS host then connects to:
    #   http://<robot-ip>:18791
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── redirect print to stderr so logs don't corrupt HTTP responses ─────────────
import builtins
_orig_print = builtins.print
def _stderr_print(*args, **kwargs):
    kwargs["file"] = sys.stderr
    _orig_print(*args, **kwargs)
builtins.print = _stderr_print

# ── lazy GraspManager singleton ───────────────────────────────────────────────
_manager = None

def _get_manager():
    global _manager
    if _manager is None:
        # Import here so the module can be imported without hardware present
        from grasp_manager import GraspManager
        calib_path = str(Path(__file__).parent / "calibration_result.npz")
        _manager = GraspManager(calib_path=calib_path)
    return _manager


# ── action dispatch table ─────────────────────────────────────────────────────

def _dispatch(action_type: str, params: dict) -> dict:
    """Route action_type to the correct GraspManager method.

    Returns a dict with keys: success (bool), result (str), error (str|None).
    """
    mgr = _get_manager()

    try:
        if action_type == "grasp":
            target = params.get("target_name", "apple")
            r = mgr.grasp_simple(target)
            return {"success": r.get("success", False), "result": str(r), "error": None}

        if action_type == "explore_and_grasp":
            target = params.get("target_name", "apple")
            r = mgr.explore_and_grasp(target)
            return {"success": r.get("success", False), "result": str(r), "error": None}

        if action_type == "explore_and_place":
            target = params.get("target_name", "trash can")
            r = mgr.explore_and_place(target)
            return {"success": r.get("success", False), "result": str(r), "error": None}

        if action_type == "explore_right_and_place":
            target = params.get("target_name", "trash can")
            r = mgr.explore_right_and_place(target)
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

        return {
            "success": False,
            "result": "",
            "error": f"Unknown action_type: {action_type!r}",
        }

    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        _stderr_print(f"[Bridge] Exception in {action_type}:\n{tb}")
        return {"success": False, "result": "", "error": tb.splitlines()[-1]}


# ── HTTP handler ──────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # suppress default access log
        _stderr_print(f"[Bridge] {self.address_string()} - {fmt % args}")

    def _send_json(self, code: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            try:
                mgr = _get_manager()
                arm_enabled = bool(getattr(mgr.arm, "_enabled", True))
            except Exception:
                arm_enabled = False
            self._send_json(200, {"status": "ok", "arm_enabled": arm_enabled})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/action":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"invalid JSON: {exc}"})
            return

        action_type = body.get("action_type", "")
        params = body.get("params", {})
        if not action_type:
            self._send_json(400, {"error": "action_type is required"})
            return

        _stderr_print(f"[Bridge] → {action_type}  params={params}")
        result = _dispatch(action_type, params)
        _stderr_print(f"[Bridge] ← {result}")
        self._send_json(200, result)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhyAgentOS HTTP bridge for SAM3 + Piper grasping system"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=18791, help="Bind port (default: 18791)")
    args = parser.parse_args()

    _stderr_print(f"[Bridge] Starting PhyAgentOS bridge on {args.host}:{args.port}")
    _stderr_print("[Bridge] Initialising GraspManager (this may take a few seconds)...")
    _get_manager()  # eager init so first request is fast
    _stderr_print("[Bridge] Ready.")

    server = HTTPServer((args.host, args.port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _stderr_print("[Bridge] Shutdown.")


if __name__ == "__main__":
    main()
