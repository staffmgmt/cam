"""
Safe WebRTC Connection Monitoring
Adds /webrtc/connections endpoint without breaking existing auth
Compatible with existing single-peer architecture
"""

from fastapi import APIRouter
import time


def add_connection_monitoring(router: APIRouter, peer_state_getter):
    @router.get("/connections")
    async def get_connection_info():
        try:
            state = None
            try:
                state = peer_state_getter() if callable(peer_state_getter) else None
            except Exception:
                state = None
            if state is None:
                return {"active_connections": 0, "status": "no_active_connection"}
            info = {
                "active_connections": 1,
                "status": "connected",
                "connection_state": getattr(state, 'pc', None) and getattr(state.pc, 'connectionState', 'unknown'),
                "uptime_seconds": time.time() - getattr(state, 'created', time.time()),
                "ice_connection_state": getattr(state, 'pc', None) and getattr(state.pc, 'iceConnectionState', 'unknown'),
                "control_channel_ready": getattr(state, 'control_channel_ready', False)
            }
            return info
        except Exception as e:
            return {"active_connections": 0, "status": "error", "error": str(e)}
