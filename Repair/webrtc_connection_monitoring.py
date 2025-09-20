"""
Safe WebRTC Connection Monitoring
Adds /webrtc/connections endpoint without breaking existing auth
Compatible with existing single-peer architecture
"""

from fastapi import APIRouter
from typing import Dict, Any
import time

# This can be added to your existing webrtc_server.py

def add_connection_monitoring(router: APIRouter, peer_state_ref):
    """Add connection monitoring endpoint to existing router"""

    @router.get("/connections")
    async def get_connection_info():
        """Get current connection information"""

        # Work with existing single peer state
        if peer_state_ref is None:
            return {
                "active_connections": 0,
                "status": "no_active_connection"
            }

        try:
            # Extract info from existing peer state structure
            connection_info = {
                "active_connections": 1,
                "status": "connected",
                "connection_state": getattr(peer_state_ref.pc, 'connectionState', 'unknown'),
                "uptime_seconds": time.time() - peer_state_ref.created if hasattr(peer_state_ref, 'created') else 0,
                "ice_connection_state": getattr(peer_state_ref.pc, 'iceConnectionState', 'unknown'),
                "control_channel_ready": getattr(peer_state_ref, 'control_channel_ready', False)
            }

            return connection_info

        except Exception as e:
            return {
                "active_connections": 0,
                "status": "error",
                "error": str(e)
            }

# Usage in your existing webrtc_server.py:
# add_connection_monitoring(router, _peer_state)
