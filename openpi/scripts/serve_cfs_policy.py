"""Serve the opt-in constrained-flow comparison adapter.

This is a separate executable so the ordinary policy server and its default
requests never install experiment hooks.  Policy construction, command-line
arguments, recording, and WebSocket transport continue to come from the
baseline server.
"""

from __future__ import annotations

import logging
from pathlib import Path
import socket
import sys

import tyro

# ``python scripts/serve_cfs_policy.py`` puts ``scripts/`` rather than the
# OpenPI project root on ``sys.path``.  Add the parent only for this executable
# so the existing ``scripts.serve_policy`` module is importable in both direct
# and package-style launches.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import serve_policy as _serve_policy  # noqa: E402

from openpi.policies import policy as _policy  # noqa: E402
from openpi.policies import crfs_constrained_flow_adapter as _cfs_adapter  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402


def main(args: _serve_policy.Args) -> None:
    """Create an ordinary policy, then wrap it in the isolated CFS adapter."""

    policy = _serve_policy.create_policy(args)
    policy_metadata = policy.metadata

    _cfs_adapter.install_constrained_flow_hooks()
    policy = _cfs_adapter.ConstrainedFlowPolicyAdapter(policy)

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating CFS server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(_serve_policy.Args))
